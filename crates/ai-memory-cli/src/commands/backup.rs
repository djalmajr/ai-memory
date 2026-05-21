//! `ai-memory backup --to <tarball>` — make a consistent point-in-time
//! snapshot of the data dir.
//!
//! Uses SQLite's online backup API so the source DB stays writable for
//! the duration of the copy. The tarball is gzip-compressed and
//! contains `wiki/`, a fresh `db/memory.sqlite` snapshot, and the
//! `config.toml` (if present).

use std::path::Path;

use ai_memory_store::Store;
use anyhow::{Context, Result, bail};
use flate2::Compression;
use flate2::write::GzEncoder;
use tracing::info;

use crate::cli::BackupArgs;
use crate::config::Config;
use crate::process_guard::{busy_message, sibling_processes};

/// Run the `backup` subcommand.
///
/// # Errors
/// Returns an error if another `ai-memory` process is running, the
/// store cannot be opened, the SQLite snapshot fails, or the tarball
/// cannot be written.
pub async fn run(config: &Config, args: BackupArgs) -> Result<()> {
    let siblings = sibling_processes();
    if !siblings.is_empty() {
        bail!(busy_message("backup", &siblings));
    }

    let store = Store::open(&config.data_dir)
        .with_context(|| format!("opening store at {}", config.data_dir.display()))?;

    let staging = tempfile::tempdir()?;
    let snapshot_path = staging.path().join("memory.sqlite");
    info!(snapshot = %snapshot_path.display(), "snapshotting SQLite");
    store
        .reader
        .snapshot_to(snapshot_path.clone())
        .await
        .context("running SQLite online backup")?;

    let dest = &args.to;
    if let Some(parent) = dest.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let file =
        std::fs::File::create(dest).with_context(|| format!("creating {}", dest.display()))?;
    let encoder = GzEncoder::new(file, Compression::default());
    let mut tar = tar::Builder::new(encoder);
    tar.mode(tar::HeaderMode::Deterministic);

    // wiki/ goes in as-is.
    let wiki_dir = config.data_dir.join("wiki");
    if wiki_dir.is_dir() {
        tar.append_dir_all("wiki", &wiki_dir)
            .with_context(|| format!("archiving {}", wiki_dir.display()))?;
    }

    // db/memory.sqlite from the snapshot, not the live file.
    tar.append_path_with_name(&snapshot_path, "db/memory.sqlite")
        .context("archiving db snapshot")?;

    // config.toml, if present.
    let cfg = config.data_dir.join("config.toml");
    if cfg.is_file() {
        tar.append_path_with_name(&cfg, "config.toml")
            .context("archiving config.toml")?;
    }

    let encoder = tar.into_inner()?;
    encoder.finish()?;

    let size = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
    info!(path = %dest.display(), bytes = size, "backup written");
    println!("backup -> {} ({})", dest.display(), human_bytes(size));
    Ok(())
}

fn human_bytes(n: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB"];
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{n} {}", UNITS[0])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

#[allow(dead_code)] // Reserved for cross-platform diagnostics later.
fn rel_path<'a>(base: &Path, full: &'a Path) -> Option<&'a Path> {
    full.strip_prefix(base).ok()
}

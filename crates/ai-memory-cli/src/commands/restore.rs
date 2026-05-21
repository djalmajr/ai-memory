//! `ai-memory restore --from <tarball>` — restore a backup tarball.
//!
//! Refuses to overwrite a non-empty data dir unless `--force` is given.
//! Refuses while another `ai-memory` process is alive. After extraction,
//! re-opens the store so any pending migrations run (and a corrupt
//! snapshot fails loudly).

use ai_memory_store::Store;
use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use tracing::info;

use crate::cli::RestoreArgs;
use crate::config::Config;
use crate::process_guard::{busy_message, sibling_processes};

/// Run the `restore` subcommand.
///
/// # Errors
/// Returns an error if another `ai-memory` process is running, the
/// data dir is non-empty without `--force`, the tarball cannot be
/// extracted, or the restored store fails to open.
pub fn run(config: &Config, args: RestoreArgs) -> Result<()> {
    let siblings = sibling_processes();
    if !siblings.is_empty() {
        bail!(busy_message("restore", &siblings));
    }

    if !args.from.is_file() {
        bail!("source tarball {} not found", args.from.display());
    }

    let wiki = config.data_dir.join("wiki");
    let db = config.data_dir.join("db").join("memory.sqlite");
    if (wiki.is_dir() && std::fs::read_dir(&wiki)?.next().is_some()) || db.is_file() {
        if !args.force {
            bail!(
                "refusing to restore: data dir at {} is non-empty (pass --force to overwrite)",
                config.data_dir.display(),
            );
        }
        // Force path: drop the existing wiki + db so the tarball can
        // populate them cleanly. Keep config.toml, logs/, models/.
        for sub in ["wiki", "db"] {
            let path = config.data_dir.join(sub);
            if path.exists() {
                std::fs::remove_dir_all(&path)?;
            }
        }
    }
    std::fs::create_dir_all(&config.data_dir)?;

    let file = std::fs::File::open(&args.from)
        .with_context(|| format!("opening {}", args.from.display()))?;
    let decoder = GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    archive.set_preserve_permissions(false);
    archive
        .unpack(&config.data_dir)
        .with_context(|| format!("extracting into {}", config.data_dir.display()))?;
    info!(from = %args.from.display(), into = %config.data_dir.display(), "tarball extracted");

    // Open + drop the store so refinery applies any pending migrations
    // and the SQLite file is validated.
    let _store = Store::open(&config.data_dir).context("opening restored store")?;
    info!("restore complete");
    println!(
        "restored {} -> {}",
        args.from.display(),
        config.data_dir.display()
    );
    Ok(())
}

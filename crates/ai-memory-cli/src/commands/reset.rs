//! `ai-memory reset --confirm` — wipe wiki/, db/, raw/ contents.
//!
//! Refuses to run while another `ai-memory` process is alive (lesson from
//! basic-memory #765, where a zombie process holding the old SQLite
//! inode caused phantom search results after a reset).

use anyhow::{Result, bail};

use crate::cli::ResetArgs;
use crate::config::Config;
use crate::process_guard::{busy_message, sibling_processes};

const SUBDIRS: &[&str] = &["wiki", "db", "raw"];

/// Run the `reset` subcommand.
///
/// # Errors
/// Returns an error if another `ai-memory` process is running, if
/// `--confirm` was not provided, or if a directory cannot be removed.
pub fn run(config: &Config, args: ResetArgs) -> Result<()> {
    let siblings = sibling_processes();
    if !siblings.is_empty() {
        bail!(busy_message("reset", &siblings));
    }

    if !args.confirm {
        for sub in SUBDIRS {
            let path = config.data_dir.join(sub);
            if path.exists() {
                println!("would remove {}", path.display());
            }
        }
        println!("(dry-run; pass --confirm to wipe)");
        return Ok(());
    }

    for sub in SUBDIRS {
        let path = config.data_dir.join(sub);
        if !path.exists() {
            continue;
        }
        std::fs::remove_dir_all(&path)?;
        std::fs::create_dir_all(&path)?;
        tracing::info!(path = %path.display(), "reset");
    }
    tracing::info!("reset complete");
    Ok(())
}

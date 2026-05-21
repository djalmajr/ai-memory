//! Shared "is another ai-memory process alive?" check.
//!
//! Used by every destructive command (`reset`, `backup`, `restore`) so
//! we never race a live writer (lesson from basic-memory #765).

use std::ffi::OsStr;

use sysinfo::System;

/// Binary name to match against `/proc/*/comm` (or platform equivalent).
pub const BIN_NAME: &str = "ai-memory";

/// Return PIDs of *other* `ai-memory` processes (excluding the current
/// process and any threads of it).
#[must_use]
pub fn sibling_processes() -> Vec<sysinfo::Pid> {
    let mut sys = System::new();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    let me = sysinfo::Pid::from_u32(std::process::id());
    let bin_os: &OsStr = OsStr::new(BIN_NAME);
    sys.processes_by_exact_name(bin_os)
        // On Linux, sysinfo lists tokio worker threads alongside the main
        // process under the same comm name. thread_kind() == None means
        // we're looking at the process leader, not one of its threads.
        .filter(|p| p.thread_kind().is_none())
        .map(sysinfo::Process::pid)
        .filter(|pid| *pid != me)
        .collect()
}

/// Format a "refusing to ..." error message for the given operation,
/// quoting sibling PIDs.
#[must_use]
pub fn busy_message(verb: &str, siblings: &[sysinfo::Pid]) -> String {
    let pids: Vec<u32> = siblings.iter().copied().map(sysinfo::Pid::as_u32).collect();
    format!(
        "refusing to {}: {} other ai-memory process(es) running (pids: {:?}). \
         Stop them first, then re-run.",
        verb,
        pids.len(),
        pids,
    )
}

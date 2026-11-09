//! Persistent queue for external lifecycle events sent through `POST /hook/batch`.
//!
//! The queue is separate from ai-memory's database and wiki. Events leave it
//! after a validated acknowledgement, which may include a server policy drop.
//! Pending events remain on disk when delivery fails or their retry window
//! expires; reaching a capacity limit rejects new input.
//!
//! New bodies are sanitized before enqueueing; stored retries stay exact. Producers must
//! apply capture exclusions before enqueueing and protect the queue directory.

pub mod ack;
pub mod fsguard;
pub mod http;
pub mod identity;
pub mod queue;
pub mod relay;

/// Wall-clock milliseconds since the Unix epoch.
///
/// Used for the retry window, which is why it is clamped at 0 rather than
/// panicking on a pre-epoch clock: a nonsensical clock must not take the process
/// down mid-flush.
pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

/// Operator-facing one-line rendering of a failure: the anyhow context chain,
/// with the SQLite cause shown exactly once.
///
/// rusqlite 0.41 chains a `SqliteFailure` onto the `libsqlite3-sys` error
/// from which its own display string is derived, so the default `{:#}` chain
/// prints the SQLite message twice for many codes (e.g. `SQLITE_NOTAFILE`:
/// `... file is not a database: Error code 26: file is not a database`).
/// This presenter collapses exactly that pair — the `rusqlite::Error` and its
/// FFI source — into one `Error code N: <message>` link and leaves everything
/// else verbatim, so contexts and non-SQLite errors render exactly as the
/// `{:#}` chain did. It only reads the chain: the error object, its downcast
/// types and `source()` links stay untouched.
pub fn format_error(error: &anyhow::Error) -> String {
    let mut rendered: Vec<String> = Vec::new();
    // When a `SqliteFailure` link is rendered, its `source()` — the next
    // link in the chain — is the redundant FFI tail and is skipped.
    let mut skip_ffi_tail = false;
    for element in error.chain() {
        if skip_ffi_tail {
            skip_ffi_tail = false;
            continue;
        }
        match element.downcast_ref::<rusqlite::Error>() {
            Some(rusqlite::Error::SqliteFailure(ffi, message)) => {
                match message {
                    Some(message) => {
                        rendered.push(format!("Error code {}: {message}", ffi.extended_code))
                    }
                    None => rendered.push(element.to_string()),
                }
                skip_ffi_tail = true;
            }
            _ => rendered.push(element.to_string()),
        }
    }
    rendered.join(": ")
}

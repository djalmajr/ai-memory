//! Local spool for lifecycle-hook events — decouples capture from the network.
//!
//! Per-tool-call hooks (`pre-tool-use`, `post-tool-use`, `user-prompt-submit`,
//! `stop`) append an event here (an instant local write) instead of POSTing
//! synchronously. The spool is drained to the server at **session boundaries**
//! (a cleanup pass at `session-start`, the main flush at `session-end`), where a
//! few seconds of latency is acceptable — unlike the per-tool-call hot path,
//! which must never block the agent.
//!
//! This makes capture reliable against a remote/slow server (no event is lost:
//! a file persists until the server answers 2xx) without ever blocking a tool
//! call. It also fits ai-memory's model: consolidation runs on `session-end`,
//! after the drain has delivered the session's observations in order.
//!
//! Each event carries its own auth so a single global spool can hold events
//! for several instances: a static token is stored inline (file mode 0600);
//! an OIDC event stores only the mode and is resolved + refreshed from
//! `auth.json` at drain time (so a token that expired while the event waited is
//! renewed rather than rejected).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ai_memory_llm::{OidcToken, refresh_access_token};
use secrecy::ExposeSecret as _;
use serde::{Deserialize, Serialize};

use super::hook_capture::{PostOutcome, build_client, post_hook};

/// Drop a spooled event after this many failed drain passes — bounds retries of
/// a permanently-undeliverable event (e.g. a server URL that never comes back).
const MAX_ATTEMPTS: u32 = 8;
/// Drop a spooled event older than this regardless of attempts (7 days), so a
/// long-dead instance can't leave the spool growing without bound.
const MAX_AGE_MS: u64 = 7 * 24 * 60 * 60 * 1000;
/// Hard cap on queued events per data dir. Enqueue prunes oldest files beyond
/// this so a down server cannot grow the hook spool without bound.
#[cfg(not(test))]
const MAX_SPOOL_FILES: usize = 10_000;
#[cfg(test)]
const MAX_SPOOL_FILES: usize = 3;

/// How a spooled event authenticates to the server when drained.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum AuthMode {
    /// A static bearer stored inline (`token`) — service-account / edge token.
    #[serde(rename = "static")]
    Static,
    /// Resolve + refresh a stored OIDC device-grant token from `auth.json`.
    #[serde(rename = "oidc")]
    Oidc,
    /// No bearer (loopback / no-auth server).
    #[serde(rename = "none")]
    Anonymous,
}

/// One spooled hook event: the full request plus how to authenticate it.
#[derive(Serialize, Deserialize, Debug)]
pub struct SpoolEntry {
    /// Full hook URL including the `?event=…&agent=…[&cwd&workspace&project]`
    /// query the agent's payload resolved to.
    pub url: String,
    /// The raw JSON event payload to POST.
    pub body: String,
    /// Enqueue time (Unix ms) — for ordering + future TTL pruning.
    pub created_ms: u64,
    /// How to authenticate this event at drain time.
    pub auth_mode: AuthMode,
    /// Static bearer, present only when `auth_mode == Static`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    /// Failed delivery attempts so far — incremented on each drain miss and used
    /// (with `created_ms`) to drop a permanently-undeliverable event.
    #[serde(default)]
    pub attempts: u32,
}

/// `<data_dir>/hook-spool` — the spool directory.
#[must_use]
pub fn spool_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("hook-spool")
}

/// Count queued spool entries (`*.json`), or 0 when the dir is missing/empty.
/// Cheaper than [`drain`] — a single `read_dir` — so the per-event hot path can
/// gate a mid-session drain on backlog size without building a client.
#[must_use]
pub fn spool_len(spool: &Path) -> usize {
    list_entries(spool).map_or(0, |f| f.len())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

/// Append an event to the spool, atomically (temp file + rename) and 0600 on
/// Unix. Never touches the network. Each hook invocation enqueues exactly one
/// event, so the `<ms>-<pid>` name is unique.
///
/// # Errors
/// Returns an error only when the spool file cannot be written.
pub fn enqueue(spool: &Path, entry: &SpoolEntry) -> std::io::Result<()> {
    std::fs::create_dir_all(spool)?;
    let name = format!("{:013}-{}.json", entry.created_ms, std::process::id());
    let tmp = spool.join(format!("{name}.tmp"));
    let final_path = spool.join(&name);
    let bytes = serde_json::to_vec(entry)?;
    write_private(&tmp, &bytes)?;
    std::fs::rename(&tmp, &final_path)?;
    prune_spool_file_count(spool);
    Ok(())
}

fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    let mut opts = std::fs::OpenOptions::new();
    opts.create(true).truncate(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        opts.mode(0o600);
    }
    let mut file = opts.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Build a [`SpoolEntry`] for the current event, choosing the auth mode from
/// the hook's flags + stored credentials (no network, no token I/O):
/// an explicit `--auth-token` → `Static`; else a present OIDC `auth.json`
/// entry → `Oidc`; else `Anonymous`.
#[must_use]
pub fn entry_for(
    url: String,
    body: String,
    auth_token: Option<&str>,
    oidc_present: bool,
) -> SpoolEntry {
    let (auth_mode, token) = match auth_token {
        Some(t) => (AuthMode::Static, Some(t.to_string())),
        None if oidc_present => (AuthMode::Oidc, None),
        None => (AuthMode::Anonymous, None),
    };
    SpoolEntry {
        url,
        body,
        created_ms: now_ms(),
        auth_mode,
        token,
        attempts: 0,
    }
}

/// Outcome of a drain pass.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct DrainResult {
    /// Events delivered (server answered 2xx) and removed from the spool.
    pub sent: usize,
    /// Events still queued (failed this pass, or skipped when the budget ran out).
    pub remaining: usize,
    /// Events discarded as undeliverable (too old or too many failed attempts).
    pub dropped: usize,
}

/// Drain the spool to the server, oldest-first, within `total_budget`. Each
/// event that gets a 2xx is deleted; failures stay for the next boundary.
/// OIDC bearer is resolved + refreshed at most once per drain and cached.
///
/// Best-effort: returns counts and never errors, so a session boundary is never
/// blocked beyond the budget and never fails the agent.
pub async fn drain(
    spool: &Path,
    data_dir: &Path,
    total_budget: Duration,
    per_event_timeout: Duration,
) -> DrainResult {
    let mut files = match list_entries(spool) {
        Some(f) => f,
        None => return DrainResult::default(),
    };
    files.sort();

    let client = build_client();
    let started = Instant::now();
    let mut oidc_cache: Option<Option<String>> = None; // outer None = not yet resolved
    let mut result = DrainResult::default();

    for path in files {
        if started.elapsed() >= total_budget {
            result.remaining += 1;
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            result.remaining += 1;
            continue;
        };
        let Ok(mut entry) = serde_json::from_slice::<SpoolEntry>(&bytes) else {
            // Unparseable spool file: drop it so it can't wedge the queue.
            let _ = std::fs::remove_file(&path);
            result.dropped += 1;
            continue;
        };

        // Prune events too old to be worth retrying (a long-dead instance).
        if now_ms().saturating_sub(entry.created_ms) > MAX_AGE_MS {
            let _ = std::fs::remove_file(&path);
            result.dropped += 1;
            continue;
        }

        let bearer: Option<String> = match entry.auth_mode {
            AuthMode::Static => entry.token.clone(),
            AuthMode::Anonymous => None,
            AuthMode::Oidc => {
                if oidc_cache.is_none() {
                    oidc_cache = Some(resolve_oidc(&client, data_dir).await);
                }
                oidc_cache.clone().flatten()
            }
        };

        match post_hook(
            &client,
            &entry.url,
            &entry.body,
            bearer.as_deref(),
            per_event_timeout,
        )
        .await
        {
            PostOutcome::Delivered => {
                let _ = std::fs::remove_file(&path);
                result.sent += 1;
            }
            // Transient server backpressure (429 `hook queue full`): the event
            // was never processed, so leave it untouched — no attempt bump, no
            // rewrite — and saturation never burns the entry's retry budget. It
            // rides the next pass unchanged; `MAX_AGE_MS` still bounds it.
            PostOutcome::Saturated => {
                result.remaining += 1;
            }
            PostOutcome::Failed => {
                entry.attempts = entry.attempts.saturating_add(1);
                if entry.attempts >= MAX_ATTEMPTS {
                    let _ = std::fs::remove_file(&path);
                    result.dropped += 1;
                } else {
                    // Persist the bumped attempt count for the next boundary.
                    let _ = rewrite_entry(&path, &entry);
                    result.remaining += 1;
                }
            }
        }
    }
    result
}

/// Overwrite a spool file in place with the updated entry (atomic temp+rename),
/// used to persist a bumped attempt count after a failed delivery.
fn rewrite_entry(path: &Path, entry: &SpoolEntry) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let bytes = serde_json::to_vec(entry)?;
    write_private(&tmp, &bytes)?;
    std::fs::rename(&tmp, path)
}

/// Resolve the bearer for a synchronous request (the session-start handoff
/// GET): a static `--auth-token` wins, else the stored OIDC token
/// (refreshed if stale), else none.
pub async fn resolve_bearer(
    client: &reqwest::Client,
    data_dir: &Path,
    auth_token: Option<&str>,
) -> Option<String> {
    match auth_token {
        Some(t) => Some(t.to_string()),
        None => resolve_oidc(client, data_dir).await,
    }
}

/// Load the stored OIDC token, refreshing (and persisting) it when stale.
/// Returns the access token, or None when there's no token / refresh failed.
async fn resolve_oidc(client: &reqwest::Client, data_dir: &Path) -> Option<String> {
    let auth_path = data_dir.join("auth.json");
    let mut token = OidcToken::load(&auth_path).ok().flatten()?;
    if token.needs_refresh() {
        let Ok(refreshed) = refresh_access_token(client, &token).await else {
            return None;
        };
        let _ = refreshed.save(&auth_path);
        token = refreshed;
    }
    Some(token.access.expose_secret().to_string())
}

fn prune_spool_file_count(spool: &Path) {
    let Some(mut files) = list_entries(spool) else {
        return;
    };
    let excess = files.len().saturating_sub(MAX_SPOOL_FILES);
    if excess == 0 {
        return;
    }
    files.sort();
    for path in files.into_iter().take(excess) {
        let _ = std::fs::remove_file(path);
    }
}

/// List `*.json` spool files (ignoring in-flight `*.json.tmp`), or None when the
/// directory doesn't exist yet.
fn list_entries(spool: &Path) -> Option<Vec<PathBuf>> {
    let read = std::fs::read_dir(spool).ok()?;
    let mut out = Vec::new();
    for ent in read.flatten() {
        let path = ent.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            out.push(path);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_for_picks_auth_mode() {
        let s = entry_for("u".into(), "{}".into(), Some("tok"), false);
        assert_eq!(s.auth_mode, AuthMode::Static);
        assert_eq!(s.token.as_deref(), Some("tok"));

        let o = entry_for("u".into(), "{}".into(), None, true);
        assert_eq!(o.auth_mode, AuthMode::Oidc);
        assert!(o.token.is_none());

        let a = entry_for("u".into(), "{}".into(), None, false);
        assert_eq!(a.auth_mode, AuthMode::Anonymous);
    }

    #[test]
    fn enqueue_then_list_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let spool = spool_dir(tmp.path());
        let entry = entry_for(
            "https://x/hook?event=stop".into(),
            "{\"session_id\":\"s\"}".into(),
            Some("tok"),
            false,
        );
        enqueue(&spool, &entry).unwrap();
        let files = list_entries(&spool).unwrap();
        assert_eq!(files.len(), 1);
        let loaded: SpoolEntry =
            serde_json::from_slice(&std::fs::read(&files[0]).unwrap()).unwrap();
        assert_eq!(loaded.url, "https://x/hook?event=stop");
        assert_eq!(loaded.auth_mode, AuthMode::Static);
        assert_eq!(loaded.token.as_deref(), Some("tok"));
    }

    #[test]
    fn enqueue_prunes_oldest_files_when_spool_exceeds_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let spool = spool_dir(tmp.path());
        for i in 0..(MAX_SPOOL_FILES + 2) {
            let mut entry = entry_for(
                format!("https://x/hook?event=e{i}"),
                "{}".into(),
                None,
                false,
            );
            entry.created_ms = i as u64;
            enqueue(&spool, &entry).unwrap();
        }

        let mut files = list_entries(&spool).unwrap();
        files.sort();
        assert_eq!(files.len(), MAX_SPOOL_FILES);
        let bodies: Vec<SpoolEntry> = files
            .iter()
            .map(|path| serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
            .collect();
        assert!(bodies.iter().all(|entry| entry.created_ms >= 2));
    }

    #[tokio::test]
    async fn drain_unreachable_leaves_events_queued() {
        let tmp = tempfile::tempdir().unwrap();
        let spool = spool_dir(tmp.path());
        // Two anonymous events pointing at an unroutable port.
        for i in 0..2 {
            let e = entry_for(
                format!("http://127.0.0.1:1/hook?event=e{i}"),
                "{}".into(),
                None,
                false,
            );
            // Distinct filenames: enqueue uses ms+pid, so space them out.
            enqueue(&spool, &e).unwrap();
            std::fs::rename(
                spool.join(format!("{:013}-{}.json", e.created_ms, std::process::id())),
                spool.join(format!("evt-{i}.json")),
            )
            .unwrap();
        }
        let r = drain(
            &spool,
            tmp.path(),
            Duration::from_secs(2),
            Duration::from_millis(200),
        )
        .await;
        assert_eq!(r.sent, 0);
        assert_eq!(r.remaining, 2);
        // Files survive for the next boundary.
        assert_eq!(list_entries(&spool).unwrap().len(), 2);
    }

    #[tokio::test]
    async fn drain_empty_spool_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let r = drain(
            &spool_dir(tmp.path()),
            tmp.path(),
            Duration::from_secs(1),
            Duration::from_millis(200),
        )
        .await;
        assert_eq!(r, DrainResult::default());
    }

    #[tokio::test]
    async fn drain_drops_event_after_max_attempts() {
        let tmp = tempfile::tempdir().unwrap();
        let spool = spool_dir(tmp.path());
        let e = entry_for(
            "http://127.0.0.1:1/hook?event=dead".into(),
            "{}".into(),
            None,
            false,
        );
        enqueue(&spool, &e).unwrap();
        let mut dropped = 0;
        for _ in 0..MAX_ATTEMPTS {
            dropped += drain(
                &spool,
                tmp.path(),
                Duration::from_secs(2),
                Duration::from_millis(100),
            )
            .await
            .dropped;
        }
        assert_eq!(
            dropped, 1,
            "the dead event is dropped once it hits MAX_ATTEMPTS"
        );
        assert!(
            list_entries(&spool).unwrap().is_empty(),
            "spool is empty after the drop"
        );
    }

    #[tokio::test]
    async fn drain_drops_stale_event() {
        let tmp = tempfile::tempdir().unwrap();
        let spool = spool_dir(tmp.path());
        std::fs::create_dir_all(&spool).unwrap();
        let mut e = entry_for(
            "http://127.0.0.1:1/hook?event=old".into(),
            "{}".into(),
            None,
            false,
        );
        e.created_ms = now_ms().saturating_sub(MAX_AGE_MS + 1);
        std::fs::write(spool.join("stale.json"), serde_json::to_vec(&e).unwrap()).unwrap();
        let r = drain(
            &spool,
            tmp.path(),
            Duration::from_secs(2),
            Duration::from_millis(100),
        )
        .await;
        assert_eq!(r.dropped, 1);
        assert_eq!(r.sent, 0);
        assert!(list_entries(&spool).unwrap().is_empty());
    }

    #[tokio::test]
    async fn drain_429_keeps_event_queued_without_bumping_attempts() {
        // A server that always answers 429 (saturation / `hook queue full`).
        // The event must ride every pass untouched: never dropped, attempts
        // never incremented — saturation must not burn the retry budget.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
            while let Ok((mut s, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = [0_u8; 1024];
                    let _ = s.read(&mut buf).await;
                    let _ = s
                        .write_all(
                            b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 4\r\nConnection: close\r\n\r\nfull",
                        )
                        .await;
                });
            }
        });

        let tmp = tempfile::tempdir().unwrap();
        let spool = spool_dir(tmp.path());
        let e = entry_for(
            format!("http://{addr}/hook?event=x"),
            "{}".into(),
            None,
            false,
        );
        enqueue(&spool, &e).unwrap();

        // Far more passes than MAX_ATTEMPTS — a 429 must never consume budget.
        for _ in 0..(MAX_ATTEMPTS + 2) {
            let r = drain(
                &spool,
                tmp.path(),
                Duration::from_secs(2),
                Duration::from_millis(500),
            )
            .await;
            assert_eq!(r.sent, 0);
            assert_eq!(r.dropped, 0, "a 429 must never drop the event");
            assert_eq!(r.remaining, 1);
        }
        let files = list_entries(&spool).unwrap();
        assert_eq!(files.len(), 1, "event still queued after many 429s");
        let loaded: SpoolEntry =
            serde_json::from_slice(&std::fs::read(&files[0]).unwrap()).unwrap();
        assert_eq!(loaded.attempts, 0, "429 must not consume the retry budget");
    }

    #[test]
    fn spool_len_counts_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let spool = spool_dir(tmp.path());
        assert_eq!(spool_len(&spool), 0, "missing dir counts as 0");
        std::fs::create_dir_all(&spool).unwrap();
        // Write distinct files directly (enqueue's ms+pid names would collide in
        // a tight loop, and its prune caps at the test MAX_SPOOL_FILES).
        for i in 0..3 {
            let e = entry_for(
                format!("http://x/hook?event=e{i}"),
                "{}".into(),
                None,
                false,
            );
            std::fs::write(
                spool.join(format!("evt-{i}.json")),
                serde_json::to_vec(&e).unwrap(),
            )
            .unwrap();
        }
        assert_eq!(spool_len(&spool), 3);
    }
}

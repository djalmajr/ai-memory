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

use super::hook_capture::{BatchOutcome, PostOutcome, build_client, post_batch, post_hook};

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

/// Max events per `POST /hook/batch` request (count bound; the byte bound below
/// also applies). Caps the blast radius of a failed batch and keeps one request
/// well under the server's body limit even with many small events.
const MAX_BATCH_ITEMS: usize = 256;
/// Soft byte budget for one `/hook/batch` body — stays under the server's 10 MiB
/// `DefaultBodyLimit` with margin for JSON framing. A chunk always carries at
/// least one event even if that event alone exceeds this.
const MAX_BATCH_BYTES: usize = 8 * 1024 * 1024;

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
#[derive(Serialize, Deserialize, Clone, Debug)]
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

/// Drain the spool to the server, oldest-first, within `total_budget`.
///
/// Events are delivered in **batches** via `POST /hook/batch`: one request
/// carries many spooled events, so the per-request cost (TLS + network RTT + the
/// edge auth hop) is amortized over the whole batch instead of paid per event.
/// That is the throughput fix — a sequential per-event drain falls behind when
/// many parallel sessions share one spool against a remote, gated server, and
/// the spool then grows to its cap and evicts undelivered events. A server
/// without `/hook/batch` (a pre-upgrade build) answers `404`/`405`, and the
/// drain transparently falls back to per-event `POST /hook`.
///
/// A delivered event is deleted; a failed one is charged a retry attempt
/// (dropped at `MAX_ATTEMPTS`); a `429` (saturation) is retried untouched so it
/// never burns the retry budget. OIDC bearer is resolved + refreshed at most
/// once per drain and cached.
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

    // Load entries oldest-first, dropping unparseable / too-old files up front
    // (a long-dead instance must not keep the spool growing). The batch path
    // then only ever sees live, parseable events.
    let mut items: Vec<(PathBuf, SpoolEntry)> = Vec::with_capacity(files.len());
    for path in files {
        let Ok(bytes) = std::fs::read(&path) else {
            result.remaining += 1;
            continue;
        };
        let Ok(entry) = serde_json::from_slice::<SpoolEntry>(&bytes) else {
            // Unparseable spool file: drop it so it can't wedge the queue.
            let _ = std::fs::remove_file(&path);
            result.dropped += 1;
            continue;
        };
        if now_ms().saturating_sub(entry.created_ms) > MAX_AGE_MS {
            let _ = std::fs::remove_file(&path);
            result.dropped += 1;
            continue;
        }
        items.push((path, entry));
    }

    let mut idx = 0;
    let mut batch_supported = true;
    while idx < items.len() {
        if started.elapsed() >= total_budget {
            result.remaining += items.len() - idx;
            break;
        }

        let bearer = entry_bearer(&items[idx].1, &client, data_dir, &mut oidc_cache).await;

        if batch_supported {
            // Extend the chunk over consecutive entries sharing the same batch
            // endpoint AND bearer (one request carries one Authorization header),
            // bounded by item count and body bytes.
            let base = batch_endpoint(&items[idx].1.url);
            let mut end = idx + 1;
            let mut bytes = entry_wire_len(&items[idx].1);
            while end < items.len() && end - idx < MAX_BATCH_ITEMS {
                if batch_endpoint(&items[end].1.url) != base {
                    break;
                }
                let next_bearer =
                    entry_bearer(&items[end].1, &client, data_dir, &mut oidc_cache).await;
                if next_bearer != bearer {
                    break;
                }
                let next_len = entry_wire_len(&items[end].1);
                if bytes + next_len > MAX_BATCH_BYTES {
                    break;
                }
                bytes += next_len;
                end += 1;
            }

            let payload = batch_payload(&items[idx..end]);
            match post_batch(&client, &base, &payload, bearer.as_deref(), per_event_timeout).await {
                BatchOutcome::Accepted(k) => {
                    let k = k.min(end - idx);
                    for (path, _) in &items[idx..idx + k] {
                        let _ = std::fs::remove_file(path);
                    }
                    result.sent += k;
                    if idx + k < end {
                        // The (idx+k)th event is the one the server stopped on
                        // (fail-fast). Charge it a failed attempt and skip past it
                        // so a single bad event can't wedge the rest — the
                        // per-event loop also advances past a failed entry.
                        bump_or_drop(&items[idx + k].0, &items[idx + k].1, &mut result);
                        idx += k + 1;
                    } else {
                        idx = end;
                    }
                }
                BatchOutcome::Saturated => {
                    // Server ingest is full; further batches would 429 too. Leave
                    // everything queued, no attempt bump (parity with per-event).
                    result.remaining += items.len() - idx;
                    break;
                }
                BatchOutcome::Unsupported => {
                    // Pre-upgrade server with no /hook/batch: fall back to the
                    // per-event path for the rest of the drain. Retry items[idx]
                    // below (don't advance).
                    batch_supported = false;
                }
                BatchOutcome::Failed => {
                    // The batch didn't land (transport error / unexpected status).
                    // Charge each item a failed attempt so a dead server still
                    // bounds retries via MAX_ATTEMPTS, then move past the chunk.
                    for (path, entry) in &items[idx..end] {
                        bump_or_drop(path, entry, &mut result);
                    }
                    idx = end;
                }
            }
        } else {
            // Per-event fallback (server without /hook/batch). Mirrors the
            // original drain's per-entry semantics exactly.
            let path = &items[idx].0;
            let entry = &items[idx].1;
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
                    let _ = std::fs::remove_file(path);
                    result.sent += 1;
                }
                PostOutcome::Saturated => {
                    result.remaining += 1;
                }
                PostOutcome::Failed => {
                    bump_or_drop(path, entry, &mut result);
                }
            }
            idx += 1;
        }
    }
    result
}

/// Resolve the bearer for a spooled entry at drain time: a `Static` token is
/// stored inline; an `Oidc` entry resolves (and refreshes) the stored token
/// once per drain via `oidc_cache`; `Anonymous` is None.
async fn entry_bearer(
    entry: &SpoolEntry,
    client: &reqwest::Client,
    data_dir: &Path,
    oidc_cache: &mut Option<Option<String>>,
) -> Option<String> {
    match entry.auth_mode {
        AuthMode::Static => entry.token.clone(),
        AuthMode::Anonymous => None,
        AuthMode::Oidc => {
            if oidc_cache.is_none() {
                *oidc_cache = Some(resolve_oidc(client, data_dir).await);
            }
            oidc_cache.clone().flatten()
        }
    }
}

/// The `/hook/batch` URL for a spooled per-event URL: strip the `?…` query and
/// append `/batch` (a spooled URL ends in `…/hook` before its query). Entries
/// whose endpoint string matches can ride one batch request.
fn batch_endpoint(url: &str) -> String {
    let path = url.split('?').next().unwrap_or(url);
    format!("{path}/batch")
}

/// Rough wire size of one event inside a batch body (`{"url":…,"body":…}` plus
/// framing) — used only to keep a chunk under [`MAX_BATCH_BYTES`].
fn entry_wire_len(entry: &SpoolEntry) -> usize {
    entry.url.len() + entry.body.len() + 32
}

/// Serialize a chunk of entries into the `/hook/batch` request body — a JSON
/// array of `{url, body}`. Each `body` is re-parsed from its stored text so a
/// malformed one becomes `null` (skipped server-side) instead of poisoning the
/// whole batch.
fn batch_payload(items: &[(PathBuf, SpoolEntry)]) -> String {
    let arr: Vec<serde_json::Value> = items
        .iter()
        .map(|(_, e)| {
            let body = serde_json::from_str::<serde_json::Value>(&e.body)
                .unwrap_or(serde_json::Value::Null);
            serde_json::json!({ "url": e.url, "body": body })
        })
        .collect();
    serde_json::to_string(&arr).unwrap_or_else(|_| "[]".to_string())
}

/// Charge a spooled entry a failed delivery attempt: drop it once it reaches
/// `MAX_ATTEMPTS`, else persist the bumped count for the next boundary. Updates
/// `result.dropped` / `result.remaining` accordingly.
fn bump_or_drop(path: &Path, entry: &SpoolEntry, result: &mut DrainResult) {
    let mut bumped = entry.clone();
    bumped.attempts = bumped.attempts.saturating_add(1);
    if bumped.attempts >= MAX_ATTEMPTS {
        let _ = std::fs::remove_file(path);
        result.dropped += 1;
    } else {
        let _ = note_retry_persist(rewrite_entry(path, &bumped));
        result.remaining += 1;
    }
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

/// Report whether persisting a bumped retry count landed; on failure, emit a
/// sanitized stderr warning (no path — a raw spool path can be a Windows verbatim
/// `\\?\…` path) instead of swallowing it, so a poison entry can't retry
/// invisibly until it ages out. Fire-and-forget: warns only, never panics or
/// blocks; the returned bool is consumed only by tests.
fn note_retry_persist(outcome: std::io::Result<()>) -> bool {
    if outcome.is_err() {
        eprintln!(
            "ai-memory hook warning: failed to persist spool retry count; \
             event may retry until it ages out"
        );
        return false;
    }
    true
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
    // The spool is at its hard cap: the oldest events are about to be deleted
    // WITHOUT ever reaching the server — silent capture loss otherwise. Surface
    // it on stderr (never stdout, which carries the hook's JSON protocol output)
    // so a sustained backlog dropping events is visible, not invisible.
    eprintln!(
        "ai-memory: hook-spool at capacity ({} > {MAX_SPOOL_FILES}); evicting {excess} oldest UNDELIVERED event(s)",
        files.len()
    );
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

    /// A mock hook server: answers `200 {"accepted":N}` to `POST /hook/batch`
    /// (N = array length in the body) and `202 queued` to a per-event
    /// `POST /hook`. Counts every request so a test can assert batching. Reads
    /// the whole request in one shot (small test payloads), mirroring the other
    /// raw-TCP mocks in this module.
    async fn serve_counting_hook(
        req_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        batch_status: &'static str,
    ) -> String {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut s, _)) = listener.accept().await {
                let rc = req_count.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0_u8; 65536];
                    let n = s.read(&mut buf).await.unwrap_or(0);
                    rc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let is_batch = req
                        .lines()
                        .next()
                        .is_some_and(|l| l.contains("/hook/batch"));
                    let (status, body) = if is_batch {
                        let payload = req.split("\r\n\r\n").nth(1).unwrap_or("");
                        let accepted = serde_json::from_str::<serde_json::Value>(payload)
                            .ok()
                            .and_then(|v| v.as_array().map(Vec::len))
                            .unwrap_or(0);
                        (batch_status, format!("{{\"accepted\":{accepted}}}"))
                    } else {
                        ("202 Accepted", "queued".to_string())
                    };
                    let resp = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = s.write_all(resp.as_bytes()).await;
                });
            }
        });
        addr.to_string()
    }

    fn write_spool_entry(spool: &Path, name: &str, url: String) {
        std::fs::create_dir_all(spool).unwrap();
        let e = entry_for(url, "{}".into(), None, false);
        std::fs::write(spool.join(name), serde_json::to_vec(&e).unwrap()).unwrap();
    }

    #[tokio::test]
    async fn drain_delivers_all_events_in_one_batch() {
        let req_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let addr = serve_counting_hook(req_count.clone(), "200 OK").await;
        let tmp = tempfile::tempdir().unwrap();
        let spool = spool_dir(tmp.path());
        for i in 0..3 {
            write_spool_entry(&spool, &format!("evt-{i}.json"), format!("http://{addr}/hook?event=e{i}"));
        }

        let r = drain(
            &spool,
            tmp.path(),
            Duration::from_secs(5),
            Duration::from_secs(2),
        )
        .await;

        assert_eq!(r.sent, 3, "all three events delivered");
        assert_eq!(r.remaining, 0);
        assert!(list_entries(&spool).unwrap().is_empty(), "spool emptied");
        assert_eq!(
            req_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "three events ride ONE /hook/batch request (RTT amortized)"
        );
    }

    #[tokio::test]
    async fn drain_falls_back_to_per_event_when_batch_unsupported() {
        // Pre-upgrade server: /hook/batch is 404, per-event /hook is 202.
        let req_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let addr = serve_counting_hook(req_count.clone(), "404 Not Found").await;
        let tmp = tempfile::tempdir().unwrap();
        let spool = spool_dir(tmp.path());
        for i in 0..2 {
            write_spool_entry(&spool, &format!("evt-{i}.json"), format!("http://{addr}/hook?event=e{i}"));
        }

        let r = drain(
            &spool,
            tmp.path(),
            Duration::from_secs(5),
            Duration::from_secs(2),
        )
        .await;

        assert_eq!(r.sent, 2, "both events delivered via per-event fallback");
        assert_eq!(r.remaining, 0);
        assert!(list_entries(&spool).unwrap().is_empty());
        // 1 rejected batch probe + 2 per-event POSTs.
        assert_eq!(
            req_count.load(std::sync::atomic::Ordering::SeqCst),
            3,
            "one /hook/batch 404, then a per-event POST per remaining event"
        );
    }

    #[test]
    fn batch_endpoint_derives_from_event_url() {
        assert_eq!(
            batch_endpoint("https://h/hook?event=stop&agent=claude-code"),
            "https://h/hook/batch"
        );
        assert_eq!(batch_endpoint("https://h/hook"), "https://h/hook/batch");
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

    #[test]
    fn note_retry_persist_reports_failure() {
        // Root-proof: feed a synthetic error so the warn / not-persisted branch is
        // exercised without provoking a real FS fault (the Docker CI gate runs as
        // root and ignores chmod-based read-only dirs).
        let failed: std::io::Result<()> = Err(std::io::Error::other("simulated rewrite failure"));
        assert!(
            !note_retry_persist(failed),
            "a failed persist is reported as not-persisted, not swallowed"
        );
    }

    #[test]
    fn note_retry_persist_reports_success() {
        assert!(
            note_retry_persist(Ok(())),
            "a successful persist is reported as persisted"
        );
    }

    #[tokio::test]
    async fn drain_stays_robust_when_retry_count_cannot_persist() {
        let tmp = tempfile::tempdir().unwrap();
        let spool = spool_dir(tmp.path());
        let e = entry_for(
            "http://127.0.0.1:1/hook?event=stuck".into(),
            "{}".into(),
            None,
            false,
        );
        enqueue(&spool, &e).unwrap();

        // Make the atomic rewrite fail in a way that survives root (the Docker gate
        // runs as root, so a chmod read-only dir wouldn't fault): occupy the entry's
        // `<name>.json.tmp` path with a directory, so `rewrite_entry`'s temp-file
        // write (an `OpenOptions` create) can't be created. `list_entries` matches
        // only `*.json`, so the `.json.tmp` directory is ignored by the drain.
        let entry_path = list_entries(&spool).unwrap().into_iter().next().unwrap();
        let mut blocker = entry_path.into_os_string();
        blocker.push(".tmp");
        std::fs::create_dir(PathBuf::from(blocker)).unwrap();

        // The persist fails, but drain must stay fire-and-forget: no panic, the
        // event is still counted as remaining, nothing dropped, the entry survives.
        let r = drain(
            &spool,
            tmp.path(),
            Duration::from_secs(2),
            Duration::from_millis(100),
        )
        .await;
        assert_eq!(r.sent, 0);
        assert_eq!(r.dropped, 0, "a persist failure must not drop the event");
        assert_eq!(
            r.remaining, 1,
            "the event stays queued for the next boundary"
        );
        assert_eq!(
            list_entries(&spool).unwrap().len(),
            1,
            "the spool entry survives a failed rewrite"
        );
    }
}

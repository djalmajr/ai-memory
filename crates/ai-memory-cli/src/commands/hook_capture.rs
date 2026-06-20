//! Native lifecycle-hook capture helpers.
//!
//! Mirrors the POSIX `hooks/lib/_lib.sh` logic so the native
//! `ai-memory hook` subcommand produces the same HTTP request the shell
//! scripts do: extract cwd from the payload, walk up for a
//! `.ai-memory.toml` marker, and build the query-string suffix. The two
//! request helpers are best-effort with shell-parity timeouts.

use std::path::{Path, PathBuf};
use std::time::Duration;

/// First top-level `cwd` string in the payload (parity with
/// `ai_memory_extract_cwd`: take the top-level value, ignore nested
/// `cwd` fields in tool payloads).
pub fn extract_cwd(payload: &serde_json::Value) -> Option<String> {
    payload
        .get("cwd")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
}

/// URL-encode the reserved characters `ai_memory_url_encode` handles.
pub fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '%' => out.push_str("%25"),
            '+' => out.push_str("%2B"),
            '&' => out.push_str("%26"),
            '=' => out.push_str("%3D"),
            '?' => out.push_str("%3F"),
            '#' => out.push_str("%23"),
            ' ' => out.push_str("%20"),
            '/' => out.push_str("%2F"),
            other => out.push(other),
        }
    }
    out
}

/// Build `&cwd=…[&workspace=…&project=…&project_strategy=…]`, mirroring
/// `ai_memory_marker_qs`: always include cwd; append marker-declared
/// fields when a `.ai-memory.toml` is found walking up toward $HOME.
pub fn marker_query_suffix(cwd: &str) -> String {
    let mut qs = format!("&cwd={}", url_encode(cwd));
    if let Some(marker) = find_marker(cwd) {
        let workspace = parse_toml_key(&marker, "workspace");
        let mut project = parse_toml_key(&marker, "project");
        let project_strategy = parse_toml_key(&marker, "project_strategy");
        if project.is_none()
            && matches!(project_strategy.as_deref(), Some("repo-root" | "repo_root"))
        {
            project = repo_root_project(cwd);
        }
        if let Some(val) = workspace {
            qs.push_str(&format!("&workspace={}", url_encode(&val)));
        }
        if let Some(val) = project {
            qs.push_str(&format!("&project={}", url_encode(&val)));
        }
        if let Some(val) = project_strategy {
            qs.push_str(&format!("&project_strategy={}", url_encode(&val)));
        }
    }
    qs
}

fn repo_root_project(cwd: &str) -> Option<String> {
    let root = ai_memory_consolidate::discover_main_repo_root(Path::new(cwd)).ok()?;
    root.file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

/// Walk up from `cwd` toward `$HOME` (or the filesystem root) looking
/// for `.ai-memory.toml`. Stops at `$HOME` to avoid leaking a parent
/// user's declaration on shared machines (parity with
/// `ai_memory_find_marker`).
fn find_marker(cwd: &str) -> Option<PathBuf> {
    let home = dirs::home_dir();
    let mut dir = Path::new(cwd);
    loop {
        let candidate = dir.join(".ai-memory.toml");
        if candidate.is_file() {
            return Some(candidate);
        }
        if home.as_deref() == Some(dir) {
            return None;
        }
        match dir.parent() {
            Some(parent) if parent != dir => dir = parent,
            _ => return None,
        }
    }
}

/// Parse a root-level `key = "value"` line (no nesting, arrays, or
/// tables), mirroring `ai_memory_parse_toml_key`. Returns the first
/// match. Avoids pulling in a TOML parser dependency.
fn parse_toml_key(file: &Path, key: &str) -> Option<String> {
    let text = std::fs::read_to_string(file).ok()?;
    for line in text.lines() {
        let trimmed = line.trim_start();
        let Some(after_key) = trimmed.strip_prefix(key) else {
            continue;
        };
        let Some(rest) = after_key.trim_start().strip_prefix('=') else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix('"') else {
            continue;
        };
        if let Some(end) = rest.find('"') {
            return Some(rest[..end].to_string());
        }
    }
    None
}

/// Build a reqwest client for the hook's one-shot requests. `no_proxy`
/// skips Windows proxy auto-detection (registry / WinINET lookups), which
/// is pure overhead for a loopback/LAN POST. Built once per invocation and
/// reused for both the event POST and the handoff GET. Default root certs
/// are kept so HTTPS targets (e.g. a TLS proxy) still work.
pub fn build_client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

/// Outcome of one spooled-event POST — enough for the drain loop to decide
/// whether a miss should cost the entry a retry attempt. Never errors, so a
/// hook/flush never fails the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostOutcome {
    /// Server acknowledged with a 2xx (the engine answers `202 queued`): the
    /// entry was delivered and can be removed from the spool.
    Delivered,
    /// Server answered `429 Too Many Requests` (`hook queue full`): transient
    /// backpressure, the event was never processed. Keep it queued WITHOUT
    /// bumping attempts so saturation never burns the entry's retry budget.
    Saturated,
    /// Any other non-2xx, or a transport error: a genuine miss that should
    /// count against `MAX_ATTEMPTS`.
    Failed,
}

/// POST the payload as JSON, best-effort. `timeout` is caller-chosen: the
/// per-tool-call hot path no longer POSTs at all (it spools); the drain calls
/// this with a budget that tolerates a remote/slow server. Returns a
/// [`PostOutcome`] (never errors) so the drain can give a 429 (saturation) a
/// free retry while still bounding genuine failures by `MAX_ATTEMPTS`.
pub async fn post_hook(
    client: &reqwest::Client,
    url: &str,
    body: &str,
    token: Option<&str>,
    timeout: Duration,
) -> PostOutcome {
    let mut req = client
        .post(url)
        .header("Content-Type", "application/json")
        .timeout(timeout)
        .body(body.to_owned());
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    match req.send().await {
        Ok(resp) if resp.status().is_success() => PostOutcome::Delivered,
        Ok(resp) if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS => {
            PostOutcome::Saturated
        }
        Ok(_) => PostOutcome::Failed,
        Err(_) => PostOutcome::Failed,
    }
}

/// Outcome of one `POST /hook/batch` request — many spooled events delivered in
/// a single round-trip, so a draining client amortizes TLS + network RTT + the
/// edge auth hop over the whole batch instead of paying it per event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchOutcome {
    /// Server committed the leading `usize` items (contiguous prefix, oldest
    /// first). Equals the request length on full success; a smaller value means
    /// the server stopped on that item (fail-fast) — the caller deletes the
    /// prefix and charges the next item a retry.
    Accepted(usize),
    /// `429` — ingest saturated; nothing was processed. Keep the whole batch and
    /// retry it later WITHOUT bumping attempts (saturation isn't a failure).
    Saturated,
    /// `404`/`405` — the server has no `/hook/batch` (a pre-upgrade build). The
    /// caller falls back to per-event `POST /hook` for the rest of the drain.
    Unsupported,
    /// Transport error or any other non-2xx: the batch didn't land. Treated as a
    /// failed attempt for each item it carried (bounded by `MAX_ATTEMPTS`).
    Failed,
}

/// POST a pre-serialized JSON array of `{url, body}` events to `<batch_url>`.
/// `bearer` authenticates the whole batch (every item shares the drain's single
/// identity). Best-effort: never errors. Reads `{"accepted": K}` from a 2xx
/// body; a 2xx whose body can't be read is treated as `Failed` (re-send rather
/// than risk dropping undelivered events).
pub async fn post_batch(
    client: &reqwest::Client,
    batch_url: &str,
    payload: &str,
    token: Option<&str>,
    timeout: Duration,
) -> BatchOutcome {
    let mut req = client
        .post(batch_url)
        .header("Content-Type", "application/json")
        .timeout(timeout)
        .body(payload.to_owned());
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    match req.send().await {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                match resp.json::<serde_json::Value>().await {
                    Ok(v) => {
                        let accepted = v
                            .get("accepted")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0) as usize;
                        BatchOutcome::Accepted(accepted)
                    }
                    Err(_) => BatchOutcome::Failed,
                }
            } else if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                BatchOutcome::Saturated
            } else if status == reqwest::StatusCode::NOT_FOUND
                || status == reqwest::StatusCode::METHOD_NOT_ALLOWED
            {
                BatchOutcome::Unsupported
            } else {
                BatchOutcome::Failed
            }
        }
        Err(_) => BatchOutcome::Failed,
    }
}

/// GET the handoff text with a caller-chosen budget. Returns None on any error
/// or an empty body. This is the one synchronous read on the agent's critical
/// path (session-start injects it as context), so the budget is larger than a
/// loopback default to tolerate a remote server.
pub async fn get_handoff(
    client: &reqwest::Client,
    url: &str,
    token: Option<&str>,
    timeout: Duration,
) -> Option<String> {
    let mut req = client.get(url).timeout(timeout);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body = resp.text().await.ok()?;
    if body.is_empty() { None } else { Some(body) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn serve_once(status: &'static str, body: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0_u8; 1024];
            let _ = stream.read(&mut buf).await;
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
        });
        format!("http://{addr}/handoff")
    }

    #[test]
    fn extracts_top_level_cwd() {
        let p: serde_json::Value =
            serde_json::from_str(r#"{"cwd":"/d/proj","tool_input":{"cwd":"/nested"}}"#).unwrap();
        assert_eq!(extract_cwd(&p).as_deref(), Some("/d/proj"));
    }

    #[test]
    fn missing_cwd_is_none() {
        let p: serde_json::Value = serde_json::from_str(r#"{"x":1}"#).unwrap();
        assert_eq!(extract_cwd(&p), None);
    }

    #[test]
    fn query_suffix_without_marker_has_only_cwd() {
        let qs = marker_query_suffix("/nonexistent/path/xyz");
        assert_eq!(qs, "&cwd=%2Fnonexistent%2Fpath%2Fxyz");
    }

    #[test]
    fn url_encode_escapes_reserved() {
        assert_eq!(url_encode("a b&c=d"), "a%20b%26c%3Dd");
    }

    #[tokio::test]
    async fn post_hook_failed_when_server_unreachable() {
        // Port 1 is unroutable; best-effort means this resolves to `Failed`
        // (a genuine miss) rather than panicking or erroring.
        let client = build_client();
        let outcome = post_hook(
            &client,
            "http://127.0.0.1:1/hook?event=pre-tool-use",
            "{}",
            None,
            Duration::from_millis(500),
        )
        .await;
        assert_eq!(outcome, PostOutcome::Failed);
    }

    #[tokio::test]
    async fn post_hook_saturated_on_429() {
        // 429 = server backpressure; the event was never processed, so the
        // drain must treat it as a free retry, not a failed attempt.
        let url = serve_once("429 Too Many Requests", "hook queue full").await;
        let outcome = post_hook(&build_client(), &url, "{}", None, Duration::from_secs(1)).await;
        assert_eq!(outcome, PostOutcome::Saturated);
    }

    #[tokio::test]
    async fn post_hook_delivered_on_2xx() {
        let url = serve_once("202 Accepted", "queued").await;
        let outcome = post_hook(&build_client(), &url, "{}", None, Duration::from_secs(1)).await;
        assert_eq!(outcome, PostOutcome::Delivered);
    }

    #[tokio::test]
    async fn get_handoff_ignores_non_success_status() {
        let url = serve_once("401 Unauthorized", "unauthorized").await;
        let got = get_handoff(&build_client(), &url, None, Duration::from_secs(1)).await;
        assert!(got.is_none(), "non-2xx body must not become context");
    }

    /// Happy-path TOML parser: extracts each declared root-level
    /// `key = "value"` pair. Mirrors the shell `ai_memory_parse_toml_key`.
    #[test]
    fn parse_toml_key_extracts_root_level_strings() {
        let tmp = tempfile::TempDir::new().unwrap();
        let marker = tmp.path().join(".ai-memory.toml");
        std::fs::write(
            &marker,
            r#"
workspace = "acme"
project = "infra"
project_strategy = "repo-root"
"#,
        )
        .unwrap();
        assert_eq!(
            parse_toml_key(&marker, "workspace").as_deref(),
            Some("acme")
        );
        assert_eq!(parse_toml_key(&marker, "project").as_deref(), Some("infra"));
        assert_eq!(
            parse_toml_key(&marker, "project_strategy").as_deref(),
            Some("repo-root")
        );
        assert_eq!(parse_toml_key(&marker, "absent"), None);
    }

    /// Shapes the naive parser deliberately doesn't handle (parity with
    /// the shell `_lib.sh` helper) — pin the contract so a future
    /// "robustify" refactor doesn't silently start matching them.
    #[test]
    fn parse_toml_key_skips_unsupported_shapes() {
        let tmp = tempfile::TempDir::new().unwrap();
        let marker = tmp.path().join(".ai-memory.toml");
        std::fs::write(
            &marker,
            r#"
# Single-quoted values are not honoured.
workspace = 'acme'
# Comments after the value are not stripped.
project = "infra" # this is fine
"#,
        )
        .unwrap();
        assert_eq!(parse_toml_key(&marker, "workspace"), None);
        // The trailing comment is appended to the value because the parser
        // looks for the first `"` — pin it so the contract is explicit.
        assert_eq!(parse_toml_key(&marker, "project").as_deref(), Some("infra"));
    }

    /// `find_marker` walks up from `cwd` until it finds `.ai-memory.toml`
    /// or reaches `$HOME`. Verify the walking — drop the marker two dirs
    /// above the simulated cwd and confirm it's found.
    #[test]
    fn find_marker_walks_up_from_cwd() {
        let tmp = tempfile::TempDir::new().unwrap();
        let marker = tmp.path().join(".ai-memory.toml");
        std::fs::write(&marker, "workspace = \"w\"\n").unwrap();
        let deep = tmp.path().join("a/b/c");
        std::fs::create_dir_all(&deep).unwrap();
        let found = find_marker(deep.to_str().unwrap());
        assert_eq!(found.as_deref(), Some(marker.as_path()));
    }

    /// `marker_query_suffix` appends `&workspace=…&project=…` (and
    /// `&project_strategy=…`) when the marker declares them. Each value is
    /// URL-encoded, so a workspace with a space round-trips as `%20`.
    #[test]
    fn marker_query_suffix_appends_marker_fields() {
        let tmp = tempfile::TempDir::new().unwrap();
        let marker = tmp.path().join(".ai-memory.toml");
        std::fs::write(
            &marker,
            r#"
workspace = "acme corp"
project = "infra"
project_strategy = "repo-root"
"#,
        )
        .unwrap();
        let cwd = tmp.path().to_str().unwrap();
        let qs = marker_query_suffix(cwd);
        // cwd is encoded first; marker fields follow in the iteration order
        // of the loop in `marker_query_suffix`.
        assert!(qs.contains("&workspace=acme%20corp"), "{qs}");
        assert!(qs.contains("&project=infra"), "{qs}");
        assert!(qs.contains("&project_strategy=repo-root"), "{qs}");
    }

    #[test]
    fn marker_query_suffix_repo_root_non_git_keeps_project_implicit() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".ai-memory.toml"),
            "workspace = \"oss\"\nproject_strategy = \"repo-root\"\n",
        )
        .unwrap();
        let child = tmp.path().join("plain-dir");
        std::fs::create_dir_all(&child).unwrap();
        let qs = marker_query_suffix(child.to_str().unwrap());
        assert!(qs.contains("&workspace=oss"), "{qs}");
        assert!(!qs.contains("&project="), "{qs}");
        assert!(qs.contains("&project_strategy=repo-root"), "{qs}");
    }

    #[test]
    fn marker_query_suffix_repo_root_collapses_out_of_tree_worktree() {
        if std::process::Command::new("git")
            .arg("--version")
            .status()
            .is_err()
        {
            return;
        }
        let tmp = tempfile::TempDir::new().unwrap();
        let repo = tmp.path().join("repos/acme-api");
        std::fs::create_dir_all(&repo).unwrap();
        assert!(
            std::process::Command::new("git")
                .args(["init", "-q"])
                .arg(&repo)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            std::process::Command::new("git")
                .arg("-C")
                .arg(&repo)
                .args([
                    "-c",
                    "user.email=t@example.com",
                    "-c",
                    "user.name=t",
                    "commit",
                    "-q",
                    "--allow-empty",
                    "-m",
                    "init",
                ])
                .status()
                .unwrap()
                .success()
        );

        let worktrees = tmp.path().join("worktrees");
        std::fs::create_dir_all(&worktrees).unwrap();
        std::fs::write(
            worktrees.join(".ai-memory.toml"),
            "workspace = \"oss\"\nproject_strategy = \"repo-root\"\n",
        )
        .unwrap();
        let wt = worktrees.join("acme-api/wt-feature");
        std::fs::create_dir_all(wt.parent().unwrap()).unwrap();
        if !std::process::Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(["worktree", "add", "-q"])
            .arg(&wt)
            .status()
            .unwrap()
            .success()
        {
            return;
        }

        let qs = marker_query_suffix(wt.to_str().unwrap());
        assert!(qs.contains("&workspace=oss"), "{qs}");
        assert!(qs.contains("&project=acme-api"), "{qs}");
        assert!(qs.contains("&project_strategy=repo-root"), "{qs}");
    }
}

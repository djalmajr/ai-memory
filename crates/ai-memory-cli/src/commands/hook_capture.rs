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
        for key in ["workspace", "project", "project_strategy"] {
            if let Some(val) = parse_toml_key(&marker, key) {
                qs.push_str(&format!("&{key}={}", url_encode(&val)));
            }
        }
    }
    qs
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

/// POST the payload as JSON, best-effort. 0.5s budget (parity with the
/// shell `--max-time 0.5`). Network errors are swallowed — a hook must
/// never block or fail the agent.
pub async fn post_hook(
    client: &reqwest::Client,
    url: &str,
    body: &str,
    token: Option<&str>,
) -> anyhow::Result<()> {
    let mut req = client
        .post(url)
        .header("Content-Type", "application/json")
        .timeout(Duration::from_millis(500))
        .body(body.to_owned());
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let _ = req.send().await; // best-effort: ignore the result
    Ok(())
}

/// GET the handoff text, 1s budget (parity with the shell handoff GET).
/// Returns None on any error or an empty body.
pub async fn get_handoff(
    client: &reqwest::Client,
    url: &str,
    token: Option<&str>,
) -> Option<String> {
    let mut req = client.get(url).timeout(Duration::from_millis(1000));
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let body = req.send().await.ok()?.text().await.ok()?;
    if body.is_empty() { None } else { Some(body) }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    async fn post_hook_returns_ok_even_when_server_unreachable() {
        // Port 1 is unroutable; best-effort means this still resolves Ok.
        let client = build_client();
        let r = post_hook(
            &client,
            "http://127.0.0.1:1/hook?event=pre-tool-use",
            "{}",
            None,
        )
        .await;
        assert!(r.is_ok());
    }
}

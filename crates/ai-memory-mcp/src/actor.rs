//! Per-request actor context — pure helper mapping `X-Memory-Actor-*`
//! HTTP headers to a typed [`ActorContext`].
//!
//! ## Why this exists
//!
//! `mcp-auth` (Go sidecar) validates the Keycloak JWT and injects 5
//! request headers derived from claims. The `memory_write_page` tool
//! method receives the raw HTTP `Parts` via rmcp's `Extension<Parts>`
//! extractor (rmcp 1.7+) and calls [`actor_from_headers`] to build the
//! [`ActorContext`] that feeds into the admission webhook chain.
//!
//! ## Why not `tokio::task_local`
//!
//! We tried that first. `rmcp::transport::streamable_http_server::tower`
//! dispatches each tool handler via `tokio::spawn` (see
//! `tower.rs:569+619+1183`), which **does not** inherit task-locals from
//! the outer axum middleware. The Extension extractor is the official
//! supported path — see
//! <https://docs.rs/rmcp/1.7/rmcp/transport/streamable_http_server/struct.StreamableHttpService.html#accessing-http-request-data-from-tool-handlers>.
//!
//! ## Wire contract
//!
//! Headers expected (set by `mcp-auth` from validated JWT claims):
//!
//! | Header | JWT claim source |
//! |---|---|
//! | `X-Memory-Actor-Agent` | DCR `client_name` (fallback `azp`) |
//! | `X-Memory-Actor-User` | `preferred_username` |
//! | `X-Memory-Actor-Sub` | `sub` |
//! | `X-Memory-Actor-Client` | `azp` (authorized party / DCR client UUID) |
//! | `X-Memory-Actor-Session-Id` | optional |
//!
//! All headers are optional — missing ones fall back to `None`. Use
//! [`ActorContext::has_any`] to skip building an admission context when
//! the actor is completely anonymous (the chain still runs with a
//! default context — webhooks decide what to do with an empty actor).

use ai_memory_core::ActorContext;
use axum::http::HeaderMap;

fn header_str(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Build an [`ActorContext`] from `X-Memory-Actor-*` headers.
///
/// Header names are matched case-insensitively (per HTTP spec).
#[must_use]
pub fn actor_from_headers(headers: &HeaderMap) -> ActorContext {
    ActorContext {
        agent: header_str(headers, "x-memory-actor-agent"),
        user: header_str(headers, "x-memory-actor-user"),
        sub: header_str(headers, "x-memory-actor-sub"),
        client: header_str(headers, "x-memory-actor-client"),
        session_id: header_str(headers, "x-memory-actor-session-id"),
        // `name`/`email` aren't carried by the `X-Memory-Actor-*` bridge;
        // the four-rung auth middleware fills those when it injects the
        // canonical `Extension<ActorContext>` directly.
        ..ActorContext::default()
    }
}

/// Parse the admission-chain loop-prevention skip list from the
/// `X-Memory-Skip-Admission-Chain` request header (comma-separated
/// webhook names). A webhook that writes back into the engine (e.g. via
/// `memory_write_page`) sets this so the chain doesn't re-invoke it on
/// the recursive write — see [`ai_memory_wiki::AdmissionContext::skip_webhooks`].
///
/// Returns an empty `Vec` when the header is absent. Entries are trimmed
/// and empty tokens dropped, so `"a, ,b,"` → `["a", "b"]`.
#[must_use]
pub fn skip_webhooks_from_headers(headers: &HeaderMap) -> Vec<String> {
    headers
        .get("x-memory-skip-admission-chain")
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            s.split(',')
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn full_header_set_maps_correctly() {
        let mut h = HeaderMap::new();
        h.insert(
            "x-memory-actor-agent",
            HeaderValue::from_static("claude-code"),
        );
        h.insert("x-memory-actor-user", HeaderValue::from_static("djalmajr"));
        h.insert("x-memory-actor-sub", HeaderValue::from_static("8f3a-uuid"));
        h.insert(
            "x-memory-actor-client",
            HeaderValue::from_static("72836f52-uuid"),
        );
        h.insert(
            "x-memory-actor-session-id",
            HeaderValue::from_static("019e6d-session"),
        );
        let ctx = actor_from_headers(&h);
        assert_eq!(ctx.agent.as_deref(), Some("claude-code"));
        assert_eq!(ctx.user.as_deref(), Some("djalmajr"));
        assert_eq!(ctx.sub.as_deref(), Some("8f3a-uuid"));
        assert_eq!(ctx.client.as_deref(), Some("72836f52-uuid"));
        assert_eq!(ctx.session_id.as_deref(), Some("019e6d-session"));
        assert!(ctx.has_any());
    }

    #[test]
    fn missing_headers_leave_none() {
        let h = HeaderMap::new();
        let ctx = actor_from_headers(&h);
        assert!(ctx.agent.is_none());
        assert!(ctx.user.is_none());
        assert!(ctx.sub.is_none());
        assert!(ctx.client.is_none());
        assert!(ctx.session_id.is_none());
        assert!(!ctx.has_any());
    }

    #[test]
    fn empty_or_whitespace_header_treated_as_none() {
        let mut h = HeaderMap::new();
        h.insert("x-memory-actor-agent", HeaderValue::from_static("   "));
        h.insert("x-memory-actor-user", HeaderValue::from_static(""));
        let ctx = actor_from_headers(&h);
        assert!(ctx.agent.is_none(), "whitespace must trim to None");
        assert!(ctx.user.is_none(), "empty must be None");
        assert!(!ctx.has_any());
    }

    #[test]
    fn case_insensitive_header_lookup() {
        // HeaderMap normalises names to lowercase on insert; this verifies
        // that the canonical `X-Memory-Actor-Agent` form also resolves.
        let mut h = HeaderMap::new();
        h.insert("X-Memory-Actor-Agent", HeaderValue::from_static("codex"));
        let ctx = actor_from_headers(&h);
        assert_eq!(ctx.agent.as_deref(), Some("codex"));
    }

    #[test]
    fn skip_webhooks_parses_csv_trims_and_drops_empties() {
        let mut h = HeaderMap::new();
        h.insert(
            "x-memory-skip-admission-chain",
            HeaderValue::from_static("contributors, ,git-mirror,"),
        );
        assert_eq!(
            skip_webhooks_from_headers(&h),
            vec!["contributors".to_string(), "git-mirror".to_string()]
        );
    }

    #[test]
    fn skip_webhooks_absent_header_is_empty() {
        let h = HeaderMap::new();
        assert!(skip_webhooks_from_headers(&h).is_empty());
    }
}

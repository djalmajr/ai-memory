//! `ai-memory hook` — emit a single lifecycle event natively.
//!
//! Reads the event payload from stdin. Instead of POSTing synchronously on the
//! agent's hot path (which would block every tool call on the network and drop
//! events against a slow/remote server), the event is **spooled** locally — an
//! instant write — and the spool is drained to the server at session
//! boundaries (a cleanup pass on `session-start`, the main flush on
//! `session-end`). The one synchronous request is the `session-start` handoff
//! GET, whose result is injected back into the agent as context.
//!
//! See `docs/windows.md#native-hook-command-claude-code-on-windows`.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use ai_memory_core::AgentKind;
use ai_memory_llm::OidcToken;

use crate::cli::HookArgs;

use super::hook_capture::{build_client, extract_cwd, get_handoff, marker_query_suffix};
use super::hook_spool;
use super::path_util::strip_windows_verbatim_prefix;

// All drain/handoff timings default to the current short values and can be
// overridden by whole-minute env vars for very high-latency or large-backlog
// instances. Two kinds: per-request timeouts cap each individual POST / handoff
// GET; session-boundary budgets cap how long a boundary spends draining (so a
// boundary never hangs unbounded).
const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(3);
const DEFAULT_HANDOFF_TIMEOUT: Duration = Duration::from_secs(3);
const DEFAULT_START_BUDGET: Duration = Duration::from_secs(3);
const DEFAULT_END_BUDGET: Duration = Duration::from_secs(10);
const MAX_OVERRIDE_MINUTES: u64 = 60;

const DRAIN_TIMEOUT_ENV: &str = "AI_MEMORY_HOOK_DRAIN_TIMEOUT_MINUTES";
const HANDOFF_TIMEOUT_ENV: &str = "AI_MEMORY_HOOK_HANDOFF_TIMEOUT_MINUTES";
const START_BUDGET_ENV: &str = "AI_MEMORY_HOOK_START_BUDGET_MINUTES";
const END_BUDGET_ENV: &str = "AI_MEMORY_HOOK_END_BUDGET_MINUTES";

const INCREMENTAL_THRESHOLD_ENV: &str = "AI_MEMORY_HOOK_INCREMENTAL_THRESHOLD";
/// Backlog size at which `post-tool-use` does a mid-session catch-up drain, so a
/// light session pays only a `read_dir`. Override via the env var above.
const DEFAULT_INCREMENTAL_THRESHOLD: usize = 32;
/// Total budget AND per-event timeout for the mid-session catch-up drain — kept
/// well under a second so a `post-tool-use` hook never stalls a tool call (one
/// in-flight POST against a slow server is bounded by this too).
const INCREMENTAL_DRAIN_BUDGET: Duration = Duration::from_millis(250);

/// Per-event POST timeout during a drain. Env: `AI_MEMORY_HOOK_DRAIN_TIMEOUT_MINUTES`.
fn drain_event_timeout() -> Duration {
    drain_event_timeout_from(env_lookup)
}
/// Synchronous handoff GET timeout. Env: `AI_MEMORY_HOOK_HANDOFF_TIMEOUT_MINUTES`.
fn handoff_timeout() -> Duration {
    handoff_timeout_from(env_lookup)
}
/// Total budget for the `session-start` cleanup drain (kept tight so session
/// start stays snappy even when the server is down — leftovers wait). Env:
/// `AI_MEMORY_HOOK_START_BUDGET_MINUTES`.
fn start_drain_budget() -> Duration {
    start_drain_budget_from(env_lookup)
}
/// Total budget for the `session-end` flush (the main delivery point; a session
/// boundary tolerates more). Env: `AI_MEMORY_HOOK_END_BUDGET_MINUTES`.
fn end_drain_budget() -> Duration {
    end_drain_budget_from(env_lookup)
}

fn drain_event_timeout_from(lookup: impl FnMut(&str) -> Option<String>) -> Duration {
    env_minutes(DRAIN_TIMEOUT_ENV, DEFAULT_DRAIN_TIMEOUT, lookup)
}

fn handoff_timeout_from(lookup: impl FnMut(&str) -> Option<String>) -> Duration {
    env_minutes(HANDOFF_TIMEOUT_ENV, DEFAULT_HANDOFF_TIMEOUT, lookup)
}

fn start_drain_budget_from(lookup: impl FnMut(&str) -> Option<String>) -> Duration {
    env_minutes(START_BUDGET_ENV, DEFAULT_START_BUDGET, lookup)
}

fn end_drain_budget_from(lookup: impl FnMut(&str) -> Option<String>) -> Duration {
    env_minutes(END_BUDGET_ENV, DEFAULT_END_BUDGET, lookup)
}

/// Backlog size at which `post-tool-use` triggers a mid-session catch-up drain.
/// Env: `AI_MEMORY_HOOK_INCREMENTAL_THRESHOLD` (positive integer).
fn incremental_drain_threshold() -> usize {
    incremental_drain_threshold_from(env_lookup)
}

fn incremental_drain_threshold_from(mut lookup: impl FnMut(&str) -> Option<String>) -> usize {
    lookup(INCREMENTAL_THRESHOLD_ENV)
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_INCREMENTAL_THRESHOLD)
}

/// Whether to run a mid-session catch-up drain for this event: only
/// `post-tool-use` (the highest-frequency event) and only once the spool backlog
/// has crossed `threshold`. Boundaries (`session-start`/`session-end`) remain
/// the main flush, so a light session never drains mid-session.
fn should_incremental_drain(event: &str, spool_len: usize, threshold: usize) -> bool {
    event == "post-tool-use" && spool_len >= threshold
}

/// When the `session-end` drain cannot clear the spool, return a concise,
/// actionable note. `None` when nothing remains queued or dropped, so a normal
/// session stays silent. The caller writes this to **stderr** (never stdout,
/// which carries the hook's JSON protocol): mirroring the spool's at-capacity
/// warning, an expected-but-noteworthy backlog is surfaced rather than left
/// silent, and the message names the two knobs the operator can turn instead of
/// the bare, scary cancelled-hook symptom.
fn session_end_deferred_note(result: &hook_spool::DrainResult) -> Option<String> {
    if result.remaining == 0 && result.dropped == 0 {
        return None;
    }

    let mut note = format!(
        "ai-memory: session-end flushed {} event(s); {} still queued for a later \
         session boundary.",
        result.sent, result.remaining,
    );
    if result.dropped > 0 {
        note.push_str(&format!(
            " {} event(s) were dropped as undeliverable after exhausting the retry budget.",
            result.dropped,
        ));
    } else {
        note.push_str(" No queued data was lost.");
    }
    note.push_str(&format!(
        " Raise {END_BUDGET_ENV} (whole minutes), lower {INCREMENTAL_THRESHOLD_ENV}, \
         or check server reachability to keep the backlog bounded."
    ));
    Some(note)
}

fn env_lookup(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Read a positive-integer minute override from `name`, falling back to the
/// built-in short default for missing / empty / non-numeric / zero values. Clamp
/// large values so a typo cannot block a hook boundary for hours or days.
fn env_minutes(
    name: &str,
    default: Duration,
    mut lookup: impl FnMut(&str) -> Option<String>,
) -> Duration {
    parse_minutes(lookup(name), default)
}

fn parse_minutes(raw: Option<String>, default: Duration) -> Duration {
    let minutes = raw
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .map(|n| n.min(MAX_OVERRIDE_MINUTES));
    match minutes {
        Some(n) => Duration::from_secs(n * 60),
        None => default,
    }
}

/// Run a single hook end-to-end. Always returns Ok and always writes a JSON
/// object to stdout — a hook must never fail the agent.
///
/// `data_dir` is the resolved global `--data-dir` (if any); used to locate the
/// spool and the stored OIDC token.
pub async fn run(data_dir: Option<PathBuf>, args: HookArgs) -> anyhow::Result<()> {
    let mut payload = String::new();
    std::io::stdin().read_to_string(&mut payload).ok();
    let json: serde_json::Value = serde_json::from_str(&payload).unwrap_or(serde_json::Value::Null);

    // Client-side opt-out: with AI_MEMORY_DROP_SUBAGENT_CAPTURES set, skip
    // marker-bearing captures from a SUBAGENT session at the source — never
    // spool or send them — so a multi-agent harness's subagent swarm cannot
    // fill the local spool or hammer the server. Shares the env var (and marker
    // detection) with the server-side `drop_subagent_captures`. The subagent
    // BOUNDARY events (`subagent-start`/`subagent-stop`) are deliberately NOT
    // dropped: forwarding them lets the server seed/clear its tail tracking and
    // drop the unmarked tail too — source-dropping them would blind that and
    // leak the tail. A hook must always emit a JSON object on stdout, so write
    // the empty no-op response and return.
    if drop_subagent_captures_enabled() && should_source_drop_subagent(&args.event, &json) {
        println!("{{}}");
        return Ok(());
    }

    // Outbound privacy scrub: with AI_MEMORY_SANITIZE_OUTBOUND set, redact
    // secrets/PII from the payload BEFORE it is spooled or sent, so they never
    // leave the host or sit plaintext in the spool. Scrubs JSON string *values*
    // (keys/structure preserved) with the built-in patterns only — the
    // fast-path skips config load, and the server applies the full sanitize
    // (built-in + operator extras) as a backstop.
    let payload = maybe_scrub_outbound(payload, &json);

    let qs = extract_cwd(&json)
        .map(|cwd| marker_query_suffix(&cwd, args.project_strategy.and_then(|s| s.baked())))
        .unwrap_or_default();
    let base = args.server_url.trim_end_matches('/');
    let dd = resolve_data_dir(data_dir.as_deref());
    let spool = hook_spool::spool_dir(&dd);

    // Spool THIS event — an instant local write, never the network. The auth
    // mode is decided without a round-trip: an explicit `--auth-token` is
    // stored inline; otherwise a present OIDC token marks the event `oidc`
    // (resolved + refreshed at drain time); otherwise anonymous.
    let oidc_present = args.auth_token.is_none()
        && OidcToken::load(&dd.join("auth.json"))
            .ok()
            .flatten()
            .is_some();
    let event_url = format!("{base}/hook?event={}&agent={}{qs}", args.event, args.agent);
    let entry = hook_spool::entry_for(
        event_url,
        payload.clone(),
        args.auth_token.as_deref(),
        oidc_present,
    );
    if hook_spool::enqueue(&spool, &entry).is_err() {
        eprintln!(
            "ai-memory hook warning: failed to spool lifecycle event; capture for this event was skipped"
        );
    }

    // Mid-session catch-up: per-event hooks only enqueue, so a heavy session
    // outpaces the boundary-only drain and the spool grows until the next
    // boundary. On `post-tool-use`, once the backlog crosses the threshold, do a
    // tightly time-boxed drain (budget == per-event timeout, sub-second) so the
    // spool stays flat without ever stalling a tool call.
    if should_incremental_drain(
        &args.event,
        hook_spool::spool_len(&spool),
        incremental_drain_threshold(),
    ) {
        let _ = hook_spool::drain(
            &spool,
            &dd,
            INCREMENTAL_DRAIN_BUDGET,
            INCREMENTAL_DRAIN_BUDGET,
        )
        .await;
    }

    // session-start: drain any backlog (e.g. from a previous session that ended
    // abruptly), then fetch + inject the pending handoff for the resuming agent.
    if args.event == "session-start" {
        let _ = hook_spool::drain(&spool, &dd, start_drain_budget(), drain_event_timeout()).await;
        // Only fetch the handoff for agents that inject the session-start
        // hook's stdout as context. Grok ignores it, so fetching here would
        // consume the handoff server-side (the GET is destructive) and then
        // discard the result — silently losing it. Those agents recover the
        // handoff on demand via the MCP `memory_handoff_accept` tool.
        if AgentKind::from_wire(&args.agent).session_start_injects_handoff() {
            let client = build_client();
            let bearer = hook_spool::resolve_bearer(&client, &dd, args.auth_token.as_deref()).await;
            let handoff_url = format!("{base}/handoff?agent={}{qs}", args.agent);
            if let Some(handoff) =
                get_handoff(&client, &handoff_url, bearer.as_deref(), handoff_timeout()).await
            {
                let envelope = serde_json::json!({
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": handoff,
                    }
                });
                println!("{envelope}");
                return Ok(());
            }
        }
    }

    // session-end: the main delivery point — flush the session's spooled
    // observations (oldest-first) so the server has them before it consolidates.
    if args.event == "session-end" {
        let result =
            hook_spool::drain(&spool, &dd, end_drain_budget(), drain_event_timeout()).await;
        if let Some(note) = session_end_deferred_note(&result) {
            eprintln!("{note}");
        }
    }

    println!("{{}}");
    Ok(())
}

/// True when `AI_MEMORY_DROP_SUBAGENT_CAPTURES` is set to a truthy value. The
/// hook fast-path skips full config loading for latency, so it reads the env
/// directly; the value mirrors the server-side `drop_subagent_captures`.
fn drop_subagent_captures_enabled() -> bool {
    std::env::var("AI_MEMORY_DROP_SUBAGENT_CAPTURES")
        .ok()
        .is_some_and(|value| is_truthy(&value))
}

/// True for the subagent lifecycle *boundary* events (`subagent-start` /
/// `subagent-stop`). These must never be source-dropped: they carry no bulk
/// payload, and the server's stateful subagent-session tracking seeds on
/// `subagent-start` (and clears on `subagent-stop`) so it can drop the
/// *unmarked* tail (`user-prompt-submit` / `stop` / `session-end`) of a
/// subagent session. Dropping them at the source would blind that tracking and
/// let the tail persist despite the opt-out.
fn is_subagent_boundary_event(event: &str) -> bool {
    matches!(
        event.trim().to_ascii_lowercase().as_str(),
        "subagent-start" | "subagent_start" | "subagent-stop" | "subagent_stop"
    )
}

/// Whether `ai-memory hook` should source-drop this event under
/// `AI_MEMORY_DROP_SUBAGENT_CAPTURES`: only marker-bearing, *non-boundary*
/// captures. Boundary events are always forwarded so the server can seed/clear
/// its tail tracking; unmarked events are forwarded so the server closes the
/// tail. Mirrors the server-side detection (`body_is_subagent`).
fn should_source_drop_subagent(event: &str, json: &serde_json::Value) -> bool {
    !is_subagent_boundary_event(event) && ai_memory_hooks::body_is_subagent(json)
}

/// Parse a boolean-ish env value: `1` / `true` / `yes` / `on` (case-insensitive).
fn is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// True when `AI_MEMORY_SANITIZE_OUTBOUND` is set to a truthy value — scrub the
/// captured payload with the built-in privacy patterns before it leaves the
/// host. Client-only (the server already sanitizes on store); mirrors the
/// env-driven, config-skipping style of the rest of the hook fast-path.
fn sanitize_outbound_enabled() -> bool {
    std::env::var("AI_MEMORY_SANITIZE_OUTBOUND")
        .ok()
        .is_some_and(|value| is_truthy(&value))
}

/// Redact secrets/PII from the outbound payload when enabled. Scrubs the JSON
/// string *values* (keys and structure preserved) so redaction can never
/// corrupt the document; if the payload is not valid JSON there is no structure
/// to protect, so the raw text is scrubbed directly. Built-in patterns only.
fn maybe_scrub_outbound(payload: String, json: &serde_json::Value) -> String {
    if !sanitize_outbound_enabled() {
        return payload;
    }
    let sanitizer = ai_memory_core::Sanitizer::builtin();
    if json.is_null() {
        return sanitizer.scrub(&payload);
    }
    let mut scrubbed = json.clone();
    scrub_json_strings(&mut scrubbed, &sanitizer);
    serde_json::to_string(&scrubbed).unwrap_or(payload)
}

/// Recursively scrub every JSON string leaf in place. Object keys are left
/// untouched (they are field names, not values) so downstream field extraction
/// keeps working.
fn scrub_json_strings(value: &mut serde_json::Value, sanitizer: &ai_memory_core::Sanitizer) {
    match value {
        serde_json::Value::String(s) => *s = sanitizer.scrub(s),
        serde_json::Value::Array(items) => {
            for item in items {
                scrub_json_strings(item, sanitizer);
            }
        }
        serde_json::Value::Object(map) => {
            for (_key, v) in map.iter_mut() {
                scrub_json_strings(v, sanitizer);
            }
        }
        _ => {}
    }
}

/// Resolve the data dir cheaply, without loading the full config (the hook
/// fast-path skips config for latency). Mirrors `config.rs`: explicit
/// `--data-dir`, else `AI_MEMORY_DATA_DIR`, else the platform local-data dir.
fn resolve_data_dir(data_dir: Option<&Path>) -> PathBuf {
    let dir = data_dir
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("AI_MEMORY_DATA_DIR").map(PathBuf::from))
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("ai-memory")
        });
    // Recover already-installed hooks that baked a safe verbatim data-dir form.
    match dir.to_str() {
        Some(s) if s.starts_with(r"\\?\") => {
            PathBuf::from(strip_windows_verbatim_prefix(s).into_owned())
        }
        _ => dir,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_truthy_accepts_common_boolean_forms() {
        for v in ["1", "true", "TRUE", " yes ", "On"] {
            assert!(is_truthy(v), "{v:?} should be truthy");
        }
        for v in ["0", "false", "no", "off", "", "maybe"] {
            assert!(!is_truthy(v), "{v:?} should not be truthy");
        }
    }

    #[test]
    fn source_drop_forwards_subagent_boundary_events() {
        let marked = serde_json::json!({ "subagentType": "general-purpose" });
        // Boundary events are forwarded even when marker-bearing, so the server
        // can seed/clear its tail tracking and drop the unmarked tail.
        assert!(!should_source_drop_subagent("subagent-start", &marked));
        assert!(!should_source_drop_subagent("subagent-stop", &marked));
        assert!(!should_source_drop_subagent("subagent_start", &marked));
        // A marker-bearing NON-boundary capture (the bulk) is still dropped.
        assert!(should_source_drop_subagent("pre-tool-use", &marked));
        // An unmarked event is never source-dropped — the server, seeded by the
        // forwarded boundary event, closes the tail itself.
        let unmarked = serde_json::json!({ "prompt": "go" });
        assert!(!should_source_drop_subagent("user-prompt-submit", &unmarked));
        assert!(!should_source_drop_subagent("session-end", &unmarked));
    }

    #[test]
    fn scrub_json_strings_redacts_values_keeps_structure() {
        let sanitizer = ai_memory_core::Sanitizer::builtin();
        let mut v = serde_json::json!({
            "session_id": "abc-123",
            "toolInput": { "command": "deploy", "token": "sk-ABCDEFGHIJ1234567890abcd" },
            "args": ["plain", "ghp_ABCDEFGHIJ1234567890XY"]
        });
        scrub_json_strings(&mut v, &sanitizer);
        // Non-secret values and the whole structure survive.
        assert_eq!(v["session_id"], "abc-123");
        assert_eq!(v["toolInput"]["command"], "deploy");
        assert_eq!(v["args"][0], "plain");
        assert!(v.get("toolInput").is_some(), "keys/structure preserved");
        // Secret values (object + array) are redacted.
        assert_eq!(v["toolInput"]["token"], "[REDACTED]");
        assert_eq!(v["args"][1], "[REDACTED]");
    }

    #[test]
    fn resolve_data_dir_strips_verbatim_prefix_from_baked_arg() {
        // Recover safe verbatim data dirs baked by older installs (#116).
        let resolved =
            resolve_data_dir(Some(Path::new(r"\\?\C:\Users\me\AppData\Local\ai-memory")));
        assert_eq!(
            resolved,
            PathBuf::from(r"C:\Users\me\AppData\Local\ai-memory")
        );
    }

    #[test]
    fn resolve_data_dir_leaves_plain_path_untouched() {
        let resolved = resolve_data_dir(Some(Path::new(r"C:\Users\me\ai-memory")));
        assert_eq!(resolved, PathBuf::from(r"C:\Users\me\ai-memory"));
    }

    #[test]
    fn should_incremental_drain_only_post_tool_use_over_threshold() {
        assert!(should_incremental_drain("post-tool-use", 32, 32));
        assert!(should_incremental_drain("post-tool-use", 100, 32));
        // below threshold: a light session never drains mid-session
        assert!(!should_incremental_drain("post-tool-use", 31, 32));
        // other events only enqueue; boundaries do the real flush
        assert!(!should_incremental_drain("pre-tool-use", 999, 32));
        assert!(!should_incremental_drain("session-start", 999, 32));
        assert!(!should_incremental_drain("session-end", 999, 32));
        assert!(!should_incremental_drain("stop", 999, 32));
    }

    #[test]
    fn incremental_threshold_parses_and_falls_back() {
        assert_eq!(incremental_drain_threshold_from(|_| Some("64".into())), 64);
        assert_eq!(
            incremental_drain_threshold_from(|_| None),
            DEFAULT_INCREMENTAL_THRESHOLD
        );
        // zero / non-numeric fall back to the default (a 0 threshold would drain
        // on every post-tool-use)
        assert_eq!(
            incremental_drain_threshold_from(|_| Some("0".into())),
            DEFAULT_INCREMENTAL_THRESHOLD
        );
        assert_eq!(
            incremental_drain_threshold_from(|_| Some("abc".into())),
            DEFAULT_INCREMENTAL_THRESHOLD
        );
    }

    #[test]
    fn session_end_note_is_silent_when_nothing_deferred() {
        // A fully-drained boundary (the normal case) must stay quiet so a light
        // session never prints a spurious warning.
        let clean = hook_spool::DrainResult {
            sent: 12,
            remaining: 0,
            dropped: 0,
        };
        assert!(session_end_deferred_note(&clean).is_none());
    }

    #[test]
    fn session_end_note_reports_deferred_backlog_and_knobs() {
        // The heavy-session condition from issue #130: a boundary drain
        // delivers some events and leaves the rest queued. The note must report
        // both counts and name the two knobs the issue's own workaround used.
        let backlog = hook_spool::DrainResult {
            sent: 500,
            remaining: 1384,
            dropped: 0,
        };
        let note = session_end_deferred_note(&backlog).expect("a backlog must produce a note");
        assert!(note.contains("500"), "reports how many were flushed");
        assert!(note.contains("1384"), "reports how many were deferred");
        assert!(
            note.contains(END_BUDGET_ENV),
            "points at the session-end budget knob"
        );
        assert!(
            note.contains(INCREMENTAL_THRESHOLD_ENV),
            "points at the mid-session threshold knob"
        );
        assert!(note.contains("No queued data was lost"));
    }

    #[test]
    fn session_end_note_reports_dropped_events_without_promising_no_loss() {
        let result = hook_spool::DrainResult {
            sent: 4,
            remaining: 2,
            dropped: 1,
        };

        let note = session_end_deferred_note(&result).expect("drops must produce a note");

        assert!(note.contains("4"), "reports delivered count");
        assert!(note.contains("2"), "reports queued count");
        assert!(note.contains("1"), "reports dropped count");
        assert!(note.contains("dropped as undeliverable"));
        assert!(!note.contains("No queued data was lost"));
    }

    #[test]
    fn parse_minutes_falls_back_on_invalid() {
        assert_eq!(
            parse_minutes(None, DEFAULT_DRAIN_TIMEOUT),
            DEFAULT_DRAIN_TIMEOUT
        );
        assert_eq!(
            parse_minutes(Some(String::new()), DEFAULT_DRAIN_TIMEOUT),
            DEFAULT_DRAIN_TIMEOUT
        );
        assert_eq!(
            parse_minutes(Some("abc".into()), DEFAULT_DRAIN_TIMEOUT),
            DEFAULT_DRAIN_TIMEOUT
        );
        // Zero is rejected (a 0-minute timeout would drop every request).
        assert_eq!(
            parse_minutes(Some("0".into()), DEFAULT_DRAIN_TIMEOUT),
            DEFAULT_DRAIN_TIMEOUT
        );
    }

    #[test]
    fn parse_minutes_honours_valid_override() {
        assert_eq!(
            parse_minutes(Some("2".into()), DEFAULT_DRAIN_TIMEOUT),
            Duration::from_secs(120)
        );
        assert_eq!(
            parse_minutes(Some("  3 ".into()), DEFAULT_DRAIN_TIMEOUT),
            Duration::from_secs(180)
        );
    }

    #[test]
    fn parse_minutes_clamps_large_values() {
        assert_eq!(
            parse_minutes(Some("999".into()), DEFAULT_DRAIN_TIMEOUT),
            Duration::from_secs(MAX_OVERRIDE_MINUTES * 60)
        );
    }

    #[test]
    fn timing_accessors_read_the_expected_env_vars() {
        fn one_minute_for(expected_name: &'static str) -> impl FnMut(&str) -> Option<String> {
            move |actual_name| {
                assert_eq!(actual_name, expected_name);
                Some("1".to_string())
            }
        }

        assert_eq!(
            drain_event_timeout_from(one_minute_for(DRAIN_TIMEOUT_ENV)),
            Duration::from_secs(60)
        );
        assert_eq!(
            handoff_timeout_from(one_minute_for(HANDOFF_TIMEOUT_ENV)),
            Duration::from_secs(60)
        );
        assert_eq!(
            start_drain_budget_from(one_minute_for(START_BUDGET_ENV)),
            Duration::from_secs(60)
        );
        assert_eq!(
            end_drain_budget_from(one_minute_for(END_BUDGET_ENV)),
            Duration::from_secs(60)
        );
    }
}

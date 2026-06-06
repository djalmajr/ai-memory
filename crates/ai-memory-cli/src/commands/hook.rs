//! `ai-memory hook` — emit a single lifecycle event natively.
//!
//! Reads the event payload from stdin and issues the same HTTP request
//! the shell hook scripts do, without spawning a shell or child
//! processes. See `docs/windows-native-hooks.md`.

use std::io::Read;

use crate::cli::HookArgs;

use super::hook_capture::{build_client, extract_cwd, get_handoff, marker_query_suffix, post_hook};

/// Run a single hook end-to-end. Always returns Ok and always writes a
/// JSON object to stdout — a hook must never fail the agent.
pub async fn run(args: HookArgs) -> anyhow::Result<()> {
    let mut payload = String::new();
    std::io::stdin().read_to_string(&mut payload).ok();
    let json: serde_json::Value = serde_json::from_str(&payload).unwrap_or(serde_json::Value::Null);

    let qs = extract_cwd(&json)
        .map(|cwd| marker_query_suffix(&cwd))
        .unwrap_or_default();
    let base = args.server_url.trim_end_matches('/');
    let token = args.auth_token.as_deref();
    let client = build_client();

    // Every event records itself via POST /hook (best-effort).
    let post_url = format!("{base}/hook?event={}&agent={}{qs}", args.event, args.agent);
    let _ = post_hook(&client, &post_url, &payload, token).await;

    // session-start ALSO pulls the pending handoff and injects it as
    // context for the resuming agent.
    if args.event == "session-start" {
        let handoff_url = format!("{base}/handoff?agent={}{qs}", args.agent);
        if let Some(handoff) = get_handoff(&client, &handoff_url, token).await {
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

    println!("{{}}");
    Ok(())
}

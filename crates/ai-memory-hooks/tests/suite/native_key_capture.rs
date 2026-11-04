//! A structured tool response whose `stdout` holds a title, a real line
//! break and a native key: the envelope's compact JSON serialization turns
//! the break into the two characters `\` and `n`, and the server sanitizer
//! must still redact the key — and its two halves — there. This is the shape
//! `HookEnvelope::from_query_and_body` produces for a closed tool event
//! (OpenCode's `payload.tool_response` object), scrubbed the way the router's
//! `durable_body` scrubs the excerpt before the per-event cap.

use ai_memory_core::Sanitizer;
use ai_memory_hooks::payload::{HookEnvelope, HookQuery};

/// A synthetic key in the native `aim_` shape; built to be unmistakably fake.
const KEY: &str = "aim_FAKEfake-FAKEfake_FAKEfakeFAKEfakeFAKEfake000";

/// Build the envelope the router builds for an OpenCode PostToolUse event and
/// scrub its body excerpt the way `router::durable_body` does.
fn scrubbed_body(stdout: &str) -> String {
    let query = HookQuery {
        event: "post-tool-use".into(),
        agent: Some("opencode".into()),
        session_id: Some("native-key-capture".into()),
        cwd: Some("/work/demo".into()),
        ..Default::default()
    };
    let raw = serde_json::json!({
        "session_id": "native-key-capture",
        "cwd": "/work/demo",
        "tool": "exec",
        "callID": "call-nkc-1",
        "payload": {
            "tool_response": {
                "stdout": stdout,
            },
        },
    });
    let envelope = HookEnvelope::from_query_and_body(query, raw);
    let excerpt = envelope
        .body_excerpt
        .clone()
        .expect("the structured tool response must yield a body excerpt");
    Sanitizer::builtin().scrub(&excerpt)
}

#[test]
fn structured_stdout_key_after_a_line_break_is_redacted_whole() {
    let out = scrubbed_body(&format!("title\n{KEY}"));
    assert!(
        out.contains("[REDACTED:ai_memory_token]"),
        "the escaped-break excerpt must be redacted: {out:?}"
    );
    assert!(!out.contains(KEY), "the full key must not survive: {out:?}");
    let (head, tail) = KEY.split_at(KEY.len() / 2);
    assert!(
        !out.contains(head) && !out.contains(tail),
        "no half of the key may survive: {out:?}"
    );
    // The router scrubs the excerpt and then `Sanitized::new` scrubs the
    // observation: the second pass must be a no-op, not a fresh leak.
    assert_eq!(
        Sanitizer::builtin().scrub(&out),
        out,
        "second pass: {out:?}"
    );
}

/// Control: the same excerpt with a space where the break was is redacted
/// too — the excerpt itself, not the escape, is what the sanitizer sees.
#[test]
fn structured_stdout_key_after_a_space_is_redacted_whole() {
    let out = scrubbed_body(&format!("title {KEY}"));
    assert!(
        out.contains("[REDACTED:ai_memory_token]"),
        "the space-delimited excerpt must be redacted: {out:?}"
    );
    assert!(!out.contains(KEY), "the full key must not survive: {out:?}");
    let (head, tail) = KEY.split_at(KEY.len() / 2);
    assert!(
        !out.contains(head) && !out.contains(tail),
        "no half of the key may survive: {out:?}"
    );
}

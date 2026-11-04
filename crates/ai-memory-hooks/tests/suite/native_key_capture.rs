//! A structured tool response whose `stdout` holds a title, a real line
//! break and a native key: the envelope's compact JSON serialization turns
//! the break into the two characters `\` and `n`, and the server sanitizer
//! must still redact the key — and its two halves — there. This is the shape
//! `HookEnvelope::from_query_and_body` produces for a closed tool event
//! (OpenCode's `payload.tool_response` object), scrubbed the way the router's
//! `durable_body` scrubs the excerpt before the per-event cap.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use ai_memory_core::{AgentKind, IngestMetrics, NewSession, ObservationKind, Sanitizer, SessionId};
use ai_memory_hooks::payload::{HookEnvelope, HookQuery};
use ai_memory_store::Store;
use ai_memory_wiki::Wiki;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tempfile::TempDir;
use tower::ServiceExt;

use crate::router::{
    CwdProbes, DEFAULT_HOOK_INGEST_MAX_IN_FLIGHT, HookState, IngestGates, IngestRateLimiter,
    ProjectCacheStore, SubagentSessionSet, hook_router,
};

/// A synthetic key in the native `aim_` shape; built to be unmistakably fake.
const KEY: &str = "aim_FAKEfake-FAKEfake_FAKEfakeFAKEfakeFAKEfake000";

const WORKSPACE: &str = "default";
const PROJECT: &str = "native-key-route";
const CWD: &str = "/work/demo";

/// Build the envelope the router builds for an OpenCode PostToolUse event and
/// scrub its body excerpt the way `router::durable_body` does.
fn scrubbed_body(stdout: &str) -> String {
    let query = HookQuery {
        event: "post-tool-use".into(),
        agent: Some("opencode".into()),
        session_id: Some("native-key-capture".into()),
        cwd: Some(CWD.into()),
        ..Default::default()
    };
    let raw = serde_json::json!({
        "session_id": "native-key-capture",
        "cwd": CWD,
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

/// One hook harness: a real store, a real wiki and the router state the
/// `/hook` route needs (same shape as the `idle_end` suite harness).
async fn harness() -> (TempDir, Store, HookState) {
    let tmp = TempDir::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    let ws = store
        .writer
        .get_or_create_workspace(WORKSPACE)
        .await
        .unwrap();
    let proj = store
        .writer
        .get_or_create_project(ws, PROJECT, None)
        .await
        .unwrap();
    let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();
    let state = HookState {
        ingest_metrics: Arc::new(IngestMetrics::default()),
        workspace_id: ws,
        project_id: proj,
        writer: store.writer.clone(),
        reader: store.reader.clone(),
        wiki,
        consolidator: None,
        sanitizer: Sanitizer::default(),
        project_cache: Arc::new(tokio::sync::Mutex::new(ProjectCacheStore::default())),
        active_project: ai_memory_core::ActiveProject::new(),
        consolidate_on_session_end: true,
        session_consolidation_notify: None,
        profile_notify: None,
        capture_assistant_enabled: false,
        claim_handoff_on_session_start: true,
        create_handoff_on_session_end: true,
        subagent_sessions: Arc::new(tokio::sync::Mutex::new(SubagentSessionSet::default())),
        ingest_rate: Arc::new(tokio::sync::Mutex::new(IngestRateLimiter::disabled())),
        home_dir: None,
        trusted_proxy_identity: false,
        ingest_semaphore: Arc::new(tokio::sync::Semaphore::new(
            DEFAULT_HOOK_INGEST_MAX_IN_FLIGHT,
        )),
        ingest_gates: IngestGates::default(),
        cwd_probes: CwdProbes::default(),
        per_user_slots: false,
        mid_session_routing: ai_memory_core::MidSessionRouting::default(),
        profile: ai_memory_core::profile::ProfileSettings::default(),
        ingest_drop_list: crate::drop_list::IngestDropList::default(),
        record_tool_names: false,
        skip_wiki_checkpoints: true,
    };
    (tmp, store, state)
}

/// Begin a session the way a lifecycle start would, so the tool event has an
/// open session to attach to.
async fn begin(store: &Store, state: &HookState) -> SessionId {
    let sid = SessionId::new();
    store
        .writer
        .begin_session(NewSession {
            id: sid,
            workspace_id: state.workspace_id,
            project_id: state.project_id,
            agent_kind: AgentKind::OpenCode,
            cwd: Some(CWD.into()),
            actor_user: None,
            occurred_at: None,
        })
        .await
        .unwrap();
    sid
}

/// POST a real `/hook` PostToolUse with an OpenCode-shaped `tool_response`
/// whose `stdout` is `stdout`, and return the HTTP status of the acceptance.
async fn post_tool_use(state: &HookState, sid: SessionId, stdout: &str) -> StatusCode {
    let app = hook_router(state.clone());
    let body = serde_json::json!({
        "session_id": sid.to_string(),
        "cwd": CWD,
        "tool": "exec",
        "callID": "call-nkc-route",
        "payload": {
            "tool_response": {
                "stdout": stdout,
            },
        },
    });
    let req = Request::builder()
        .method("POST")
        .uri(format!(
            "/hook?event=post-tool-use&agent=opencode&session_id={sid}&cwd={CWD}&workspace={WORKSPACE}&project={PROJECT}"
        ))
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    app.oneshot(req).await.unwrap().status()
}

/// Poll an async condition (bounded) until it holds; returns whether it did.
///
/// The hook route returns 202 as soon as the effect task is SPAWNED (the
/// hook path is fire-and-forget: no completion barrier is offered), so a
/// test that must observe the effect's store state has no deterministic
/// hand-off point to wait on. Bounded polling against a monotone condition
/// (a row appears) is the faithful substitute.
async fn until<'a, F>(mut cond: F) -> bool
where
    F: FnMut() -> Pin<Box<dyn Future<Output = bool> + Send + 'a>>,
{
    for _ in 0..200 {
        if cond().await {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    cond().await
}

/// The re-revision's canonical shape, end to end: the `stdout` of a
/// structured tool response holds a real control character and a native key.
/// The envelope's compact JSON serialization writes the control out as the
/// literal `\u00XX` escape (or `\u001b[32m` for an ESC + colour code), and
/// `strip_control` does not touch that text — only the native rule's
/// serialized-control prefix can reach the key. The observation persisted by
/// the real `/hook` route must carry the marker and no part of the key.
#[tokio::test]
async fn a_native_key_after_a_serialized_control_reaches_sqlite_redacted() {
    let (_tmp, store, state) = harness().await;
    let sid = begin(&store, &state).await;

    // Real control characters in the `stdout`; the envelope serializes each
    // one the way `serde_json` spells it.
    let cases = [
        ("real vertical tab", format!("title\n\x0b{KEY}")),
        (
            "real ESC + CSI colour code",
            format!("title\u{1b}[32m{KEY}"),
        ),
        // Control: the real LF is what the original rule already covered.
        ("real line break", format!("title\n{KEY}")),
    ];
    for (index, (what, stdout)) in cases.into_iter().enumerate() {
        let status = post_tool_use(&state, sid, &stdout).await;
        assert_eq!(status, StatusCode::ACCEPTED, "{what}: the hook must accept");

        // Fire-and-forget: wait for this case's own row. The count must
        // reach `index + 1`, not merely one — after the first case, an
        // `any(PostToolUse)` wait would be satisfied by the previous row and
        // the assertions below would run on that row instead of the one this
        // case just sent, so a leaked row could never fail the test.
        let wanted = index + 1;
        let persisted = until(|| {
            let reader = store.reader.clone();
            Box::pin(async move {
                reader
                    .observations_for_session(sid)
                    .await
                    .unwrap()
                    .iter()
                    .filter(|o| o.kind == ObservationKind::PostToolUse)
                    .count()
                    >= wanted
            })
        })
        .await;
        assert!(persisted, "{what}: the observation row never appeared");

        let rows = store
            .reader
            .observations_for_session(sid)
            .await
            .unwrap()
            .into_iter()
            .filter(|o| o.kind == ObservationKind::PostToolUse)
            .collect::<Vec<_>>();
        assert!(rows.len() >= wanted, "{what}: fewer tool rows than sent");
        // This case's row is the newest one.
        let body = &rows.last().unwrap().body;
        assert!(
            body.contains("[REDACTED:ai_memory_token]"),
            "{what}: the persisted line carries no marker: {body:?}"
        );
        assert!(
            !body.contains(KEY),
            "{what}: the full key reached SQLite: {body:?}"
        );
        let (head, tail) = KEY.split_at(KEY.len() / 2);
        assert!(
            !body.contains(head) && !body.contains(tail),
            "{what}: a half of the key reached SQLite: {body:?}"
        );
    }
}

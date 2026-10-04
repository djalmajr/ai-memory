//! Passive provider-health recording wrappers.
//!
//! These wrappers never probe providers on their own. They only record
//! the result of calls the server was already going to make.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use jiff::Timestamp;
use serde::{Deserialize, Serialize};

use crate::embedding::Embedder;
use crate::error::{LlmError, LlmResult};
use crate::provider::LlmProvider;
use crate::types::{ChatRequest, ChatResponse, LlmOperationId};

const MAX_ERROR_MESSAGE_CHARS: usize = 1024;

/// Process-scoped health recorder for the configured provider roles.
#[derive(Clone, Default)]
pub struct ProviderHealth {
    llm: ProviderRoleHealth,
    embedding: ProviderRoleHealth,
    /// The configured LLM provider, held only so `snapshot()` can read its
    /// [`LlmProvider::candidate_health`] — a plain provider always returns
    /// empty here, a [`crate::fallback::FallbackLlmProvider`] returns its
    /// ordered candidate list. Never dereferenced for anything but that
    /// read, so this adds no behavior beyond passive reporting.
    llm_provider: Arc<Mutex<Option<Arc<dyn LlmProvider>>>>,
}

impl ProviderHealth {
    /// Return a serializable snapshot of the latest recorded state.
    #[must_use]
    pub fn snapshot(&self) -> ProviderHealthSnapshot {
        let llm_candidates = self
            .llm_provider
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|provider| provider.candidate_health())
            .unwrap_or_default();
        ProviderHealthSnapshot {
            llm: self.llm.snapshot(),
            llm_candidates,
            embedding: self.embedding.snapshot(),
        }
    }

    /// Mark the LLM role as configured and wrap it with passive recording.
    #[must_use]
    pub fn wrap_llm_provider(
        &self,
        inner: Arc<dyn LlmProvider>,
        provider: impl Into<String>,
        model: impl Into<String>,
        retry_hint: Option<String>,
    ) -> Arc<dyn LlmProvider> {
        let health = self.llm.clone();
        health.configure(provider.into(), model.into(), None, retry_hint);
        *self.llm_provider.lock().unwrap_or_else(|e| e.into_inner()) = Some(inner.clone());
        Arc::new(HealthRecordingLlmProvider { inner, health })
    }

    /// Mark the embedding role as configured and wrap it with passive recording.
    #[must_use]
    pub fn wrap_embedder(
        &self,
        inner: Arc<dyn Embedder>,
        provider: impl Into<String>,
        model: impl Into<String>,
        dim: u32,
    ) -> Arc<dyn Embedder> {
        let health = self.embedding.clone();
        health.configure(provider.into(), model.into(), Some(dim), None);
        Arc::new(HealthRecordingEmbedder { inner, health })
    }
}

/// Wire-format provider-health status.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderHealthStatus {
    /// The role has no provider configured.
    #[default]
    Disabled,
    /// The role is configured, but no provider call has happened in this process.
    Unknown,
    /// The last provider call succeeded.
    Ok,
    /// The last provider call failed.
    Error,
}

impl ProviderHealthStatus {
    /// Canonical kebab-case wire string.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Unknown => "unknown",
            Self::Ok => "ok",
            Self::Error => "error",
        }
    }
}

/// Wire-format health snapshot for all provider roles.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderHealthSnapshot {
    /// LLM provider role.
    pub llm: ProviderRoleHealthSnapshot,
    /// Ordered LLM fallback-chain candidate state (`llm_fallbacks`), in
    /// declaration order (index 0 is the primary provider). Empty for a
    /// single configured provider or no provider at all.
    /// `#[serde(default)]` so a response from a server built before this
    /// field existed still deserializes for a newer CLI.
    #[serde(default)]
    pub llm_candidates: Vec<CandidateHealth>,
    /// Embedding provider role.
    pub embedding: ProviderRoleHealthSnapshot,
}

/// One candidate's passive health state in an ordered LLM fallback chain.
/// See `docs/llm-provider-fallback.md`. Every entry here is by construction
/// a configured chain member; redaction is total — only labels, timestamps,
/// an HTTP status, and a short error class are recorded, never a response
/// body or credential.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateHealth {
    /// Candidate provider label.
    pub provider: String,
    /// Candidate model id.
    pub model: String,
    /// Whether this candidate answered the most recently completed chain
    /// call.
    pub last_selected: bool,
    /// Timestamp of the last call this candidate answered successfully.
    pub last_success_at: Option<Timestamp>,
    /// Timestamp of the last call this candidate failed.
    pub last_error_at: Option<Timestamp>,
    /// HTTP status captured from the last error, when available.
    pub last_error_status: Option<u16>,
    /// Redacted error class captured from the last error
    /// ([`LlmError::class`]), never a response body or secret.
    pub last_error_class: Option<String>,
    /// When this candidate's circuit reopens, if a transient failure has it
    /// currently open. `None` once the cooldown has elapsed, even if no
    /// later call has explicitly closed it.
    pub circuit_open_until: Option<Timestamp>,
}

/// Wire-format health snapshot for one provider role.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderRoleHealthSnapshot {
    /// Current role status.
    pub status: ProviderHealthStatus,
    /// Configured provider label, when the role is enabled.
    pub provider: Option<String>,
    /// Configured model, when the role is enabled.
    pub model: Option<String>,
    /// Configured embedding dimensionality. Only set for the embedding role.
    pub dim: Option<u32>,
    /// Timestamp of the last provider call, successful or failed.
    pub last_call_at: Option<Timestamp>,
    /// Timestamp of the last successful provider call.
    pub last_success_at: Option<Timestamp>,
    /// Timestamp of the last failed provider call.
    pub last_error_at: Option<Timestamp>,
    /// HTTP status captured from the last error, when available.
    pub last_error_status: Option<u16>,
    /// Truncated message captured from the last error, when available.
    pub last_error_message: Option<String>,
    /// Manual retry command hint. Set only for LLM provider errors.
    pub retry_hint: Option<String>,
}

impl Default for ProviderRoleHealthSnapshot {
    fn default() -> Self {
        Self {
            status: ProviderHealthStatus::Disabled,
            provider: None,
            model: None,
            dim: None,
            last_call_at: None,
            last_success_at: None,
            last_error_at: None,
            last_error_status: None,
            last_error_message: None,
            retry_hint: None,
        }
    }
}

#[derive(Clone)]
struct ProviderRoleHealth {
    state: Arc<Mutex<ProviderRoleState>>,
}

impl Default for ProviderRoleHealth {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(ProviderRoleState::disabled())),
        }
    }
}

impl ProviderRoleHealth {
    fn configure(
        &self,
        provider: String,
        model: String,
        dim: Option<u32>,
        retry_hint: Option<String>,
    ) {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) =
            ProviderRoleState::configured(provider, model, dim, retry_hint);
    }

    fn snapshot(&self) -> ProviderRoleHealthSnapshot {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .snapshot()
    }

    fn record_result<T>(&self, result: &LlmResult<T>) {
        match result {
            Ok(_) => self.record_success(),
            Err(err) => self.record_error(err),
        }
    }

    fn record_success(&self) {
        let now = Timestamp::now();
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.status = ProviderHealthStatus::Ok;
        state.last_call_at = Some(now);
        state.last_success_at = Some(now);
        state.last_error_at = None;
        state.last_error_status = None;
        state.last_error_message = None;
    }

    fn record_error(&self, err: &LlmError) {
        let now = Timestamp::now();
        let (status, message) = error_status_and_message(err);
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.status = ProviderHealthStatus::Error;
        state.last_call_at = Some(now);
        state.last_error_at = Some(now);
        state.last_error_status = status;
        state.last_error_message = Some(message);
    }
}

#[derive(Clone)]
struct ProviderRoleState {
    status: ProviderHealthStatus,
    provider: Option<String>,
    model: Option<String>,
    dim: Option<u32>,
    last_call_at: Option<Timestamp>,
    last_success_at: Option<Timestamp>,
    last_error_at: Option<Timestamp>,
    last_error_status: Option<u16>,
    last_error_message: Option<String>,
    retry_hint: Option<String>,
}

impl ProviderRoleState {
    fn disabled() -> Self {
        Self {
            status: ProviderHealthStatus::Disabled,
            provider: None,
            model: None,
            dim: None,
            last_call_at: None,
            last_success_at: None,
            last_error_at: None,
            last_error_status: None,
            last_error_message: None,
            retry_hint: None,
        }
    }

    fn configured(
        provider: String,
        model: String,
        dim: Option<u32>,
        retry_hint: Option<String>,
    ) -> Self {
        Self {
            status: ProviderHealthStatus::Unknown,
            provider: Some(provider),
            model: Some(model),
            dim,
            last_call_at: None,
            last_success_at: None,
            last_error_at: None,
            last_error_status: None,
            last_error_message: None,
            retry_hint,
        }
    }

    fn snapshot(&self) -> ProviderRoleHealthSnapshot {
        ProviderRoleHealthSnapshot {
            status: self.status,
            provider: self.provider.clone(),
            model: self.model.clone(),
            dim: self.dim,
            last_call_at: self.last_call_at,
            last_success_at: self.last_success_at,
            last_error_at: self.last_error_at,
            last_error_status: self.last_error_status,
            last_error_message: self.last_error_message.clone(),
            retry_hint: if self.status == ProviderHealthStatus::Error {
                self.retry_hint.clone()
            } else {
                None
            },
        }
    }
}

struct HealthRecordingLlmProvider {
    inner: Arc<dyn LlmProvider>,
    health: ProviderRoleHealth,
}

#[async_trait]
impl LlmProvider for HealthRecordingLlmProvider {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    async fn complete(&self, request: ChatRequest) -> LlmResult<ChatResponse> {
        let result = self.inner.complete(request).await;
        self.health.record_result(&result);
        result
    }

    /// Forward the caller's operation id to the inner provider instead of
    /// falling through to the trait default, which would drop it and make
    /// the inner admission layer mint a fresh one — an external retry of the
    /// same logical operation (the consolidator's fast retry) must keep the
    /// id that correlated the first attempt on the wire. One observation per
    /// return, exactly like the id-less path.
    async fn complete_with_operation_id(
        &self,
        request: ChatRequest,
        operation_id: LlmOperationId,
    ) -> LlmResult<ChatResponse> {
        let result = self
            .inner
            .complete_with_operation_id(request, operation_id)
            .await;
        self.health.record_result(&result);
        result
    }

    async fn complete_structured_raw(
        &self,
        request: ChatRequest,
        schema: serde_json::Value,
    ) -> LlmResult<serde_json::Value> {
        let result = self.inner.complete_structured_raw(request, schema).await;
        self.health.record_result(&result);
        result
    }

    /// Same id-forwarding contract as [`Self::complete_with_operation_id`]
    /// for the structured path, which is the one the consolidator's retry
    /// loop actually drives.
    async fn complete_structured_raw_with_operation_id(
        &self,
        request: ChatRequest,
        schema: serde_json::Value,
        operation_id: LlmOperationId,
    ) -> LlmResult<serde_json::Value> {
        let result = self
            .inner
            .complete_structured_raw_with_operation_id(request, schema, operation_id)
            .await;
        self.health.record_result(&result);
        result
    }

    fn candidate_health(&self) -> Vec<CandidateHealth> {
        self.inner.candidate_health()
    }
}

struct HealthRecordingEmbedder {
    inner: Arc<dyn Embedder>,
    health: ProviderRoleHealth,
}

#[async_trait]
impl Embedder for HealthRecordingEmbedder {
    fn provider(&self) -> &'static str {
        self.inner.provider()
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    fn model_identity(&self) -> String {
        // Not the default (`self.model().to_string()`): every configured
        // embedder is wrapped in this type before being handed to the rest
        // of the server (`serve.rs`'s `wrap_embedder`), so without this
        // override every caller of `model_identity` — the refuse-on-mismatch
        // check, backfill, retrieval, cleanup — would silently see the
        // wire model name instead of the inner embedder's actual document-
        // prefix-aware identity.
        self.inner.model_identity()
    }

    fn dim(&self) -> u32 {
        self.inner.dim()
    }

    async fn embed(&self, text: &str) -> LlmResult<Vec<f32>> {
        let result = self.inner.embed(text).await;
        self.health.record_result(&result);
        result
    }

    async fn embed_document(&self, text: &str) -> LlmResult<Vec<f32>> {
        let result = self.inner.embed_document(text).await;
        self.health.record_result(&result);
        result
    }

    async fn embed_query(&self, text: &str) -> LlmResult<Vec<f32>> {
        let result = self.inner.embed_query(text).await;
        self.health.record_result(&result);
        result
    }
}

fn error_status_and_message(err: &LlmError) -> (Option<u16>, String) {
    match err {
        LlmError::Http(e) => (
            e.status().map(|status| status.as_u16()),
            truncate_error_message(&e.to_string()),
        ),
        LlmError::Provider { status, body } => (Some(*status), truncate_error_message(body)),
        LlmError::RateLimited { .. }
        | LlmError::Capacity { .. }
        | LlmError::AmbiguousRetryExhausted { .. } => (err.http_status(), err.class().to_string()),
        _ => (None, truncate_error_message(&err.to_string())),
    }
}

fn truncate_error_message(message: &str) -> String {
    let mut chars = message.chars();
    let truncated: String = chars.by_ref().take(MAX_ERROR_MESSAGE_CHARS).collect();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    struct FakeLlm {
        fail: bool,
    }

    #[async_trait]
    impl LlmProvider for FakeLlm {
        fn name(&self) -> &'static str {
            "fake"
        }

        fn model(&self) -> &str {
            "fake-model"
        }

        async fn complete(&self, _request: ChatRequest) -> LlmResult<ChatResponse> {
            if self.fail {
                return Err(LlmError::Provider {
                    status: 401,
                    body: "bad token".to_string(),
                });
            }
            Ok(ChatResponse {
                text: "pong".to_string(),
                usage: None,
                model: "fake-model".to_string(),
            })
        }

        async fn complete_structured_raw(
            &self,
            _request: ChatRequest,
            _schema: serde_json::Value,
        ) -> LlmResult<serde_json::Value> {
            self.complete(ChatRequest::user_prompt("ping"))
                .await
                .map(|_| serde_json::json!({ "ok": true }))
        }
    }

    struct TaskAwareEmbedder;

    /// Captures the operation ids the inner provider actually receives and
    /// counts id-less invocations, so the wrapper tests can assert the id
    /// survives the recording layer and exactly one inner call (one health
    /// observation) happens per external return.
    struct IdCapturingLlm {
        chat_ids: Mutex<Vec<LlmOperationId>>,
        structured_ids: Mutex<Vec<LlmOperationId>>,
        /// Any id-less invocation means the wrapper lost the operation
        /// identity in the trait-default fallback.
        id_less_calls: AtomicUsize,
        /// When set, the first structured call fails transiently (sentinel
        /// body only — never real request material).
        fail_first_structured: AtomicBool,
    }

    impl std::default::Default for IdCapturingLlm {
        fn default() -> Self {
            Self {
                chat_ids: Mutex::new(Vec::new()),
                structured_ids: Mutex::new(Vec::new()),
                id_less_calls: AtomicUsize::new(0),
                fail_first_structured: AtomicBool::new(false),
            }
        }
    }

    impl IdCapturingLlm {
        fn failing_first_structured() -> Self {
            Self {
                fail_first_structured: AtomicBool::new(true),
                ..Self::default()
            }
        }
    }

    #[async_trait]
    impl LlmProvider for IdCapturingLlm {
        fn name(&self) -> &'static str {
            "id-capturing"
        }

        fn model(&self) -> &str {
            "fake-model"
        }

        async fn complete(&self, _request: ChatRequest) -> LlmResult<ChatResponse> {
            self.id_less_calls.fetch_add(1, Ordering::SeqCst);
            Ok(ChatResponse {
                text: "pong".to_string(),
                usage: None,
                model: "fake-model".to_string(),
            })
        }

        async fn complete_with_operation_id(
            &self,
            _request: ChatRequest,
            operation_id: LlmOperationId,
        ) -> LlmResult<ChatResponse> {
            self.chat_ids.lock().unwrap().push(operation_id);
            Ok(ChatResponse {
                text: "pong".to_string(),
                usage: None,
                model: "fake-model".to_string(),
            })
        }

        async fn complete_structured_raw(
            &self,
            _request: ChatRequest,
            _schema: serde_json::Value,
        ) -> LlmResult<serde_json::Value> {
            self.id_less_calls.fetch_add(1, Ordering::SeqCst);
            Ok(serde_json::json!({ "ok": true }))
        }

        async fn complete_structured_raw_with_operation_id(
            &self,
            _request: ChatRequest,
            _schema: serde_json::Value,
            operation_id: LlmOperationId,
        ) -> LlmResult<serde_json::Value> {
            self.structured_ids.lock().unwrap().push(operation_id);
            if self.fail_first_structured.swap(false, Ordering::SeqCst) {
                return Err(LlmError::Provider {
                    status: 503,
                    body: "sentinel transient failure".to_string(),
                });
            }
            Ok(serde_json::json!({ "ok": true }))
        }
    }

    /// Acceptance 1: the wrapper forwards the caller's id — not a minted
    /// one — to the inner provider on both the chat and the structured path.
    /// Removing either override makes this fail: the trait default would
    /// reach the id-less inner methods and nothing would be captured.
    #[tokio::test]
    async fn llm_wrapper_preserves_the_operation_id_for_chat_and_structured() {
        let health = ProviderHealth::default();
        let fake = Arc::new(IdCapturingLlm::default());
        let llm = health.wrap_llm_provider(
            Arc::clone(&fake) as Arc<dyn LlmProvider>,
            "fake",
            "fake-model",
            None,
        );

        let chat_id = LlmOperationId::new();
        let chat = llm
            .complete_with_operation_id(ChatRequest::user_prompt("ping"), chat_id)
            .await
            .unwrap();
        assert_eq!(chat.text, "pong");

        let structured_id = LlmOperationId::new();
        assert_ne!(
            chat_id, structured_id,
            "distinct operations are distinct ids"
        );
        let value = llm
            .complete_structured_raw_with_operation_id(
                ChatRequest::user_prompt("ping"),
                serde_json::json!({ "type": "object" }),
                structured_id,
            )
            .await
            .unwrap();
        assert_eq!(value, serde_json::json!({ "ok": true }));

        let chat_ids = fake.chat_ids.lock().unwrap().clone();
        let structured_ids = fake.structured_ids.lock().unwrap().clone();
        assert_eq!(
            chat_ids,
            vec![chat_id],
            "chat path forwards the caller's id"
        );
        assert_eq!(
            structured_ids,
            vec![structured_id],
            "structured path forwards the caller's id"
        );
        assert_eq!(
            fake.id_less_calls.load(Ordering::SeqCst),
            0,
            "the wrapper must not fall back to the id-less inner methods"
        );

        // One observation per return: two returns, no error, last state Ok.
        let after = health.snapshot().llm;
        assert_eq!(after.status, ProviderHealthStatus::Ok);
        assert!(after.last_call_at.is_some());
        assert!(after.last_success_at.is_some());
        assert!(after.last_error_at.is_none());
        assert!(after.last_error_message.is_none());
    }

    /// Acceptance 2: two external structured calls with the same id — first
    /// failing transiently, second succeeding, as the consolidator's fast
    /// retry does — both reach the inner provider with that one id, and the
    /// health role observes each return (error, then success).
    #[tokio::test]
    async fn llm_wrapper_keeps_one_id_across_retries_and_records_each_return() {
        let health = ProviderHealth::default();
        let fake = Arc::new(IdCapturingLlm::failing_first_structured());
        let llm = health.wrap_llm_provider(
            Arc::clone(&fake) as Arc<dyn LlmProvider>,
            "fake",
            "fake-model",
            None,
        );

        let operation_id = LlmOperationId::new();
        let schema = serde_json::json!({ "type": "object" });

        let first = llm
            .complete_structured_raw_with_operation_id(
                ChatRequest::user_prompt("ping"),
                schema.clone(),
                operation_id,
            )
            .await;
        let LlmError::Provider { status, .. } = first.as_ref().unwrap_err() else {
            panic!("the first attempt must fail: {first:?}");
        };
        assert_eq!(*status, 503);
        assert!(
            first.as_ref().unwrap_err().is_transient(),
            "the failure is fast-retryable, like the consolidator's retry budget"
        );
        let after_failure = health.snapshot().llm;
        assert_eq!(after_failure.status, ProviderHealthStatus::Error);
        assert_eq!(after_failure.last_error_status, Some(503));
        assert!(after_failure.last_call_at.is_some());

        let second = llm
            .complete_structured_raw_with_operation_id(
                ChatRequest::user_prompt("ping"),
                schema,
                operation_id,
            )
            .await
            .unwrap();
        assert_eq!(second, serde_json::json!({ "ok": true }));
        let after_success = health.snapshot().llm;
        assert_eq!(after_success.status, ProviderHealthStatus::Ok);
        assert!(after_success.last_success_at.is_some());
        assert!(
            after_success.last_error_at.is_none(),
            "the success observation supersedes the error"
        );
        assert!(after_success.last_error_message.is_none());

        let ids = fake.structured_ids.lock().unwrap().clone();
        assert_eq!(
            ids.len(),
            2,
            "one inner call per external return, no internal re-dispatch"
        );
        let (id_1, id_2) = (ids[0], ids[1]);
        assert_eq!(
            id_1, id_2,
            "both attempts of one operation carry the same id"
        );
        assert_eq!(
            id_1, operation_id,
            "the inner provider sees the caller's id"
        );
        assert_eq!(fake.id_less_calls.load(Ordering::SeqCst), 0);
    }

    #[async_trait]
    impl Embedder for TaskAwareEmbedder {
        fn provider(&self) -> &'static str {
            "google"
        }

        fn model(&self) -> &str {
            "gemini-embedding-001"
        }

        fn dim(&self) -> u32 {
            2
        }

        async fn embed(&self, _text: &str) -> LlmResult<Vec<f32>> {
            Ok(vec![0.0, 0.0])
        }

        async fn embed_document(&self, _text: &str) -> LlmResult<Vec<f32>> {
            Ok(vec![1.0, 0.0])
        }

        async fn embed_query(&self, _text: &str) -> LlmResult<Vec<f32>> {
            Ok(vec![0.0, 1.0])
        }
    }

    #[tokio::test]
    async fn llm_wrapper_records_unknown_success_and_error() {
        let health = ProviderHealth::default();
        let retry_hint =
            "ai-memory llm-test --provider anthropic-oauth --model claude-sonnet-4-6 --prompt ping"
                .to_string();
        let llm = health.wrap_llm_provider(
            Arc::new(FakeLlm { fail: false }),
            "anthropic-oauth",
            "claude-sonnet-4-6",
            Some(retry_hint.clone()),
        );

        let before = health.snapshot().llm;
        assert_eq!(before.status, ProviderHealthStatus::Unknown);
        assert_eq!(before.provider.as_deref(), Some("anthropic-oauth"));
        assert_eq!(before.model.as_deref(), Some("claude-sonnet-4-6"));
        assert!(before.retry_hint.is_none());

        llm.complete(ChatRequest::user_prompt("ping"))
            .await
            .unwrap();
        let after_success = health.snapshot().llm;
        assert_eq!(after_success.status, ProviderHealthStatus::Ok);
        assert!(after_success.last_call_at.is_some());
        assert!(after_success.last_success_at.is_some());
        assert!(after_success.last_error_message.is_none());

        let llm = health.wrap_llm_provider(
            Arc::new(FakeLlm { fail: true }),
            "anthropic-oauth",
            "claude-sonnet-4-6",
            Some(retry_hint.clone()),
        );
        let err = llm.complete(ChatRequest::user_prompt("ping")).await;
        assert!(err.is_err());
        let after_error = health.snapshot().llm;
        assert_eq!(after_error.status, ProviderHealthStatus::Error);
        assert_eq!(after_error.last_error_status, Some(401));
        assert_eq!(after_error.last_error_message.as_deref(), Some("bad token"));
        assert_eq!(after_error.retry_hint.as_deref(), Some(retry_hint.as_str()));
    }

    #[tokio::test]
    async fn embedder_wrapper_records_and_preserves_task_specific_methods() {
        let health = ProviderHealth::default();
        let embedder = health.wrap_embedder(
            Arc::new(TaskAwareEmbedder),
            "google",
            "gemini-embedding-001",
            2,
        );

        let before = health.snapshot().embedding;
        assert_eq!(before.status, ProviderHealthStatus::Unknown);
        assert_eq!(before.dim, Some(2));

        assert_eq!(
            embedder.embed_document("doc").await.unwrap(),
            vec![1.0, 0.0]
        );
        assert_eq!(embedder.embed_query("query").await.unwrap(), vec![0.0, 1.0]);

        let after = health.snapshot().embedding;
        assert_eq!(after.status, ProviderHealthStatus::Ok);
        assert!(after.last_call_at.is_some());
    }

    /// Every configured embedder is wrapped in `HealthRecordingEmbedder`
    /// before reaching the rest of the server (`serve.rs`'s
    /// `wrap_embedder`), so a caller of `model_identity` — the
    /// refuse-on-mismatch check, backfill, retrieval, cleanup — only ever
    /// sees the wrapper, never the inner embedder directly. Without the
    /// wrapper's own `model_identity` override, the default trait impl
    /// (`self.model().to_string()`) would silently discard a document
    /// prefix's identity fingerprint. `TaskAwareEmbedder` doesn't
    /// distinguish this (it never overrides `model_identity` either, so a
    /// missing wrapper override would coincidentally still pass against
    /// it); a real `OpenAiCompatEmbedder` with a document prefix set does.
    #[test]
    fn embedder_wrapper_forwards_the_inner_model_identity_override() {
        let inner =
            crate::OpenAiCompatEmbedder::new("http://localhost:9/v1", None, "nomic-embed-text", 8)
                .expect("embedder builds")
                .with_prefixes("query: ", "passage: ");
        let inner_identity = inner.model_identity();
        assert_ne!(
            inner_identity,
            inner.model(),
            "the fixture must actually have a distinct identity, or this test proves nothing"
        );

        let health = ProviderHealth::default();
        let wrapped = health.wrap_embedder(Arc::new(inner), "openai-compat", "nomic-embed-text", 8);
        assert_eq!(wrapped.model_identity(), inner_identity);
    }

    #[test]
    fn provider_health_status_wire_labels() {
        for (status, label) in [
            (ProviderHealthStatus::Disabled, "disabled"),
            (ProviderHealthStatus::Unknown, "unknown"),
            (ProviderHealthStatus::Ok, "ok"),
            (ProviderHealthStatus::Error, "error"),
        ] {
            assert_eq!(status.as_str(), label);
            assert_eq!(
                serde_json::to_value(status).unwrap(),
                serde_json::json!(label)
            );
            assert_eq!(
                serde_json::from_value::<ProviderHealthStatus>(serde_json::json!(label)).unwrap(),
                status
            );
        }
    }
}

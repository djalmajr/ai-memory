//! One admission point for every chat request made by a server process.

use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use tokenizers::Tokenizer;
use tokio::sync::{Semaphore, SemaphorePermit};
use tokio::time::Instant;
use tracing::{info, warn};

use crate::error::{LlmError, LlmResult};
use crate::provider::LlmProvider;
use crate::types::{ChatRequest, ChatResponse, LlmOperationId};

const MAX_WAITERS: usize = 32;
const AMBIGUOUS_COOLDOWN: Duration = Duration::from_secs(60);
const MAX_RETRY_AFTER_SECS: u64 = 300;
// The configured tokenizer counts each content field and schema separately.
// Reserve room for ChatML roles, separators, and provider-added framing.
const CHAT_OVERHEAD_TOKENS: usize = 1_024;
const MESSAGE_OVERHEAD_TOKENS: usize = 16;

/// Process-local admission and tokenized input limit for one LLM chain.
///
/// Construct once around the *entire* primary/fallback chain, then clone the
/// resulting `Arc<dyn LlmProvider>` into every job. This keeps fallback
/// attempts under the same permit as the original call.
pub struct AdmittedLlmProvider {
    inner: Arc<dyn LlmProvider>,
    in_flight: Semaphore,
    waiting: AtomicUsize,
    max_waiters: usize,
    tokenizer: Option<Tokenizer>,
    max_input_tokens: Option<usize>,
    ambiguous_cooldown: Duration,
    jitter_max_secs: u8,
}

struct Waiting<'a>(&'a AtomicUsize);

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

impl AdmittedLlmProvider {
    /// Build a server-wide LLM admission boundary. A token cap requires the
    /// exact model tokenizer JSON; startup fails closed when it is missing.
    ///
    /// # Errors
    /// Returns a configuration error if the requested tokenizer cannot load.
    pub fn new(
        inner: Arc<dyn LlmProvider>,
        max_input_tokens: Option<usize>,
        tokenizer_path: Option<&Path>,
    ) -> LlmResult<Self> {
        let tokenizer =
            match (max_input_tokens, tokenizer_path) {
                (Some(0), _) => {
                    return Err(LlmError::NotConfigured(
                        "llm_max_input_tokens must be greater than zero".into(),
                    ));
                }
                (Some(_), Some(path)) => Some(Tokenizer::from_file(path).map_err(|_| {
                    LlmError::NotConfigured("cannot load llm_tokenizer_path".into())
                })?),
                (Some(_), None) => {
                    return Err(LlmError::NotConfigured(
                        "llm_tokenizer_path is required with llm_max_input_tokens".into(),
                    ));
                }
                (None, _) => None,
            };
        Ok(Self {
            inner,
            in_flight: Semaphore::new(1),
            waiting: AtomicUsize::new(0),
            max_waiters: MAX_WAITERS,
            tokenizer,
            max_input_tokens,
            ambiguous_cooldown: AMBIGUOUS_COOLDOWN,
            jitter_max_secs: 16,
        })
    }

    fn count_input_tokens(
        &self,
        request: &ChatRequest,
        schema: Option<&serde_json::Value>,
    ) -> LlmResult<usize> {
        let Some(tokenizer) = &self.tokenizer else {
            return Ok(0);
        };
        let mut total = CHAT_OVERHEAD_TOKENS
            .saturating_add(MESSAGE_OVERHEAD_TOKENS.saturating_mul(request.messages.len() + 1));
        if let Some(system) = &request.system {
            total = total.saturating_add(
                tokenizer
                    .encode(system.as_str(), false)
                    .map_err(|_| LlmError::NotConfigured("LLM input tokenization failed".into()))?
                    .len(),
            );
        }
        for message in &request.messages {
            total = total.saturating_add(
                tokenizer
                    .encode(message.content.as_str(), false)
                    .map_err(|_| LlmError::NotConfigured("LLM input tokenization failed".into()))?
                    .len(),
            );
        }
        if let Some(schema) = schema {
            total = total.saturating_add(
                tokenizer
                    .encode(schema.to_string(), false)
                    .map_err(|_| LlmError::NotConfigured("LLM schema tokenization failed".into()))?
                    .len(),
            );
        }
        Ok(total)
    }

    fn enforce_input_limit(
        &self,
        request: &ChatRequest,
        schema: Option<&serde_json::Value>,
    ) -> LlmResult<()> {
        if let Some(max) = self.max_input_tokens {
            let tokens = self.count_input_tokens(request, schema)?;
            if tokens > max {
                return Err(LlmError::InputLimit { tokens, max });
            }
            tracing::debug!(tokens, max, "LLM input token check passed");
        }
        Ok(())
    }

    async fn acquire(&self) -> LlmResult<SemaphorePermit<'_>> {
        if let Ok(permit) = self.in_flight.try_acquire() {
            return Ok(permit);
        }
        self.waiting
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < self.max_waiters).then_some(count + 1)
            })
            .map_err(|count| {
                warn!(queued = count, "LLM admission queue full");
                LlmError::AdmissionFull
            })?;
        let waiting = Waiting(&self.waiting);
        let started = Instant::now();
        let permit = self
            .in_flight
            .acquire()
            .await
            .map_err(|_| LlmError::AdmissionFull)?;
        drop(waiting);
        info!(
            wait_ms = started.elapsed().as_millis(),
            "LLM admission wait completed"
        );
        Ok(permit)
    }

    async fn execute<T, F, Fut>(
        &self,
        request: &ChatRequest,
        schema: Option<&serde_json::Value>,
        mut action: F,
    ) -> LlmResult<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = LlmResult<T>>,
    {
        self.enforce_input_limit(request, schema)?;
        // Keep this permit across an ambiguous delivery's quarantine and
        // sole replay. The provider may still be decoding the first request;
        // admitting another job during that window would defeat the process-
        // local serialization guarantee.
        let permit = self.acquire().await?;
        let result = match action().await {
            Err(error) if error.is_ambiguous_delivery() => {
                let jitter = u64::from(uuid::Uuid::new_v4().as_bytes()[0] % self.jitter_max_secs);
                let delay = self.ambiguous_cooldown + Duration::from_secs(jitter);
                warn!(delay_secs = delay.as_secs(), class = error.class(), status = ?error.http_status(), "LLM delivery uncertain; delaying one replay");
                // The first failure may have reached the model. Even a safe
                // capacity response on the replay must not escape to an
                // outer retry loop, which would turn this into a third call.
                tokio::time::sleep(delay).await;
                match action().await {
                    Ok(value) => Ok(value),
                    Err(last) => Err(LlmError::AmbiguousRetryExhausted {
                        class: last.class(),
                        status: last.http_status(),
                    }),
                }
            }
            Err(error) if error.retry_after_secs().is_some() => {
                let delay = error.retry_after_secs().unwrap_or_default();
                if delay > MAX_RETRY_AFTER_SECS {
                    return Err(error);
                }
                info!(delay_secs = delay, "LLM 429 requested delayed retry");
                // A rate-limited request is safe to replay after the server's
                // requested delay. If that one replay is ambiguous, make it
                // terminal so callers cannot immediately duplicate it again.
                tokio::time::sleep(Duration::from_secs(delay)).await;
                match action().await {
                    Ok(value) => Ok(value),
                    Err(last) if last.is_ambiguous_delivery() => {
                        Err(LlmError::AmbiguousRetryExhausted {
                            class: last.class(),
                            status: last.http_status(),
                        })
                    }
                    Err(last) => Err(last),
                }
            }
            other => other,
        };
        drop(permit);
        result
    }
}

#[async_trait]
impl LlmProvider for AdmittedLlmProvider {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    async fn complete(&self, request: ChatRequest) -> LlmResult<ChatResponse> {
        self.complete_with_operation_id(request, LlmOperationId::new())
            .await
    }

    async fn complete_with_operation_id(
        &self,
        request: ChatRequest,
        operation_id: LlmOperationId,
    ) -> LlmResult<ChatResponse> {
        self.execute(&request, None, || {
            self.inner
                .complete_with_operation_id(request.clone(), operation_id)
        })
        .await
    }

    async fn complete_structured_raw(
        &self,
        request: ChatRequest,
        schema: serde_json::Value,
    ) -> LlmResult<serde_json::Value> {
        self.complete_structured_raw_with_operation_id(request, schema, LlmOperationId::new())
            .await
    }

    async fn complete_structured_raw_with_operation_id(
        &self,
        request: ChatRequest,
        schema: serde_json::Value,
        operation_id: LlmOperationId,
    ) -> LlmResult<serde_json::Value> {
        self.execute(&request, Some(&schema), || {
            self.inner.complete_structured_raw_with_operation_id(
                request.clone(),
                schema.clone(),
                operation_id,
            )
        })
        .await
    }

    fn candidate_health(&self) -> Vec<crate::health::CandidateHealth> {
        self.inner.candidate_health()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokio::sync::Notify;
    use wiremock::matchers::path;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::types::{ChatMessage, Usage};

    struct ScriptedProvider {
        calls: AtomicUsize,
        replies: Mutex<VecDeque<LlmResult<ChatResponse>>>,
    }

    impl ScriptedProvider {
        fn new(replies: Vec<LlmResult<ChatResponse>>) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                replies: Mutex::new(replies.into()),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for ScriptedProvider {
        fn name(&self) -> &'static str {
            "scripted"
        }

        fn model(&self) -> &str {
            "test"
        }

        async fn complete(&self, _request: ChatRequest) -> LlmResult<ChatResponse> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            self.replies
                .lock()
                .unwrap()
                .pop_front()
                .expect("one scripted reply per call")
        }

        async fn complete_structured_raw(
            &self,
            request: ChatRequest,
            _schema: serde_json::Value,
        ) -> LlmResult<serde_json::Value> {
            self.complete(request).await?;
            Ok(serde_json::json!({"ok": true}))
        }
    }

    fn response() -> LlmResult<ChatResponse> {
        Ok(ChatResponse {
            text: "ok".into(),
            usage: Some(Usage {
                input_tokens: 1,
                output_tokens: 1,
            }),
            model: "test".into(),
        })
    }

    struct BlockingProvider {
        active: AtomicUsize,
        max_active: AtomicUsize,
        started: Notify,
        release: Semaphore,
    }

    #[async_trait]
    impl LlmProvider for BlockingProvider {
        fn name(&self) -> &'static str {
            "blocking"
        }

        fn model(&self) -> &str {
            "test"
        }

        async fn complete(&self, _request: ChatRequest) -> LlmResult<ChatResponse> {
            let active = self.active.fetch_add(1, Ordering::AcqRel) + 1;
            self.max_active.fetch_max(active, Ordering::AcqRel);
            self.started.notify_one();
            let permit = self.release.acquire().await.expect("test semaphore open");
            permit.forget();
            self.active.fetch_sub(1, Ordering::AcqRel);
            response()
        }

        async fn complete_structured_raw(
            &self,
            request: ChatRequest,
            _schema: serde_json::Value,
        ) -> LlmResult<serde_json::Value> {
            self.complete(request).await?;
            Ok(serde_json::json!({"ok": true}))
        }
    }

    // Mutation captured: removing the permit around provider awaits lets two
    // different LLM job types enter the provider concurrently.
    #[tokio::test]
    async fn plain_and_structured_calls_share_one_permit() {
        let inner = Arc::new(BlockingProvider {
            active: AtomicUsize::new(0),
            max_active: AtomicUsize::new(0),
            started: Notify::new(),
            release: Semaphore::new(0),
        });
        let guard = Arc::new(AdmittedLlmProvider::new(inner.clone(), None, None).unwrap());
        let first = {
            let guard = guard.clone();
            tokio::spawn(async move { guard.complete(ChatRequest::user_prompt("first")).await })
        };
        inner.started.notified().await;
        let second = {
            let guard = guard.clone();
            tokio::spawn(async move {
                guard
                    .complete_structured_raw(
                        ChatRequest::user_prompt("second"),
                        serde_json::json!({"type":"object"}),
                    )
                    .await
            })
        };
        tokio::task::yield_now().await;
        assert_eq!(inner.active.load(Ordering::Acquire), 1);
        inner.release.add_permits(2);
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
        assert_eq!(inner.max_active.load(Ordering::Acquire), 1);
    }

    // Mutation captured: accepting an unbounded waiter permits an arbitrary
    // number of jobs to pile up behind one slow GPU request.
    #[tokio::test]
    async fn queue_rejects_the_request_after_its_limit() {
        let inner = Arc::new(BlockingProvider {
            active: AtomicUsize::new(0),
            max_active: AtomicUsize::new(0),
            started: Notify::new(),
            release: Semaphore::new(0),
        });
        let mut admitted = AdmittedLlmProvider::new(inner.clone(), None, None).unwrap();
        admitted.max_waiters = 1;
        let guard = Arc::new(admitted);
        let first = {
            let guard = guard.clone();
            tokio::spawn(async move { guard.complete(ChatRequest::user_prompt("first")).await })
        };
        inner.started.notified().await;
        let second = {
            let guard = guard.clone();
            tokio::spawn(async move { guard.complete(ChatRequest::user_prompt("second")).await })
        };
        while guard.waiting.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        let third = guard.complete(ChatRequest::user_prompt("third")).await;
        assert!(matches!(third, Err(LlmError::AdmissionFull)));
        inner.release.add_permits(2);
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
    }

    // Mutation captured: moving the cap after the provider call would spend
    // tokens on an oversized request before returning InputLimit.
    #[tokio::test]
    async fn tokenized_input_limit_rejects_before_provider_call() {
        let inner = Arc::new(ScriptedProvider::new(vec![response()]));
        let mut guard = AdmittedLlmProvider::new(inner.clone(), None, None).unwrap();
        let model = WordLevel::builder()
            .vocab(
                [("a".to_string(), 0), ("[UNK]".to_string(), 1)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        guard.tokenizer = Some(tokenizer);
        guard.max_input_tokens = Some(1_090);
        let request = ChatRequest {
            system: None,
            messages: vec![ChatMessage::user(vec!["a"; 40].join(" "))],
            max_tokens: 1,
            temperature: None,
        };
        assert!(matches!(
            guard.complete(request).await,
            Err(LlmError::InputLimit { .. })
        ));
        assert_eq!(inner.calls.load(Ordering::Acquire), 0);
    }

    // Mutation captured: silently accepting a missing tokenizer would leave
    // the configured token ceiling unenforced after server startup.
    #[test]
    fn capped_provider_fails_startup_without_a_readable_tokenizer() {
        let inner = Arc::new(ScriptedProvider::new(vec![]));
        let tmp = tempfile::tempdir().unwrap();
        assert!(matches!(
            AdmittedLlmProvider::new(inner.clone(), Some(16_000), None),
            Err(LlmError::NotConfigured(_))
        ));
        assert!(matches!(
            AdmittedLlmProvider::new(inner, Some(16_000), Some(&tmp.path().join("missing.json"))),
            Err(LlmError::NotConfigured(_))
        ));
    }

    // Mutation captured: retrying a 502 immediately fails the elapsed-time
    // assertion; retrying it twice fails the call count.
    #[tokio::test]
    async fn ambiguous_response_gets_one_delayed_replay() {
        let inner = Arc::new(ScriptedProvider::new(vec![
            Err(LlmError::Provider {
                status: 502,
                body: "client gave up".into(),
            }),
            response(),
        ]));
        let mut guard = AdmittedLlmProvider::new(inner.clone(), None, None).unwrap();
        guard.ambiguous_cooldown = Duration::from_millis(20);
        guard.jitter_max_secs = 1;
        let started = Instant::now();
        guard.complete(ChatRequest::user_prompt("x")).await.unwrap();
        assert!(started.elapsed() >= Duration::from_millis(20));
        assert_eq!(inner.calls.load(Ordering::Acquire), 2);
    }

    // Mutation captured: dropping the permit before the cooldown lets a
    // queued request enter while the ambiguous delivery is quarantined.
    #[tokio::test]
    async fn ambiguous_cooldown_keeps_queued_calls_out() {
        let inner = Arc::new(ScriptedProvider::new(vec![
            Err(LlmError::Provider {
                status: 502,
                body: "client gave up".into(),
            }),
            response(),
            response(),
        ]));
        let mut admitted = AdmittedLlmProvider::new(inner.clone(), None, None).unwrap();
        admitted.ambiguous_cooldown = Duration::from_millis(40);
        admitted.jitter_max_secs = 1;
        let guard = Arc::new(admitted);

        let first = {
            let guard = guard.clone();
            tokio::spawn(async move { guard.complete(ChatRequest::user_prompt("first")).await })
        };
        while inner.calls.load(Ordering::Acquire) < 1 {
            tokio::task::yield_now().await;
        }

        let second = {
            let guard = guard.clone();
            tokio::spawn(async move { guard.complete(ChatRequest::user_prompt("second")).await })
        };
        let queued = tokio::time::timeout(Duration::from_millis(10), async {
            while guard.waiting.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await;
        assert!(queued.is_ok(), "the second call should remain queued");
        assert_eq!(inner.calls.load(Ordering::Acquire), 1);

        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
        assert_eq!(inner.calls.load(Ordering::Acquire), 3);
    }

    // Mutation captured: classifying reqwest timeouts as fast failures would
    // replay a request that the HTTP server had already accepted.
    #[tokio::test]
    async fn real_http_timeout_uses_the_ambiguous_delivery_path() {
        let server = MockServer::start().await;
        Mock::given(path("/slow"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(200)))
            .mount(&server)
            .await;
        let timeout = reqwest::Client::new()
            .get(format!("{}/slow", server.uri()))
            .timeout(Duration::from_millis(10))
            .send()
            .await
            .unwrap_err();
        assert!(timeout.is_timeout());
        let inner = Arc::new(ScriptedProvider::new(vec![
            Err(LlmError::Http(timeout)),
            response(),
        ]));
        let mut guard = AdmittedLlmProvider::new(inner.clone(), None, None).unwrap();
        guard.ambiguous_cooldown = Duration::from_millis(20);
        guard.jitter_max_secs = 1;
        let started = Instant::now();
        guard.complete(ChatRequest::user_prompt("x")).await.unwrap();
        assert!(started.elapsed() >= Duration::from_millis(20));
        assert_eq!(inner.calls.load(Ordering::Acquire), 2);
    }

    // Mutation captured: a second ambiguous failure must not escape as a
    // transient 5xx that the caller then retries a third time.
    #[tokio::test]
    async fn failed_delayed_replay_is_terminal_to_outer_retry_loops() {
        let inner = Arc::new(ScriptedProvider::new(vec![
            Err(LlmError::Provider {
                status: 499,
                body: "cancelled".into(),
            }),
            Err(LlmError::Provider {
                status: 503,
                body: "busy".into(),
            }),
        ]));
        let mut guard = AdmittedLlmProvider::new(inner.clone(), None, None).unwrap();
        guard.ambiguous_cooldown = Duration::from_millis(1);
        guard.jitter_max_secs = 1;
        let error = guard
            .complete(ChatRequest::user_prompt("x"))
            .await
            .unwrap_err();
        assert!(matches!(error, LlmError::AmbiguousRetryExhausted { .. }));
        assert!(!error.is_fast_retryable());
        assert_eq!(inner.calls.load(Ordering::Acquire), 2);
    }

    // Mutation captured: dropping Retry-After prevents the one provider
    // requested replay even when the delay is explicitly zero.
    #[tokio::test]
    async fn rate_limit_replays_only_with_retry_after() {
        let inner = Arc::new(ScriptedProvider::new(vec![
            Err(LlmError::RateLimited {
                body: "rate limited".into(),
                retry_after_secs: Some(0),
            }),
            response(),
        ]));
        let guard = AdmittedLlmProvider::new(inner.clone(), None, None).unwrap();
        guard.complete(ChatRequest::user_prompt("x")).await.unwrap();
        assert_eq!(inner.calls.load(Ordering::Acquire), 2);
    }

    // Mutation captured: returning the second 502 directly lets an outer
    // fast-retry loop issue a third request after the delayed replay.
    #[tokio::test]
    async fn rate_limit_replay_of_ambiguous_failure_is_terminal() {
        let inner = Arc::new(ScriptedProvider::new(vec![
            Err(LlmError::RateLimited {
                body: "rate limited".into(),
                retry_after_secs: Some(0),
            }),
            Err(LlmError::Provider {
                status: 502,
                body: "gateway failure".into(),
            }),
        ]));
        let guard = AdmittedLlmProvider::new(inner.clone(), None, None).unwrap();
        let error = guard
            .complete(ChatRequest::user_prompt("x"))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            LlmError::AmbiguousRetryExhausted {
                class: "provider",
                status: Some(502),
            }
        ));
        assert!(!error.is_fast_retryable());
        assert_eq!(inner.calls.load(Ordering::Acquire), 2);
    }
}

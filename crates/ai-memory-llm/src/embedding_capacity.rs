//! Capacity isolation for in-process embedding inference.
//!
//! Local model inference is CPU-heavy and already parallelizes internally.
//! Letting background page writes fill every inference slot can more than
//! double interactive query latency. [`EmbeddingCapacity`] bounds total work
//! while reserving one slot for query embeddings.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{Embedder, LlmError, LlmResult};

/// Embedder wrapper that bounds inference and reserves foreground capacity.
///
/// `capacity` must be at least two. Document and generic embeddings may use
/// at most `capacity - 1` slots, so an interactive query can enter without
/// waiting for background indexing work to finish.
pub struct EmbeddingCapacity {
    inner: Arc<dyn Embedder>,
    total: Arc<Semaphore>,
    documents: Arc<Semaphore>,
}

impl EmbeddingCapacity {
    /// Wrap `inner` with a bounded scheduler.
    ///
    /// # Errors
    /// Returns [`LlmError::NotConfigured`] when fewer than two slots are
    /// requested because that cannot reserve foreground capacity.
    pub fn new(inner: Arc<dyn Embedder>, capacity: usize) -> LlmResult<Self> {
        if capacity < 2 {
            return Err(LlmError::NotConfigured(
                "embedding capacity must be at least 2".into(),
            ));
        }
        Ok(Self {
            inner,
            total: Arc::new(Semaphore::new(capacity)),
            documents: Arc::new(Semaphore::new(capacity - 1)),
        })
    }

    async fn acquire_document_capacity(
        &self,
    ) -> LlmResult<(OwnedSemaphorePermit, OwnedSemaphorePermit)> {
        let document = self
            .documents
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| capacity_closed())?;
        let total = self
            .total
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| capacity_closed())?;
        Ok((document, total))
    }

    async fn finish_task(
        task: tokio::task::JoinHandle<LlmResult<Vec<f32>>>,
    ) -> LlmResult<Vec<f32>> {
        task.await
            .map_err(|e| LlmError::UnexpectedShape(format!("embedding task: {e}")))?
    }
}

fn capacity_closed() -> LlmError {
    LlmError::UnexpectedShape("embedding capacity scheduler closed".into())
}

#[async_trait]
impl Embedder for EmbeddingCapacity {
    fn provider(&self) -> &'static str {
        self.inner.provider()
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    fn model_identity(&self) -> String {
        self.inner.model_identity()
    }

    fn dim(&self) -> u32 {
        self.inner.dim()
    }

    async fn embed(&self, text: &str) -> LlmResult<Vec<f32>> {
        let capacity = self.acquire_document_capacity().await?;
        let inner = Arc::clone(&self.inner);
        let text = text.to_owned();
        Self::finish_task(tokio::spawn(async move {
            let _capacity = capacity;
            inner.embed(&text).await
        }))
        .await
    }

    async fn embed_document(&self, text: &str) -> LlmResult<Vec<f32>> {
        let capacity = self.acquire_document_capacity().await?;
        let inner = Arc::clone(&self.inner);
        let text = text.to_owned();
        Self::finish_task(tokio::spawn(async move {
            let _capacity = capacity;
            inner.embed_document(&text).await
        }))
        .await
    }

    async fn embed_query(&self, text: &str) -> LlmResult<Vec<f32>> {
        let capacity = self
            .total
            .clone()
            .try_acquire_owned()
            .map_err(|_| LlmError::CapacityBusy("embedding"))?;
        let inner = Arc::clone(&self.inner);
        let text = text.to_owned();
        Self::finish_task(tokio::spawn(async move {
            let _capacity = capacity;
            inner.embed_query(&text).await
        }))
        .await
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::sync::mpsc;

    use super::*;

    struct BlockingEmbedder {
        entered: mpsc::UnboundedSender<&'static str>,
        releases: Arc<Semaphore>,
    }

    impl BlockingEmbedder {
        async fn block(&self, operation: &'static str) -> LlmResult<Vec<f32>> {
            self.entered.send(operation).unwrap();
            self.releases
                .acquire()
                .await
                .map_err(|_| capacity_closed())?
                .forget();
            Ok(vec![1.0])
        }
    }

    #[async_trait]
    impl Embedder for BlockingEmbedder {
        fn provider(&self) -> &'static str {
            "blocking"
        }

        fn model(&self) -> &str {
            "test"
        }

        fn model_identity(&self) -> String {
            "document-identity".into()
        }

        fn dim(&self) -> u32 {
            1
        }

        async fn embed(&self, _text: &str) -> LlmResult<Vec<f32>> {
            self.block("generic").await
        }

        async fn embed_document(&self, _text: &str) -> LlmResult<Vec<f32>> {
            self.block("document").await
        }

        async fn embed_query(&self, _text: &str) -> LlmResult<Vec<f32>> {
            self.block("query").await
        }
    }

    #[test]
    fn requires_a_foreground_reservation() {
        let (entered, _entered_rx) = mpsc::unbounded_channel();
        let inner = Arc::new(BlockingEmbedder {
            entered,
            releases: Arc::new(Semaphore::new(0)),
        });
        assert!(EmbeddingCapacity::new(inner, 1).is_err());
    }

    #[tokio::test]
    async fn query_enters_while_second_document_waits() {
        let (entered, mut entered_rx) = mpsc::unbounded_channel();
        let releases = Arc::new(Semaphore::new(0));
        let inner = Arc::new(BlockingEmbedder {
            entered,
            releases: Arc::clone(&releases),
        });
        let stage: Arc<dyn Embedder> = Arc::new(EmbeddingCapacity::new(inner, 2).unwrap());

        let first_document = {
            let stage = Arc::clone(&stage);
            tokio::spawn(async move { stage.embed_document("one").await })
        };
        assert_eq!(entered_rx.recv().await, Some("document"));

        let second_document = {
            let stage = Arc::clone(&stage);
            tokio::spawn(async move { stage.embed_document("two").await })
        };
        let query = {
            let stage = Arc::clone(&stage);
            tokio::spawn(async move { stage.embed_query("find it").await })
        };

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), entered_rx.recv())
                .await
                .unwrap(),
            Some("query"),
            "the reserved query slot must remain available"
        );
        assert!(
            crate::embed_search_query(stage.as_ref(), "another query")
                .await
                .is_none(),
            "a saturated query must degrade immediately to non-vector search"
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(25), entered_rx.recv())
                .await
                .is_err(),
            "the second document must wait for the document slot"
        );

        releases.add_permits(1);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), entered_rx.recv())
                .await
                .unwrap(),
            Some("document")
        );
        releases.add_permits(2);
        first_document.await.unwrap().unwrap();
        second_document.await.unwrap().unwrap();
        query.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn cancelled_caller_does_not_release_capacity_before_inference_finishes() {
        let (entered, mut entered_rx) = mpsc::unbounded_channel();
        let releases = Arc::new(Semaphore::new(0));
        let stage: Arc<dyn Embedder> = Arc::new(
            EmbeddingCapacity::new(
                Arc::new(BlockingEmbedder {
                    entered,
                    releases: Arc::clone(&releases),
                }),
                2,
            )
            .unwrap(),
        );

        let cancelled = {
            let stage = Arc::clone(&stage);
            tokio::spawn(async move { stage.embed_document("cancel me").await })
        };
        assert_eq!(entered_rx.recv().await, Some("document"));
        cancelled.abort();
        assert!(cancelled.await.unwrap_err().is_cancelled());

        let next = {
            let stage = Arc::clone(&stage);
            tokio::spawn(async move { stage.embed_document("wait for capacity").await })
        };
        assert!(
            tokio::time::timeout(Duration::from_millis(25), entered_rx.recv())
                .await
                .is_err(),
            "cancelled caller must not free a still-running inference slot"
        );

        releases.add_permits(1);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), entered_rx.recv())
                .await
                .unwrap(),
            Some("document")
        );
        releases.add_permits(1);
        next.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn generic_embedding_keeps_inner_method_semantics() {
        let (entered, mut entered_rx) = mpsc::unbounded_channel();
        let releases = Arc::new(Semaphore::new(0));
        let stage = Arc::new(
            EmbeddingCapacity::new(
                Arc::new(BlockingEmbedder {
                    entered,
                    releases: Arc::clone(&releases),
                }),
                2,
            )
            .unwrap(),
        );
        let task = tokio::spawn(async move { stage.embed("generic input").await });
        assert_eq!(entered_rx.recv().await, Some("generic"));
        releases.add_permits(1);
        task.await.unwrap().unwrap();
    }

    #[test]
    fn delegates_embedding_identity() {
        let (entered, _entered_rx) = mpsc::unbounded_channel();
        let stage = EmbeddingCapacity::new(
            Arc::new(BlockingEmbedder {
                entered,
                releases: Arc::new(Semaphore::new(0)),
            }),
            2,
        )
        .unwrap();
        assert_eq!(stage.provider(), "blocking");
        assert_eq!(stage.model(), "test");
        assert_eq!(stage.model_identity(), "document-identity");
        assert_eq!(stage.dim(), 1);
    }
}

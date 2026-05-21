//! Read-only connection pool and query helpers.
//!
//! WAL mode lets us have unlimited concurrent readers alongside the single
//! writer, so the pool is mostly about bounding file-descriptor usage and
//! avoiding `Connection::open` overhead on hot paths. Pool eviction is a
//! soft cap: a connection that comes back when the pool is already full
//! is simply dropped.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ai_memory_core::{PageId, PagePath};
use parking_lot::Mutex;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde::Serialize;

use crate::error::{StoreError, StoreResult};

/// One hit returned by [`ReaderPool::search_pages`].
#[derive(Debug, Clone, Serialize)]
pub struct PageHit {
    /// Stable identifier for this page version.
    pub id: PageId,
    /// Relative path within the wiki tree.
    pub path: PagePath,
    /// Page title.
    pub title: String,
    /// FTS5 snippet of the body around the matched terms (HTML-marked).
    pub snippet: String,
    /// FTS5 rank score (lower is better — closer to query terms).
    pub rank: f64,
}

/// Aggregate counts surfaced by [`ReaderPool::status_counts`] and consumed
/// by `ai-memory status`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct StatusCounts {
    /// Pages with `is_latest = 1`.
    pub pages_latest: u64,
    /// All page versions including superseded ones.
    pub pages_all: u64,
    /// Total sessions ever recorded.
    pub sessions: u64,
    /// Total observations across all sessions.
    pub observations: u64,
}

/// Cheap, cloneable read-only connection pool handle.
#[derive(Clone)]
pub struct ReaderPool {
    inner: Arc<Inner>,
}

struct Inner {
    db_path: PathBuf,
    pool: Mutex<Vec<Connection>>,
    soft_cap: usize,
}

impl ReaderPool {
    /// Initialise the pool. Connections are opened lazily on first use.
    ///
    /// # Errors
    /// Currently infallible, but reserved so we can pre-open connections
    /// in a later milestone.
    pub fn new(db_path: &Path, soft_cap: usize) -> StoreResult<Self> {
        Ok(Self {
            inner: Arc::new(Inner {
                db_path: db_path.to_path_buf(),
                pool: Mutex::new(Vec::with_capacity(soft_cap.max(1))),
                soft_cap: soft_cap.max(1),
            }),
        })
    }

    /// Run a synchronous closure against a pooled read-only connection.
    ///
    /// The closure runs on the tokio blocking pool so it never starves the
    /// async runtime. If the pool is empty we open a fresh connection;
    /// on return we keep it only when the pool is below its soft cap.
    ///
    /// # Errors
    /// Returns [`StoreError::PoolPanic`] if the blocking task panics; any
    /// error returned by the closure is propagated unchanged.
    pub async fn with_conn<F, T>(&self, f: F) -> StoreResult<T>
    where
        F: FnOnce(&Connection) -> StoreResult<T> + Send + 'static,
        T: Send + 'static,
    {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let conn = checkout(&inner)?;
            let result = f(&conn);
            checkin(&inner, conn);
            result
        })
        .await
        .map_err(|e| StoreError::PoolPanic(e.to_string()))?
    }

    /// Run a full-text search against the FTS5 index and return the top
    /// matches, limited to `is_latest = 1` rows.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn search_pages(&self, query: String, limit: usize) -> StoreResult<Vec<PageHit>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT pages.id, pages.path, pages.title, \
                        snippet(pages_fts, 1, '<mark>', '</mark>', '…', 24) AS snip, \
                        pages_fts.rank \
                 FROM pages_fts \
                 JOIN pages ON pages.rowid = pages_fts.rowid \
                 WHERE pages_fts MATCH ?1 AND pages.is_latest = 1 \
                 ORDER BY pages_fts.rank \
                 LIMIT ?2",
            )?;
            #[allow(clippy::cast_possible_wrap)]
            let rows = stmt.query_map(params![query, limit as i64], |row| {
                let id_bytes: Vec<u8> = row.get(0)?;
                let path: String = row.get(1)?;
                let title: String = row.get(2)?;
                let snippet: String = row.get(3)?;
                let rank: f64 = row.get(4)?;
                Ok((id_bytes, path, title, snippet, rank))
            })?;

            let mut hits = Vec::new();
            for row in rows {
                let (id_bytes, path, title, snippet, rank) = row?;
                hits.push(PageHit {
                    id: PageId::from_slice(&id_bytes)?,
                    path: PagePath::new(path)?,
                    title,
                    snippet,
                    rank,
                });
            }
            Ok(hits)
        })
        .await
    }

    /// Return the N most-recently-updated `is_latest = 1` pages.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn recent_pages(&self, limit: usize) -> StoreResult<Vec<PageHit>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, path, title, \
                        substr(body, 1, 240) AS snip, \
                        CAST(updated_at AS REAL) AS rank \
                 FROM pages \
                 WHERE is_latest = 1 \
                 ORDER BY updated_at DESC \
                 LIMIT ?1",
            )?;
            #[allow(clippy::cast_possible_wrap)]
            let rows = stmt.query_map(params![limit as i64], |row| {
                let id_bytes: Vec<u8> = row.get(0)?;
                let path: String = row.get(1)?;
                let title: String = row.get(2)?;
                let snippet: String = row.get(3)?;
                let rank: f64 = row.get(4)?;
                Ok((id_bytes, path, title, snippet, rank))
            })?;
            let mut hits = Vec::new();
            for row in rows {
                let (id_bytes, path, title, snippet, rank) = row?;
                hits.push(PageHit {
                    id: PageId::from_slice(&id_bytes)?,
                    path: PagePath::new(path)?,
                    title,
                    snippet,
                    rank,
                });
            }
            Ok(hits)
        })
        .await
    }

    /// Snapshot the database to `dest_path` using SQLite's online backup
    /// API. The source DB stays writable for the duration of the copy.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn snapshot_to(&self, dest_path: PathBuf) -> StoreResult<()> {
        self.with_conn(move |conn| {
            conn.backup(rusqlite::DatabaseName::Main, &dest_path, None)
                .map_err(StoreError::from)
        })
        .await
    }

    /// Return aggregate counts for the `status` view.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn status_counts(&self) -> StoreResult<StatusCounts> {
        self.with_conn(|conn| {
            let pages_latest: u64 = count(conn, "SELECT COUNT(*) FROM pages WHERE is_latest = 1")?;
            let pages_all: u64 = count(conn, "SELECT COUNT(*) FROM pages")?;
            let sessions: u64 = count(conn, "SELECT COUNT(*) FROM sessions")?;
            let observations: u64 = count(conn, "SELECT COUNT(*) FROM observations")?;
            Ok(StatusCounts {
                pages_latest,
                pages_all,
                sessions,
                observations,
            })
        })
        .await
    }
}

fn count(conn: &Connection, sql: &str) -> StoreResult<u64> {
    let n: Option<i64> = conn.query_row(sql, [], |row| row.get(0)).optional()?;
    Ok(u64::try_from(n.unwrap_or(0)).unwrap_or(0))
}

fn checkout(inner: &Inner) -> StoreResult<Connection> {
    if let Some(conn) = inner.pool.lock().pop() {
        return Ok(conn);
    }
    open_read_only(&inner.db_path)
}

fn checkin(inner: &Inner, conn: Connection) {
    let mut pool = inner.pool.lock();
    if pool.len() < inner.soft_cap {
        pool.push(conn);
    }
}

fn open_read_only(path: &Path) -> StoreResult<Connection> {
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY
        | OpenFlags::SQLITE_OPEN_URI
        | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = Connection::open_with_flags(path, flags)?;
    conn.pragma_update(None, "busy_timeout", 5_000)?;
    Ok(conn)
}

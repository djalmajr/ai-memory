//! Store-level error type.

use thiserror::Error;

use ai_memory_core::MemoryError;

/// Result alias used throughout the store crate.
pub type StoreResult<T> = Result<T, StoreError>;

/// Errors raised by the store layer.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StoreError {
    /// Underlying SQLite error.
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// Migration runner failed.
    #[error("migration: {0}")]
    Migration(#[from] refinery::Error),

    /// I/O failed (e.g. opening the DB file).
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// JSON serialisation failure (frontmatter).
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),

    /// Writer actor has shut down.
    #[error("writer actor is no longer running")]
    WriterClosed,

    /// A `spawn_blocking` task panicked or was cancelled.
    #[error("reader pool task did not complete: {0}")]
    PoolPanic(String),

    /// Re-export of [`MemoryError`] for cross-crate propagation.
    #[error(transparent)]
    Memory(#[from] MemoryError),

    /// A project rename was rejected because the destination name is already
    /// in use by another project in the same workspace.
    #[error("project name '{0}' is already taken in this workspace")]
    ProjectNameTaken(String),

    /// The supplied project name failed validation (empty, slash, etc.).
    #[error("invalid project name: {0}")]
    InvalidProjectName(String),

    /// A UNIQUE constraint was violated by an insert (e.g. duplicate
    /// `users.username` / `users.email`). The string carries a
    /// human-readable explanation the CLI / admin endpoint surfaces
    /// verbatim.
    #[error("duplicate: {0}")]
    Duplicate(String),

    /// An OS primitive failed (e.g. the CSPRNG read inside
    /// [`crate::users::generate_token`]). Carries the OS error
    /// description.
    #[error("os error: {0}")]
    Os(String),
}

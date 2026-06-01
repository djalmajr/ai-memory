//! Admin HTTP routes — state-touching operations invoked by the CLI
//! over plain HTTP (not MCP). Currently exposes:
//!
//! - `POST /admin/backup`         — snapshot db + wiki into a gzip tarball (binary response).
//! - `POST /admin/bootstrap`      — ingest a pre-collected source bundle
//!   into seed wiki pages via the configured LLM provider.
//! - `GET  /admin/status`         — lifetime counts + server data-dir info.
//! - `GET  /admin/search?q=`      — FTS5 hits against the wiki index.
//! - `POST /admin/reorg`          — retro-fit sessions to per-cwd projects.
//! - `POST /admin/lint`           — run the M8 lint pass.
//! - `POST /admin/forget-sweep`   — run the M8 retention sweep.
//! - `POST /admin/embed`          — backfill embeddings for latest pages.
//! - `POST /admin/commit`         — stage + commit the wiki tree via git.
//! - `POST /admin/purge-project`  — delete a project and all its data.
//! - `POST /admin/rename-project` — rename a project (column-only; no files move).
//! - `POST /admin/write-page`     — write or update a wiki page atomically.
//! - `GET  /admin/read-page`      — fetch the full body of a single wiki page by path.
//!
//! The CLI is responsible for filesystem access (collecting sources from
//! the project repo, rendering output for humans); the server is
//! responsible for all state reads/writes against the wiki + SQLite.

use std::sync::Arc;

use std::io::Seek;
use std::path::PathBuf;

use ai_memory_consolidate::{
    Bootstrap, BootstrapConfig, BootstrapOutcome, BootstrapSource, SourceCounts,
    prune_sources_to_budget, run_lint, run_sweep,
};
use ai_memory_core::{
    DEFAULT_PROJECT_NAME, DEFAULT_WORKSPACE_NAME, PagePath, ProjectId, SessionId, Tier, WorkspaceId,
};
use ai_memory_llm::{Embedder, LlmProvider, ProviderHealth, ProviderHealthSnapshot};
use ai_memory_store::{
    DecayParams, EmbeddingWrite, ReaderPool, StoreError, WriterHandle, f32_vec_to_bytes,
};
use ai_memory_wiki::{AdmissionContext, AdmissionOp, Wiki, WikiError, WritePageRequest};
use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use flate2::Compression;
use flate2::write::GzEncoder;
use serde::{Deserialize, Serialize};
use tokio_util::io::ReaderStream;
use tracing::{info, warn};

const EMBEDDING_WRITE_BATCH: usize = 100;

/// Shared state for the admin router.
#[derive(Clone)]
pub struct AdminState {
    /// Writer actor handle — used to get-or-create workspace/project.
    pub writer: WriterHandle,
    /// Reader pool — used by the idempotency check inside Bootstrap.
    pub reader: ReaderPool,
    /// Wiki handle — pages are written here.
    pub wiki: Wiki,
    /// Optional LLM provider. When `None`, bootstrap returns 503.
    pub llm: Option<Arc<dyn LlmProvider>>,
    /// Optional embedder. When `None`, `/admin/embed` returns 503.
    pub embedder: Option<Arc<dyn Embedder>>,
    /// Passive process-scoped health recorder for configured providers.
    pub provider_health: ProviderHealth,
    /// Retention-decay parameters forwarded from server config.
    pub decay_params: DecayParams,
    /// Server's resolved data directory (e.g. `/data` in the docker
    /// image). Surfaced via `/admin/status` so the CLI can report
    /// "where the wiki + db actually live".
    pub data_dir: PathBuf,
    /// Resolved SQLite path inside `data_dir`. Same purpose as above.
    pub db_path: PathBuf,
    /// Server's bind address — informational, surfaced in /admin/status.
    pub bind: String,
    /// Serialises concurrent bootstrap requests. Bootstrap fans out
    /// into an LLM call + a multi-page wiki write + a git commit;
    /// running two in parallel would race the `commit_all` git ops and
    /// stack LLM cost unnecessarily. The mutex is held for the entire
    /// handler so a second caller waits its turn (request stays open
    /// until the lock is acquired, then proceeds normally).
    pub bootstrap_lock: Arc<tokio::sync::Mutex<()>>,
    /// Per-server token pepper, used by the user-management endpoints
    /// (`POST /admin/users`, `…/rotate-token`) to hash freshly-issued
    /// tokens before they land in `users.token_hash`. `None` when the
    /// operator hasn't set `[auth].token_pepper` in config (single-user
    /// installs that predate v0.8); user-management endpoints then
    /// return 503 `multi-user not enabled`.
    pub token_pepper: Option<ai_memory_store::TokenPepper>,
}

/// JSON request body for `POST /admin/bootstrap`.
#[derive(Deserialize)]
struct BootstrapRequest {
    /// Workspace name (auto-created if it doesn't exist).
    workspace: String,
    /// Project name (auto-created if it doesn't exist).
    project: String,
    /// Sources pre-collected on the client side.
    sources: Vec<BootstrapSource>,
    /// Original collection size before client-side prune (if any).
    #[serde(default)]
    sources_collected: Option<usize>,
    /// Maximum input tokens for LLM call.
    #[serde(default = "default_max_input_tokens")]
    max_input_tokens: usize,
    /// Per-LLM-call input cap; larger bundles are split into chunks.
    #[serde(default = "default_chunk_input_tokens")]
    chunk_input_tokens: usize,
    /// Skip the LLM call and page writes — returns a dry-run outcome.
    #[serde(default)]
    dry_run: bool,
    /// Allow re-bootstrap when `wiki/bootstrap.md` already exists.
    #[serde(default)]
    force: bool,
}

fn default_max_input_tokens() -> usize {
    50_000
}

fn default_chunk_input_tokens() -> usize {
    ai_memory_consolidate::DEFAULT_CHUNK_INPUT_TOKENS
}

/// Build the admin axum [`Router`]. Mounts:
/// - `POST /admin/backup`
/// - `POST /admin/bootstrap`
/// - `GET  /admin/status`
/// - `GET  /admin/search`
/// - `GET  /admin/read-page`
/// - `POST /admin/reorg`
/// - `POST /admin/lint`
/// - `POST /admin/forget-sweep`
/// - `POST /admin/embed`
/// - `POST /admin/commit`
/// - `POST /admin/purge-project`
/// - `POST /admin/rename-project`
/// - `POST /admin/write-page`
pub fn admin_router(state: AdminState) -> Router {
    Router::new()
        .route("/admin/backup", post(handle_backup))
        .route("/admin/bootstrap", post(handle_bootstrap))
        .route("/admin/status", get(handle_status))
        .route("/admin/search", get(handle_search))
        .route("/admin/read-page", get(handle_read_page))
        .route("/admin/reorg", post(handle_reorg))
        .route("/admin/lint", post(handle_lint))
        .route("/admin/forget-sweep", post(handle_forget_sweep))
        .route("/admin/embed", post(handle_embed))
        .route("/admin/commit", post(handle_commit))
        .route("/admin/purge-project", post(handle_purge_project))
        .route("/admin/rename-project", post(handle_rename_project))
        .route("/admin/write-page", post(handle_write_page))
        .route(
            "/admin/users",
            get(handle_list_users).post(handle_create_user),
        )
        .route("/admin/users/{username}/expire", post(handle_expire_user))
        .route("/admin/users/{username}/revive", post(handle_revive_user))
        .route(
            "/admin/users/{username}/rotate-token",
            post(handle_rotate_user_token),
        )
        .with_state(Arc::new(state))
}

// ---------------------------------------------------------------------
// backup
// ---------------------------------------------------------------------

/// Handler for `POST /admin/backup`.
///
/// Snapshots the live SQLite DB via the online backup API, then
/// tar-gzips `wiki/`, the snapshot, and `config.toml` (if present)
/// into a tempfile. The response streams the file instead of buffering
/// the full archive in memory.
async fn handle_backup(State(state): State<Arc<AdminState>>) -> Response {
    match build_backup_tarball_file(&state).await {
        Ok(file) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/gzip")
            .header(
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"backup.tar.gz\"",
            )
            .body(Body::from_stream(ReaderStream::new(file)))
            .unwrap_or_else(|_| {
                Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Body::empty())
                    .unwrap()
            }),
        Err(e) => {
            warn!(error = %e, "backup failed");
            let body = serde_json::to_vec(&serde_json::json!({ "error": e.to_string() }))
                .unwrap_or_default();
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body))
                .unwrap_or_else(|_| {
                    Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Body::empty())
                        .unwrap()
                })
        }
    }
}

async fn build_backup_tarball_file(state: &AdminState) -> anyhow::Result<tokio::fs::File> {
    let staging = tempfile::tempdir()?;
    let snapshot_path = staging.path().join("memory.sqlite");
    info!(snapshot = %snapshot_path.display(), "snapshotting SQLite for backup");
    state
        .reader
        .snapshot_to(snapshot_path.clone())
        .await
        .map_err(|e| anyhow::anyhow!("sqlite snapshot: {e}"))?;

    let mut tar_file = tempfile::tempfile()?;
    {
        let encoder = GzEncoder::new(&mut tar_file, Compression::default());
        let mut tar = tar::Builder::new(encoder);
        tar.mode(tar::HeaderMode::Deterministic);

        let wiki_dir = state.data_dir.join("wiki");
        if wiki_dir.is_dir() {
            tar.append_dir_all("wiki", &wiki_dir)
                .map_err(|e| anyhow::anyhow!("archiving wiki/: {e}"))?;
        }

        tar.append_path_with_name(&snapshot_path, "db/memory.sqlite")
            .map_err(|e| anyhow::anyhow!("archiving db snapshot: {e}"))?;

        let cfg = state.data_dir.join("config.toml");
        if cfg.is_file() {
            tar.append_path_with_name(&cfg, "config.toml")
                .map_err(|e| anyhow::anyhow!("archiving config.toml: {e}"))?;
        }

        let encoder = tar.into_inner()?;
        encoder.finish()?;
    }
    tar_file.sync_data()?;
    tar_file.rewind()?;
    Ok(tokio::fs::File::from_std(tar_file))
}

// ---------------------------------------------------------------------
// status
// ---------------------------------------------------------------------

/// JSON response body for `GET /admin/status`. The CLI's `status`
/// subcommand renders this either as JSON (`--json`) or as a small
/// human-friendly text block.
#[derive(Debug, Serialize)]
pub struct StatusReport {
    /// Server binary version (Cargo package version).
    pub version: String,
    /// Absolute data directory the server uses (server-side path).
    pub data_dir: String,
    /// Bind address the HTTP transport is listening on.
    pub bind: String,
    /// Absolute SQLite path inside `data_dir`.
    pub db_path: String,
    /// Lifetime counts: pages_latest, pages_all, sessions, observations.
    pub counts: ai_memory_store::StatusCounts,
    /// Derived-index and retrieval-readiness diagnostics.
    pub derived: ai_memory_store::DerivedIndexStatus,
    /// Passive process-scoped provider health.
    pub providers: ProviderHealthSnapshot,
}

async fn handle_status(State(state): State<Arc<AdminState>>) -> impl IntoResponse {
    match state.reader.status_counts().await {
        Ok(counts) => match state.reader.derived_index_status().await {
            Ok(derived) => {
                let report = StatusReport {
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    data_dir: state.data_dir.display().to_string(),
                    bind: state.bind.clone(),
                    db_path: state.db_path.display().to_string(),
                    counts,
                    derived,
                    providers: state.provider_health.snapshot(),
                };
                (
                    StatusCode::OK,
                    Json(serde_json::to_value(&report).unwrap_or_else(|_| serde_json::json!({}))),
                )
            }
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            ),
        },
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        ),
    }
}

// ---------------------------------------------------------------------
// search
// ---------------------------------------------------------------------

/// Query string for `GET /admin/search?q=…&limit=…`.
#[derive(Debug, Deserialize)]
struct SearchQuery {
    /// FTS5 query expression.
    q: String,
    /// Workspace name to search within. When omitted with `project`, admin search is global.
    #[serde(default)]
    workspace: Option<String>,
    /// Project name to search within. When omitted with `workspace`, admin search is global.
    #[serde(default)]
    project: Option<String>,
    /// Max number of hits to return. Capped at 100 server-side.
    #[serde(default = "default_search_limit")]
    limit: usize,
}

fn default_search_limit() -> usize {
    10
}

async fn handle_search(
    State(state): State<Arc<AdminState>>,
    Query(query): Query<SearchQuery>,
) -> impl IntoResponse {
    let limit = query.limit.clamp(1, 100);
    let search_result = match (query.workspace.as_deref(), query.project.as_deref()) {
        (Some(workspace), Some(project)) => match resolve_ws_proj(&state, workspace, project).await
        {
            Ok((ws, proj)) => {
                state
                    .reader
                    .search_pages_for_project(ws, proj, query.q, limit)
                    .await
            }
            Err(e) => return e,
        },
        _ => state.reader.search_pages(query.q, limit).await,
    };
    match search_result {
        Ok(hits) => (
            StatusCode::OK,
            Json(serde_json::to_value(&hits).unwrap_or_else(|_| serde_json::json!([]))),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        ),
    }
}

// ---------------------------------------------------------------------
// read-page
// ---------------------------------------------------------------------

/// Query string for `GET /admin/read-page`.
///
/// Two modes:
/// - Path mode: `workspace` + `project` + `path` — direct lookup.
/// - Query mode: `workspace` + `project` + `q` — FTS5 search scoped to
///   the given project; fetches the top-ranking hit's full body.
#[derive(Debug, Deserialize)]
struct ReadPageQuery {
    /// Workspace name (required).
    workspace: String,
    /// Project name (required).
    project: String,
    /// Direct wiki path (e.g. `notes/foo.md`). Takes precedence over `q`.
    #[serde(default)]
    path: Option<String>,
    /// FTS5 query. Used when `path` is absent; fetches the top hit's full body.
    #[serde(default)]
    q: Option<String>,
}

/// Response body for `GET /admin/read-page`.
#[derive(Debug, Serialize)]
struct ReadPageResponse {
    path: String,
    workspace: String,
    project: String,
    title: Option<String>,
    body: String,
    frontmatter: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    served_from: Option<&'static str>,
}

async fn handle_read_page(
    State(state): State<Arc<AdminState>>,
    Query(query): Query<ReadPageQuery>,
) -> impl IntoResponse {
    let (ws, proj) = match resolve_ws_proj(&state, &query.workspace, &query.project).await {
        Ok(ids) => ids,
        Err(e) => return e,
    };

    // Resolve the page path: direct `path` takes precedence over `q`.
    let page_path = if let Some(raw) = query.path {
        match ai_memory_core::PagePath::new(raw) {
            Ok(p) => p,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": format!("invalid path: {e}") })),
                );
            }
        }
    } else if let Some(q) = query.q {
        let hits = match state
            .reader
            .search_pages_for_project(ws, proj, q.clone(), 1)
            .await
        {
            Ok(h) => h,
            Err(e) => return internal_err(e.to_string()),
        };
        match hits.into_iter().next() {
            Some(h) => h.path,
            None => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({ "error": format!("no pages found for query {q:?}") })),
                );
            }
        }
    } else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "provide `path` or `q`" })),
        );
    };

    match state.wiki.read_page(ws, proj, &page_path) {
        Ok(md) => {
            let title = md
                .frontmatter
                .get("title")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let resp = ReadPageResponse {
                path: page_path.to_string(),
                workspace: query.workspace,
                project: query.project,
                title,
                body: md.body,
                frontmatter: md.frontmatter,
                served_from: None,
            };
            (
                StatusCode::OK,
                Json(serde_json::to_value(&resp).unwrap_or_else(|_| serde_json::json!({}))),
            )
        }
        // Only a missing markdown file can fall back to the DB copy. Other disk
        // errors belong to the source-of-truth file and must be surfaced.
        Err(disk_err) if is_missing_wiki_file(&disk_err) => match state
            .reader
            .page_body_by_ids(ws, proj, page_path.as_str())
            .await
        {
            Ok(Some(stored)) => {
                let frontmatter: serde_json::Value = serde_json::from_str(&stored.frontmatter_json)
                    .unwrap_or(serde_json::Value::Null);
                let title = frontmatter
                    .get("title")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .or(Some(stored.title));
                let resp = ReadPageResponse {
                    path: page_path.to_string(),
                    workspace: query.workspace,
                    project: query.project,
                    title,
                    body: stored.body,
                    frontmatter,
                    served_from: Some("db-fallback"),
                };
                (
                    StatusCode::OK,
                    Json(serde_json::to_value(&resp).unwrap_or_else(|_| serde_json::json!({}))),
                )
            }
            Ok(None) => (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": disk_err.to_string() })),
            ),
            Err(e) => internal_err(e.to_string()),
        },
        Err(disk_err) => internal_err(disk_err.to_string()),
    }
}

// ---------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------

/// Build a 500 response carrying the given message.
fn internal_err(msg: impl Into<String>) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({ "error": msg.into() })),
    )
}

fn is_missing_wiki_file(err: &WikiError) -> bool {
    matches!(err, WikiError::Io(e) if e.kind() == std::io::ErrorKind::NotFound)
}

/// Resolve workspace + project IDs, creating them if absent. Returns
/// either the IDs or a ready-to-return error response.
async fn resolve_ws_proj(
    state: &AdminState,
    workspace: &str,
    project: &str,
) -> Result<(WorkspaceId, ProjectId), (StatusCode, Json<serde_json::Value>)> {
    let ws = state
        .writer
        .get_or_create_workspace(workspace.to_string())
        .await
        .map_err(|e| internal_err(format!("workspace: {e}")))?;
    let proj = state
        .writer
        .get_or_create_project(ws, project.to_string(), None)
        .await
        .map_err(|e| internal_err(format!("project: {e}")))?;
    Ok((ws, proj))
}

/// Look up workspace + project by name **without** auto-creating them.
/// Returns `(WorkspaceId, ProjectId)` on success, or a ready-to-return
/// 404/500 error response. Used by destructive handlers (purge, rename)
/// where auto-creation would silently succeed on a typo.
async fn lookup_ws_proj_no_create(
    state: &AdminState,
    workspace: &str,
    project: &str,
) -> Result<(WorkspaceId, ProjectId), (StatusCode, Json<serde_json::Value>)> {
    let ws_id = match state.reader.find_workspace(workspace.to_string()).await {
        Ok(Some(id)) => id,
        Ok(None) => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": format!("workspace '{workspace}' not found")
                })),
            ));
        }
        Err(e) => return Err(internal_err(e.to_string())),
    };
    let proj_id = match state.reader.find_project(ws_id, project.to_string()).await {
        Ok(Some(id)) => id,
        Ok(None) => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": format!("project '{project}' not found in workspace '{workspace}'")
                })),
            ));
        }
        Err(e) => return Err(internal_err(e.to_string())),
    };
    Ok((ws_id, proj_id))
}

// ---------------------------------------------------------------------
// bootstrap
// ---------------------------------------------------------------------

async fn handle_bootstrap(
    State(state): State<Arc<AdminState>>,
    Json(req): Json<BootstrapRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    // Dry-runs are READ-ONLY: no LLM call, no project creation, no
    // wiki write, no git commit. Early-return BEFORE any state-touching
    // call so a smoke-test (e.g. `--dry-run` from a tempdir) cannot
    // pollute the project list with throwaway names. Dry-runs can also
    // proceed in parallel with anything else — no mutex needed.
    if req.dry_run {
        return dry_run_outcome(
            req.sources,
            req.sources_collected,
            req.max_input_tokens,
            req.chunk_input_tokens,
        );
    }

    // Live runs from here on need LLM + workspace/project resolution.
    if state.llm.is_none() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "LLM provider not configured on server"
            })),
        ));
    }
    let (ws, proj) = resolve_ws_proj(&state, &req.workspace, &req.project).await?;

    // Serialise live bootstrap runs. Two parallel `process_sources`
    // calls would race the wiki's `commit_all` (libgit2 ops on the
    // same repo) and stack LLM cost unnecessarily. The wait is
    // operator-visible only in log lines; the request stays open
    // until the lock is acquired, then proceeds normally.
    if state.bootstrap_lock.try_lock().is_err() {
        info!(
            "another bootstrap is in progress; queueing — \
             this request waits for the active one to finish"
        );
    }
    let _bootstrap_guard = state.bootstrap_lock.lock().await;

    let llm = Arc::clone(
        state
            .llm
            .as_ref()
            .expect("llm is Some: checked above (non-dry-run without LLM returns 503)"),
    );

    let cfg = BootstrapConfig {
        // repo_path is unused by process_sources — the path field is
        // only consumed by collect_sources on the client side.
        repo_path: std::path::PathBuf::new(),
        workspace_id: ws,
        project_id: proj,
        max_input_tokens: req.max_input_tokens,
        chunk_input_tokens: req.chunk_input_tokens,
        sources_collected: req.sources_collected,
        // The individual include_* flags don't matter here: sources
        // are already collected; process_sources ignores them.
        include_git: true,
        include_readme: true,
        include_docs: true,
        include_code: true,
        since: None,
        dry_run: req.dry_run,
        force: req.force,
    };

    let bootstrap = Bootstrap {
        reader: state.reader.clone(),
        wiki: state.wiki.clone(),
        llm,
    };

    match bootstrap.process_sources(&cfg, req.sources).await {
        Ok(outcome) => Ok((
            StatusCode::OK,
            Json(serde_json::to_value(&outcome).unwrap_or_else(|_| serde_json::json!({}))),
        )),
        Err(e) => Err(bootstrap_error_response(e)),
    }
}

/// Build a dry-run [`BootstrapOutcome`] without an LLM by applying the
/// same budget-pruning logic that `Bootstrap::process_sources` would use.
/// Map a [`BootstrapError`] to the appropriate HTTP status code.
///
/// - `NoSources` / `AlreadyBootstrapped` → 422 (validation failures).
/// - `Llm` → 502 (upstream provider failure).
/// - Everything else → 500 (unexpected server-side error).
fn bootstrap_error_response(
    e: ai_memory_consolidate::BootstrapError,
) -> (StatusCode, Json<serde_json::Value>) {
    use ai_memory_consolidate::BootstrapError;
    let status = match &e {
        BootstrapError::NoSources | BootstrapError::AlreadyBootstrapped => {
            StatusCode::UNPROCESSABLE_ENTITY
        }
        BootstrapError::Llm(_) => StatusCode::BAD_GATEWAY,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, Json(serde_json::json!({ "error": e.to_string() })))
}

/// Build a dry-run [`BootstrapOutcome`] without an LLM by applying the
/// same budget-pruning logic that `Bootstrap::process_sources` would use.
fn dry_run_outcome(
    sources: Vec<BootstrapSource>,
    sources_collected: Option<usize>,
    max_input_tokens: usize,
    chunk_input_tokens: usize,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, Json<serde_json::Value>)> {
    use ai_memory_consolidate::BootstrapError;
    if sources.is_empty() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": BootstrapError::NoSources.to_string()
            })),
        ));
    }
    let incoming = sources.len();
    let (kept, _dropped, total) = prune_sources_to_budget(sources, max_input_tokens);
    if kept.is_empty() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": BootstrapError::NoSources.to_string()
            })),
        ));
    }
    let collected = sources_collected.unwrap_or(incoming);
    let sources_sent = kept.len();
    let sources_dropped = collected.saturating_sub(sources_sent);
    let counts = SourceCounts::from_sources(&kept);
    let chunk_budget =
        ai_memory_consolidate::effective_chunk_budget(chunk_input_tokens, max_input_tokens);
    let llm_chunks = ai_memory_consolidate::plan_bootstrap_chunks(kept.clone(), chunk_budget).len();
    let outcome = BootstrapOutcome {
        sources_collected: collected,
        sources_sent,
        sources_dropped,
        sources_by_kind: counts,
        estimated_input_tokens: total,
        pages_written: Vec::new(),
        rationale: "(dry-run; LLM not invoked)".to_string(),
        dry_run: true,
        llm_chunks,
    };
    Ok((
        StatusCode::OK,
        Json(serde_json::to_value(&outcome).unwrap_or_else(|_| serde_json::json!({}))),
    ))
}

// ---------------------------------------------------------------------
// reorg
// ---------------------------------------------------------------------

/// JSON request body for `POST /admin/reorg`.
#[derive(Deserialize)]
struct ReorgRequest {
    /// Show what would change without writing.
    #[serde(default)]
    dry_run: bool,
}

/// One entry in the reorg plan (serialised in the response).
#[derive(Debug, Serialize)]
pub struct ReorgPlanEntry {
    /// Session UUID string.
    pub session_id: String,
    /// Working directory the session was started in.
    pub cwd: String,
    /// Basename-derived project name the session will move to.
    pub new_project: String,
}

/// Summary counts returned alongside the plan.
#[derive(Debug, Serialize)]
pub struct ReorgSummaryJson {
    /// Sessions whose `project_id` was changed.
    pub sessions_moved: usize,
    /// Observations updated to match their session's new project.
    pub observations_updated: usize,
    /// `is_latest=1` pages marked `is_latest=0` (mash-up graveyard).
    pub pages_graveyarded: usize,
    /// Number of distinct per-cwd projects referenced in the plan.
    pub distinct_new_projects: usize,
}

/// Full response for `POST /admin/reorg`.
#[derive(Debug, Serialize)]
pub struct ReorgReport {
    /// `true` when `dry_run` was requested.
    pub dry_run: bool,
    /// All sessions that need (or needed) moving, with their target project.
    pub plan: Vec<ReorgPlanEntry>,
    /// Counts after execution (zeros when `dry_run=true`).
    pub summary: ReorgSummaryJson,
}

/// Read every session that has a non-empty `cwd` field, returning
/// `(session_id, project_id, cwd)` triples ordered by `started_at`.
async fn list_sessions_with_cwd(
    reader: &ai_memory_store::ReaderPool,
) -> Result<Vec<(SessionId, ProjectId, String)>, StoreError> {
    reader
        .with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, project_id, cwd \
                     FROM sessions \
                     WHERE cwd IS NOT NULL AND cwd != '' \
                     ORDER BY started_at",
            )?;
            let rows = stmt.query_map([], |row| {
                let id_bytes: Vec<u8> = row.get(0)?;
                let proj_bytes: Vec<u8> = row.get(1)?;
                let cwd: String = row.get(2)?;
                Ok((id_bytes, proj_bytes, cwd))
            })?;
            let mut out = Vec::new();
            for r in rows {
                let (id_bytes, proj_bytes, cwd) = r?;
                let sid = SessionId::from_slice(&id_bytes).map_err(StoreError::Memory)?;
                let pid = ProjectId::from_slice(&proj_bytes).map_err(StoreError::Memory)?;
                out.push((sid, pid, cwd));
            }
            Ok(out)
        })
        .await
}

/// Build the reorg plan: for each session with a new project basename,
/// resolve-or-create the target project, return the plan entries plus
/// the set of session updates and the distinct-project count.
async fn build_reorg_plan(
    state: &AdminState,
    ws: WorkspaceId,
    sessions: Vec<(SessionId, ProjectId, String)>,
) -> Result<(Vec<ReorgPlanEntry>, Vec<(SessionId, ProjectId)>, usize), StoreError> {
    // Resolve target project per distinct cwd (basename-derived).
    let mut cwd_to_proj: std::collections::HashMap<String, (WorkspaceId, ProjectId, String)> =
        std::collections::HashMap::new();
    for (_, _, cwd) in &sessions {
        if cwd_to_proj.contains_key(cwd.as_str()) {
            continue;
        }
        let project_name = std::path::Path::new(cwd.as_str())
            .file_name()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| "unknown".to_string());
        let proj = state
            .writer
            .get_or_create_project(ws, project_name.clone(), Some(cwd.clone()))
            .await?;
        cwd_to_proj.insert(cwd.clone(), (ws, proj, project_name));
    }

    // Build plan — sessions whose project_id already matches are skipped.
    let mut plan_entries: Vec<ReorgPlanEntry> = Vec::new();
    let mut writer_plan: Vec<(SessionId, ProjectId)> = Vec::new();
    for (session_id, old_project_id, cwd) in &sessions {
        let (_, new_project_id, project_name) = &cwd_to_proj[cwd.as_str()];
        if *new_project_id == *old_project_id {
            continue;
        }
        plan_entries.push(ReorgPlanEntry {
            session_id: session_id.to_string(),
            cwd: cwd.clone(),
            new_project: project_name.clone(),
        });
        writer_plan.push((*session_id, *new_project_id));
    }

    let distinct_new_projects: std::collections::HashSet<ProjectId> =
        writer_plan.iter().map(|(_, pid)| *pid).collect();

    Ok((plan_entries, writer_plan, distinct_new_projects.len()))
}

async fn handle_reorg(
    State(state): State<Arc<AdminState>>,
    Json(req): Json<ReorgRequest>,
) -> impl IntoResponse {
    let ws = match state
        .writer
        .get_or_create_workspace(DEFAULT_WORKSPACE_NAME)
        .await
    {
        Ok(id) => id,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("workspace: {e}") })),
            );
        }
    };

    let sessions = match list_sessions_with_cwd(&state.reader).await {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            );
        }
    };

    if sessions.is_empty() {
        let report = ReorgReport {
            dry_run: req.dry_run,
            plan: Vec::new(),
            summary: ReorgSummaryJson {
                sessions_moved: 0,
                observations_updated: 0,
                pages_graveyarded: 0,
                distinct_new_projects: 0,
            },
        };
        return (
            StatusCode::OK,
            Json(serde_json::to_value(&report).unwrap_or_else(|_| serde_json::json!({}))),
        );
    }

    let (plan_entries, writer_plan, distinct_count) =
        match build_reorg_plan(&state, ws, sessions).await {
            Ok(t) => t,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": format!("project: {e}") })),
                );
            }
        };

    if req.dry_run || writer_plan.is_empty() {
        let report = ReorgReport {
            dry_run: req.dry_run,
            plan: plan_entries,
            summary: ReorgSummaryJson {
                sessions_moved: 0,
                observations_updated: 0,
                pages_graveyarded: 0,
                distinct_new_projects: distinct_count,
            },
        };
        return (
            StatusCode::OK,
            Json(serde_json::to_value(&report).unwrap_or_else(|_| serde_json::json!({}))),
        );
    }

    let summary = match state.writer.reorg_sessions(writer_plan).await {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            );
        }
    };

    let report = ReorgReport {
        dry_run: false,
        plan: plan_entries,
        summary: ReorgSummaryJson {
            sessions_moved: summary.sessions_moved,
            observations_updated: summary.observations_updated,
            pages_graveyarded: summary.pages_graveyarded,
            distinct_new_projects: distinct_count,
        },
    };
    (
        StatusCode::OK,
        Json(serde_json::to_value(&report).unwrap_or_else(|_| serde_json::json!({}))),
    )
}

// ---------------------------------------------------------------------
// lint
// ---------------------------------------------------------------------

/// JSON request body for `POST /admin/lint`.
#[derive(Deserialize)]
struct LintRequest {
    /// Workspace name (auto-created if absent).
    #[serde(default = "default_workspace")]
    workspace: String,
    /// Project name (auto-created if absent).
    #[serde(default = "default_project")]
    project: String,
    /// Don't write the lint report page.
    #[serde(default)]
    dry_run: bool,
    /// Skip the LLM contradiction pass (rule-based findings only).
    /// When absent, defaults to `false` (LLM pass runs if a provider
    /// is configured).
    #[serde(default)]
    no_llm: bool,
}

fn default_workspace() -> String {
    DEFAULT_WORKSPACE_NAME.to_string()
}

fn default_project() -> String {
    DEFAULT_PROJECT_NAME.to_string()
}

async fn handle_lint(
    State(state): State<Arc<AdminState>>,
    Json(req): Json<LintRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let (ws, proj) = resolve_ws_proj(&state, &req.workspace, &req.project).await?;

    run_lint(
        &state.reader,
        &state.wiki,
        state.llm.as_ref(),
        ws,
        proj,
        req.dry_run,
        !req.no_llm,
    )
    .await
    .map(|report| {
        (
            StatusCode::OK,
            Json(serde_json::to_value(&report).unwrap_or_else(|_| serde_json::json!({}))),
        )
    })
    .map_err(|e| internal_err(e.to_string()))
}

// ---------------------------------------------------------------------
// forget-sweep
// ---------------------------------------------------------------------

/// JSON request body for `POST /admin/forget-sweep`.
#[derive(Deserialize)]
struct ForgetSweepRequest {
    /// Workspace name (auto-created if absent).
    #[serde(default = "default_workspace")]
    workspace: String,
    /// Project name (auto-created if absent).
    #[serde(default = "default_project")]
    project: String,
    /// Report what would be evicted without actually mutating.
    #[serde(default)]
    dry_run: bool,
}

async fn handle_forget_sweep(
    State(state): State<Arc<AdminState>>,
    Json(req): Json<ForgetSweepRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let (ws, proj) = resolve_ws_proj(&state, &req.workspace, &req.project).await?;

    run_sweep(
        &state.reader,
        &state.writer,
        ws,
        proj,
        &state.decay_params,
        req.dry_run,
    )
    .await
    .map(|report| {
        (
            StatusCode::OK,
            Json(serde_json::to_value(&report).unwrap_or_else(|_| serde_json::json!({}))),
        )
    })
    .map_err(|e| internal_err(e.to_string()))
}

// ---------------------------------------------------------------------
// embed
// ---------------------------------------------------------------------

/// JSON request body for `POST /admin/embed`.
#[derive(Deserialize)]
struct EmbedRequest {
    /// Workspace name (auto-created if absent).
    #[serde(default = "default_workspace")]
    workspace: String,
    /// Project name (auto-created if absent). Ignored when
    /// [`Self::all_projects`] is true.
    #[serde(default = "default_project")]
    project: String,
    /// When true, regenerates embeddings even for pages that already
    /// have one matching the current (provider, model, dim).
    #[serde(default)]
    reembed: bool,
    /// When true, count pages that would be embedded/skipped without
    /// calling the embedder or writing anything.
    #[serde(default)]
    dry_run: bool,
    /// When true, embed every project in `workspace` instead of a
    /// single `project`. The CLI sets this for `--force` /
    /// `--reembed` without an explicit `--project` so model migrations
    /// reach stale rows in every namespace.
    #[serde(default)]
    all_projects: bool,
}

/// Summary response from `POST /admin/embed`.
#[derive(Debug, Serialize)]
pub struct EmbedReport {
    /// Pages that were actually embedded (zero in dry-run).
    pub embedded: usize,
    /// Pages skipped because a matching embedding already existed.
    pub skipped: usize,
    /// Pages that failed to embed (read error or provider error).
    pub failed: usize,
    /// Pages that would be embedded in a live run (only meaningful
    /// when `dry_run` was requested).
    pub would_embed: usize,
    /// Provider name.
    pub provider: String,
    /// Model identifier.
    pub model: String,
    /// Embedding dimensionality.
    pub dim: u32,
}

#[derive(Default)]
struct EmbedCounts {
    embedded: usize,
    skipped: usize,
    failed: usize,
    would_embed: usize,
}

impl EmbedCounts {
    fn absorb(&mut self, other: Self) {
        self.embedded += other.embedded;
        self.skipped += other.skipped;
        self.failed += other.failed;
        self.would_embed += other.would_embed;
    }
}

async fn embed_project_pages(
    state: &AdminState,
    embedder: &Arc<dyn Embedder>,
    ws: WorkspaceId,
    proj: ProjectId,
    reembed: bool,
    dry_run: bool,
) -> Result<EmbedCounts, (StatusCode, Json<serde_json::Value>)> {
    let provider = embedder.provider().to_string();
    let model = embedder.model().to_string();
    let dim = embedder.dim();

    let candidates = state
        .reader
        .decay_candidates(ws, proj)
        .await
        .map_err(|e| internal_err(e.to_string()))?;

    let already: std::collections::HashSet<_> = if reembed {
        std::collections::HashSet::new()
    } else {
        state
            .reader
            .embedded_page_ids(ws, proj, provider.clone(), model.clone(), dim)
            .await
            .map_err(|e| internal_err(e.to_string()))?
            .into_iter()
            .collect()
    };

    let mut counts = EmbedCounts::default();
    let mut pending = Vec::with_capacity(EMBEDDING_WRITE_BATCH);

    for cand in candidates {
        if !reembed && already.contains(&cand.id) {
            counts.skipped += 1;
            continue;
        }
        if dry_run {
            counts.would_embed += 1;
            continue;
        }
        let md = match state.wiki.read_page(ws, proj, &cand.path) {
            Ok(m) => m,
            Err(e) => {
                warn!(path = %cand.path, error = %e, "embed: skip unreadable page");
                counts.failed += 1;
                continue;
            }
        };
        if md.body.trim().is_empty() {
            counts.skipped += 1;
            continue;
        }
        let vec = match embedder.embed_document(&md.body).await {
            Ok(v) => v,
            Err(e) => {
                warn!(path = %cand.path, error = %e, "embed: provider call failed");
                counts.failed += 1;
                continue;
            }
        };
        pending.push(EmbeddingWrite {
            page_id: cand.id,
            vector_bytes: f32_vec_to_bytes(&vec),
            provider: provider.clone(),
            model: model.clone(),
            dim,
        });
        if pending.len() >= EMBEDDING_WRITE_BATCH {
            flush_embedding_batch(
                &state.writer,
                &mut pending,
                &mut counts.embedded,
                &mut counts.failed,
            )
            .await;
        }
    }
    flush_embedding_batch(
        &state.writer,
        &mut pending,
        &mut counts.embedded,
        &mut counts.failed,
    )
    .await;

    Ok(counts)
}

async fn handle_embed(
    State(state): State<Arc<AdminState>>,
    Json(req): Json<EmbedRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let embedder = match state.embedder.clone() {
        Some(e) => e,
        None => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "embedder not configured on server"
                })),
            ));
        }
    };

    let provider = embedder.provider().to_string();
    let model = embedder.model().to_string();
    let dim = embedder.dim();

    let mut totals = EmbedCounts::default();

    if req.all_projects {
        if let Some(ws) = state
            .reader
            .find_workspace(req.workspace.clone())
            .await
            .map_err(|e| internal_err(e.to_string()))?
        {
            if req.reembed && !req.dry_run {
                let purged = state
                    .writer
                    .delete_stale_page_embeddings(ws, None, provider.clone(), model.clone(), dim)
                    .await
                    .map_err(|e| internal_err(e.to_string()))?;
                info!(purged, provider = %provider, model = %model, "purged stale page_embeddings before workspace re-embed");
            }

            let summaries = state
                .reader
                .list_projects_with_stats()
                .await
                .map_err(|e| internal_err(e.to_string()))?;
            for summary in summaries
                .into_iter()
                .filter(|p| p.workspace_name == req.workspace)
            {
                let Some(proj) = state
                    .reader
                    .find_project(ws, summary.project_name.clone())
                    .await
                    .map_err(|e| internal_err(e.to_string()))?
                else {
                    continue;
                };
                let partial =
                    embed_project_pages(&state, &embedder, ws, proj, req.reembed, req.dry_run)
                        .await?;
                totals.absorb(partial);
            }
        }
    } else {
        let (ws, proj) = resolve_ws_proj(&state, &req.workspace, &req.project).await?;
        if req.reembed && !req.dry_run {
            let purged = state
                .writer
                .delete_stale_page_embeddings(ws, Some(proj), provider.clone(), model.clone(), dim)
                .await
                .map_err(|e| internal_err(e.to_string()))?;
            info!(purged, provider = %provider, model = %model, "purged stale page_embeddings before project re-embed");
        }
        totals = embed_project_pages(&state, &embedder, ws, proj, req.reembed, req.dry_run).await?;
    }

    let report = EmbedReport {
        embedded: totals.embedded,
        skipped: totals.skipped,
        failed: totals.failed,
        would_embed: totals.would_embed,
        provider,
        model,
        dim,
    };
    Ok((
        StatusCode::OK,
        Json(serde_json::to_value(&report).unwrap_or_else(|_| serde_json::json!({}))),
    ))
}

async fn flush_embedding_batch(
    writer: &WriterHandle,
    pending: &mut Vec<EmbeddingWrite>,
    embedded: &mut usize,
    failed: &mut usize,
) {
    if pending.is_empty() {
        return;
    }
    let batch = std::mem::replace(pending, Vec::with_capacity(EMBEDDING_WRITE_BATCH));
    let count = batch.len();
    if let Err(e) = writer.store_embeddings(batch).await {
        *failed += count;
        warn!(count, error = %e, "embed: store_embeddings failed");
    } else {
        *embedded += count;
    }
}

// ---------------------------------------------------------------------
// commit
// ---------------------------------------------------------------------

/// JSON request body for `POST /admin/commit`.
#[derive(Deserialize)]
struct CommitRequest {
    /// Commit message.
    message: String,
}

async fn handle_commit(
    State(state): State<Arc<AdminState>>,
    Json(req): Json<CommitRequest>,
) -> impl IntoResponse {
    match state.wiki.commit_all(&req.message) {
        Ok(Some(oid)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "committed": true,
                "oid": oid.to_string(),
            })),
        ),
        Ok(None) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "committed": false,
                "reason": "nothing to commit",
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        ),
    }
}

// ---------------------------------------------------------------------
// purge-project
// ---------------------------------------------------------------------

/// JSON request body for `POST /admin/purge-project`.
#[derive(Deserialize)]
struct PurgeProjectRequest {
    /// Workspace name. The workspace must already exist; we 404 if absent.
    workspace: String,
    /// Project name. The project must already exist; we 404 if absent.
    project: String,
    /// Mandatory confirmation flag. Without `confirm: true` the server
    /// returns 400 — purging is destructive and irreversible.
    confirm: bool,
}

/// Wire-format summary returned by `POST /admin/purge-project`.
#[derive(Debug, Serialize)]
pub struct PurgeProjectReport {
    /// Human-readable `workspace/project` label.
    pub label: String,
    /// Number of `pages` rows deleted (all versions).
    pub pages_deleted: u64,
    /// Number of `sessions` rows deleted.
    pub sessions_deleted: u64,
    /// Number of `observations` rows deleted.
    pub observations_deleted: u64,
    /// Number of `handoffs` rows deleted.
    pub handoffs_deleted: u64,
    /// Number of `page_embeddings` rows deleted.
    pub embeddings_deleted: u64,
    /// Paths removed from disk (the project's UUID-namespaced directory).
    pub files_deleted: Vec<String>,
    /// Paths that could not be removed from disk (non-fatal; DB rows are gone).
    pub files_failed: Vec<String>,
}

async fn handle_purge_project(
    State(state): State<Arc<AdminState>>,
    Json(req): Json<PurgeProjectRequest>,
) -> impl IntoResponse {
    if !req.confirm {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "destructive operation requires confirm=true"
            })),
        );
    }

    // Look up workspace and project IDs without auto-creating.
    let (ws_id, proj_id) =
        match lookup_ws_proj_no_create(&state, &req.workspace, &req.project).await {
            Ok(ids) => ids,
            Err(e) => return e,
        };

    let label = format!("{}/{}", req.workspace, req.project);

    // Admission must run before any destructive work. A reject-policy webhook
    // is allowed to abort the purge while DB rows and files are still intact.
    // Seed names from the request so mirrors do not depend on DB lookup after
    // the rows are purged below.
    let purge_ctx = AdmissionContext {
        workspace: req.workspace.clone(),
        project: req.project.clone(),
        op: AdmissionOp::PurgeProject,
        ..Default::default()
    };
    let resolved_purge_ctx = match state
        .wiki
        .admit_purge_project(ws_id, proj_id, Some(purge_ctx))
        .await
    {
        Ok(ctx) => ctx,
        Err(e) => return internal_err(e.to_string()),
    };

    let summary = match state.writer.purge_project(ws_id, proj_id, &label).await {
        Ok(s) => s,
        Err(e) => return internal_err(e.to_string()),
    };

    // Remove the entire per-project directory: <wiki_root>/<ws_uuid>/<proj_uuid>/.
    // DB cascade already deleted all rows. Directory removal remains best-effort
    // and is reported separately, matching the pre-admission purge contract.
    let proj_root_str = state
        .wiki
        .project_root(ws_id, proj_id)
        .display()
        .to_string();
    let mut files_deleted: Vec<String> = Vec::new();
    let mut files_failed: Vec<String> = Vec::new();
    match state.wiki.remove_project_dir(ws_id, proj_id) {
        Ok(()) => {
            files_deleted.push(proj_root_str);
        }
        Err(e) => {
            warn!(path = %proj_root_str, error = %e, "purge-project: failed to remove project dir");
            files_failed.push(proj_root_str);
        }
    }
    state
        .wiki
        .dispatch_purge_project(resolved_purge_ctx.as_ref());

    let report = PurgeProjectReport {
        label: summary.label,
        pages_deleted: summary.pages_deleted,
        sessions_deleted: summary.sessions_deleted,
        observations_deleted: summary.observations_deleted,
        handoffs_deleted: summary.handoffs_deleted,
        embeddings_deleted: summary.embeddings_deleted,
        files_deleted,
        files_failed,
    };

    (
        StatusCode::OK,
        Json(serde_json::to_value(&report).unwrap_or_else(|_| serde_json::json!({}))),
    )
}

// ---------------------------------------------------------------------
// rename-project
// ---------------------------------------------------------------------

/// JSON request body for `POST /admin/rename-project`.
#[derive(Deserialize)]
struct RenameProjectRequest {
    /// Workspace name (must exist; we don't auto-create on rename).
    workspace: String,
    /// Current project name.
    from: String,
    /// New project name. Must be non-empty, no slashes.
    to: String,
}

/// Wire-format summary returned by `POST /admin/rename-project`.
#[derive(Debug, Serialize)]
pub struct RenameProjectSummary {
    /// Workspace name.
    pub workspace: String,
    /// Previous project name.
    pub from: String,
    /// New project name.
    pub to: String,
    /// Number of `is_latest=1` pages now under the renamed project.
    /// No files move — this is purely an informational count.
    pub pages: u64,
}

async fn handle_rename_project(
    State(state): State<Arc<AdminState>>,
    Json(req): Json<RenameProjectRequest>,
) -> impl IntoResponse {
    // Look up workspace + source project; 404 if either is absent.
    let (ws_id, proj_id) = match lookup_ws_proj_no_create(&state, &req.workspace, &req.from).await {
        Ok(ids) => ids,
        Err(e) => return e,
    };

    // Step 3: execute the rename. The writer validates the name and
    // returns ProjectNameTaken / InvalidProjectName on conflicts.
    if let Err(e) = state
        .writer
        .rename_project(ws_id, proj_id, req.to.clone())
        .await
    {
        let status = match &e {
            StoreError::ProjectNameTaken(_) | StoreError::InvalidProjectName(_) => {
                StatusCode::UNPROCESSABLE_ENTITY
            }
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        return (status, Json(serde_json::json!({ "error": e.to_string() })));
    }

    // Step 4: count is_latest pages that now belong to the renamed project.
    // COUNT(*) always produces a row, so no optional() needed. We pass the
    // 16-byte project id as a plain &[u8] slice to avoid importing rusqlite
    // (not a direct dependency of this crate).
    let pid_bytes = *proj_id.as_bytes();
    let pages = match state
        .reader
        .with_conn(move |conn| {
            let n: i64 = conn.query_row(
                "SELECT COUNT(*) FROM pages WHERE project_id = ?1 AND is_latest = 1",
                [&pid_bytes[..]],
                |row| row.get(0),
            )?;
            Ok(u64::try_from(n).unwrap_or(0))
        })
        .await
    {
        Ok(n) => n,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            );
        }
    };

    let summary = RenameProjectSummary {
        workspace: req.workspace,
        from: req.from,
        to: req.to,
        pages,
    };
    (
        StatusCode::OK,
        Json(serde_json::to_value(&summary).unwrap_or_else(|_| serde_json::json!({}))),
    )
}

// ---------------------------------------------------------------------
// write-page
// ---------------------------------------------------------------------

/// JSON request body for `POST /admin/write-page`.
#[derive(Deserialize)]
struct WritePageAdminRequest {
    /// Workspace name (auto-created if absent).
    workspace: String,
    /// Project name (auto-created if absent).
    project: String,
    /// Relative wiki path (e.g. `concepts/foo.md`).
    path: String,
    /// Markdown body. The server frames it with frontmatter; pass plain body content.
    body: String,
    /// Optional title; derived from first H1 or path stem when absent.
    #[serde(default)]
    title: Option<String>,
    /// Semantic kind (`fact`, `rule`, `decision`, `gotcha`). Stored in
    /// the page frontmatter; the reader falls back to a path-derived
    /// kind when absent.
    #[serde(default)]
    kind: Option<String>,
    /// Tier name (`working`, `episodic`, `semantic`, `procedural`).
    #[serde(default = "default_write_tier")]
    tier: String,
    /// Tags to attach to the page.
    #[serde(default)]
    tags: Vec<String>,
    /// Pin the page so the decay sweep skips it.
    #[serde(default)]
    pinned: bool,
}

fn default_write_tier() -> String {
    "semantic".to_string()
}

/// JSON response body for `POST /admin/write-page`.
#[derive(Serialize)]
struct WritePageResponse {
    /// UUID of the written page.
    page_id: String,
    /// Canonical wiki path.
    path: String,
}

async fn handle_write_page(
    State(state): State<Arc<AdminState>>,
    actor_ext: Option<axum::Extension<ai_memory_core::ActorContext>>,
    author_ext: Option<axum::Extension<ai_memory_core::UserId>>,
    level_ext: Option<axum::Extension<ai_memory_core::AuthLevel>>,
    headers: HeaderMap,
    Json(req): Json<WritePageAdminRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let tier: Tier = req.tier.parse().map_err(|_| {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": format!("unknown tier '{}'", req.tier)
            })),
        )
    })?;

    let path = PagePath::new(req.path.clone()).map_err(|e| {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({ "error": format!("invalid path: {e}") })),
        )
    })?;

    let (ws, proj) = resolve_ws_proj(&state, &req.workspace, &req.project).await?;

    let mut fm = serde_json::Map::new();
    if let Some(title) = &req.title {
        fm.insert("title".into(), serde_json::Value::String(title.clone()));
    }
    if let Some(kind) = req.kind.as_deref() {
        let kind = kind.trim();
        if !kind.is_empty() {
            fm.insert("kind".into(), serde_json::Value::String(kind.to_string()));
        }
    }
    if !req.tags.is_empty() {
        fm.insert(
            "tags".into(),
            serde_json::Value::Array(
                req.tags
                    .into_iter()
                    .map(serde_json::Value::String)
                    .collect(),
            ),
        );
    }
    if req.pinned {
        fm.insert("pinned".into(), serde_json::Value::Bool(true));
    }
    let frontmatter = if fm.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::Value::Object(fm)
    };

    let actor = actor_ext
        .map(|axum::Extension(actor)| actor)
        .unwrap_or_else(ai_memory_core::ActorContext::anonymous);
    let author_id = author_ext.map(|axum::Extension(author_id)| author_id);
    let skip_webhooks = match level_ext.map(|axum::Extension(level)| level) {
        Some(ai_memory_core::AuthLevel::User) => Vec::new(),
        Some(ai_memory_core::AuthLevel::Root | ai_memory_core::AuthLevel::Anonymous) | None => {
            crate::actor::skip_webhooks_from_headers(&headers)
        }
    };
    let admission_ctx = if actor.has_any() || !skip_webhooks.is_empty() {
        Some(AdmissionContext {
            op: AdmissionOp::WritePage,
            skip_webhooks,
            ..AdmissionContext::default()
        })
    } else {
        None
    };

    let page_id = state
        .wiki
        .write_page(WritePageRequest {
            workspace_id: ws,
            project_id: proj,
            path: path.clone(),
            frontmatter,
            body: req.body,
            tier,
            pinned: req.pinned,
            title: req.title,
            admission_ctx,
            author_id,
            actor,
        })
        .await
        .map_err(|e| internal_err(e.to_string()))?;

    Ok((
        StatusCode::OK,
        Json(
            serde_json::to_value(WritePageResponse {
                page_id: page_id.to_string(),
                path: path.to_string(),
            })
            .unwrap_or_else(|_| serde_json::json!({})),
        ),
    ))
}

// ---------------------------------------------------------------------
// user management (root-only)
// ---------------------------------------------------------------------

/// JSON request body for `POST /admin/users`.
#[derive(Debug, Deserialize)]
struct CreateUserRequest {
    username: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    email: Option<String>,
}

/// JSON response for `POST /admin/users` and `…/rotate-token`.
/// Carries the plaintext token EXACTLY once — the client (CLI or
/// admin browser) must surface it to the operator and never persist
/// it; only the SHA-256 digest is kept in the DB. Subsequent reads
/// (`GET /admin/users`) omit the token field entirely.
#[derive(Debug, Serialize)]
struct UserWithTokenResponse {
    user: ai_memory_core::User,
    token: String,
}

/// JSON response for `GET /admin/users` and lifecycle ops that don't
/// issue a new token (expire, revive).
#[derive(Debug, Serialize)]
struct UserResponse {
    user: ai_memory_core::User,
}

/// JSON response for `GET /admin/users`.
#[derive(Debug, Serialize)]
struct UserListResponse {
    users: Vec<ai_memory_core::User>,
}

/// Gate any handler in this section on a root-level request. Returns
/// the matching error response for the actor's tier (401 anonymous,
/// 403 user) or `Ok(())` for root.
fn require_root(
    level: ai_memory_core::AuthLevel,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    if level.is_root() {
        return Ok(());
    }
    let (code, msg) = match level {
        ai_memory_core::AuthLevel::Anonymous => (
            StatusCode::UNAUTHORIZED,
            "user management requires authentication",
        ),
        ai_memory_core::AuthLevel::User => (StatusCode::FORBIDDEN, "user management is root-only"),
        ai_memory_core::AuthLevel::Root => unreachable!("guarded above"),
    };
    Err((code, Json(serde_json::json!({ "error": msg }))))
}

/// Get the active token-pepper. Returns 503 when multi-user wasn't
/// configured — same shape as `/admin/embed` returns when no embedder
/// is wired.
fn require_pepper(
    state: &AdminState,
) -> Result<&ai_memory_store::TokenPepper, (StatusCode, Json<serde_json::Value>)> {
    state.token_pepper.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "multi-user not enabled (set [auth].token_pepper in config or run `ai-memory init`)"
            })),
        )
    })
}

/// Handler for `POST /admin/users`.
///
/// Validates the input, generates a fresh 32-byte token, hashes it with
/// the per-server pepper, and inserts the row. Returns
/// `UserWithTokenResponse` so the caller can display the plaintext
/// token exactly once.
async fn handle_create_user(
    State(state): State<Arc<AdminState>>,
    axum::Extension(level): axum::Extension<ai_memory_core::AuthLevel>,
    Json(req): Json<CreateUserRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    require_root(level)?;
    let pepper = require_pepper(&state)?;

    let mut new_user = ai_memory_core::NewUser {
        username: req.username,
        name: req.name,
        email: req.email,
    };
    new_user
        .validate()
        .map_err(|e| validation_error(e.to_string()))?;

    let token = ai_memory_store::generate_token().map_err(|e| internal_err(e.to_string()))?;
    let token_hash = ai_memory_store::hash_token(&token, pepper);

    let user_id = state
        .writer
        .create_user(new_user.clone(), token_hash)
        .await
        .map_err(map_user_store_err)?;

    // Round-trip through the reader so we surface the same canonical
    // shape `GET /admin/users` returns (incl. created_at).
    let user = state
        .reader
        .find_user_by_id(user_id)
        .await
        .map_err(|e| internal_err(e.to_string()))?
        .ok_or_else(|| internal_err("created user vanished from store".to_string()))?;

    Ok((
        StatusCode::OK,
        Json(serde_json::to_value(UserWithTokenResponse { user, token }).unwrap_or_default()),
    ))
}

/// Handler for `GET /admin/users`. Includes users with expired tokens
/// (the response's `token_expired_at` field distinguishes them); the
/// CLI list renderer shows an "expired" flag.
async fn handle_list_users(
    State(state): State<Arc<AdminState>>,
    axum::Extension(level): axum::Extension<ai_memory_core::AuthLevel>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    require_root(level)?;
    let users = state
        .reader
        .list_users()
        .await
        .map_err(|e| internal_err(e.to_string()))?;
    Ok((
        StatusCode::OK,
        Json(serde_json::to_value(UserListResponse { users }).unwrap_or_default()),
    ))
}

/// Handler for `POST /admin/users/:username/expire`. Idempotent: the
/// first call stamps `token_expired_at = now()`, subsequent calls
/// leave the original timestamp untouched (via COALESCE in the store).
async fn handle_expire_user(
    State(state): State<Arc<AdminState>>,
    axum::Extension(level): axum::Extension<ai_memory_core::AuthLevel>,
    axum::extract::Path(username): axum::extract::Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    require_root(level)?;
    let user = lookup_user_by_username(&state, &username).await?;
    state
        .writer
        .expire_user_token(user.id)
        .await
        .map_err(|e| internal_err(e.to_string()))?;
    // Re-read to surface the new token_expired_at in the response.
    let user = state
        .reader
        .find_user_by_id(user.id)
        .await
        .map_err(|e| internal_err(e.to_string()))?
        .ok_or_else(|| internal_err("user vanished after expire".to_string()))?;
    Ok((
        StatusCode::OK,
        Json(serde_json::to_value(UserResponse { user }).unwrap_or_default()),
    ))
}

/// Handler for `POST /admin/users/:username/revive`. Clears
/// `token_expired_at`. Idempotent (revive on an active user is a no-op).
async fn handle_revive_user(
    State(state): State<Arc<AdminState>>,
    axum::Extension(level): axum::Extension<ai_memory_core::AuthLevel>,
    axum::extract::Path(username): axum::extract::Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    require_root(level)?;
    let user = lookup_user_by_username(&state, &username).await?;
    state
        .writer
        .revive_user_token(user.id)
        .await
        .map_err(|e| internal_err(e.to_string()))?;
    let user = state
        .reader
        .find_user_by_id(user.id)
        .await
        .map_err(|e| internal_err(e.to_string()))?
        .ok_or_else(|| internal_err("user vanished after revive".to_string()))?;
    Ok((
        StatusCode::OK,
        Json(serde_json::to_value(UserResponse { user }).unwrap_or_default()),
    ))
}

/// Handler for `POST /admin/users/:username/rotate-token`. Issues a
/// fresh token, hashes it with the server pepper, replaces the row's
/// `token_hash`, and implicitly clears `token_expired_at` (rotating
/// makes the new token usable immediately even if the prior one was
/// expired). Returns the plaintext token once.
async fn handle_rotate_user_token(
    State(state): State<Arc<AdminState>>,
    axum::Extension(level): axum::Extension<ai_memory_core::AuthLevel>,
    axum::extract::Path(username): axum::extract::Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    require_root(level)?;
    let pepper = require_pepper(&state)?;
    let user = lookup_user_by_username(&state, &username).await?;

    let token = ai_memory_store::generate_token().map_err(|e| internal_err(e.to_string()))?;
    let token_hash = ai_memory_store::hash_token(&token, pepper);

    let updated = state
        .writer
        .rotate_user_token(user.id, token_hash)
        .await
        .map_err(|e| internal_err(e.to_string()))?;
    if !updated {
        return Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "user vanished mid-rotation" })),
        ));
    }
    let user = state
        .reader
        .find_user_by_id(user.id)
        .await
        .map_err(|e| internal_err(e.to_string()))?
        .ok_or_else(|| internal_err("user vanished after rotate".to_string()))?;
    Ok((
        StatusCode::OK,
        Json(serde_json::to_value(UserWithTokenResponse { user, token }).unwrap_or_default()),
    ))
}

/// Shared lookup helper: 404 when the user doesn't exist, else returns
/// the row. Used by every per-username handler so error shapes stay
/// uniform across `expire` / `revive` / `rotate-token`.
async fn lookup_user_by_username(
    state: &AdminState,
    username: &str,
) -> Result<ai_memory_core::User, (StatusCode, Json<serde_json::Value>)> {
    let found = state
        .reader
        .find_user_by_username(username.to_string())
        .await
        .map_err(|e| internal_err(e.to_string()))?;
    found.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("no such user: {username}") })),
        )
    })
}

/// Convert a username/email validation error into a 400 response.
fn validation_error(msg: String) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": msg })),
    )
}

/// Map StoreError to the right HTTP status. UNIQUE violations on
/// username / email become 409; everything else is a 500.
fn map_user_store_err(e: ai_memory_store::StoreError) -> (StatusCode, Json<serde_json::Value>) {
    match e {
        ai_memory_store::StoreError::Duplicate(msg) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": msg })),
        ),
        ai_memory_store::StoreError::Memory(ai_memory_core::MemoryError::InvalidUsername(msg))
        | ai_memory_store::StoreError::Memory(ai_memory_core::MemoryError::InvalidEmail(msg)) => {
            validation_error(msg)
        }
        other => internal_err(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_memory_store::Store;
    use axum::body::to_bytes;
    use axum::http::Request;
    use tempfile::TempDir;
    use tower::ServiceExt;

    #[tokio::test]
    async fn status_reports_provider_health_block() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();
        let router = admin_router(AdminState {
            writer: store.writer.clone(),
            reader: store.reader.clone(),
            wiki,
            llm: None,
            embedder: None,
            provider_health: ProviderHealth::default(),
            decay_params: DecayParams::default(),
            data_dir: tmp.path().to_path_buf(),
            db_path: store.db_path().to_path_buf(),
            bind: "127.0.0.1:49374".to_string(),
            bootstrap_lock: Arc::new(tokio::sync::Mutex::new(())),
            token_pepper: None,
        });

        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/admin/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["providers"]["llm"]["status"], "disabled");
        assert_eq!(json["providers"]["embedding"]["status"], "disabled");
    }

    fn read_page_test_router() -> (TempDir, Router) {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();
        let router = admin_router(AdminState {
            writer: store.writer.clone(),
            reader: store.reader.clone(),
            wiki,
            llm: None,
            embedder: None,
            provider_health: ProviderHealth::default(),
            decay_params: DecayParams::default(),
            data_dir: tmp.path().to_path_buf(),
            db_path: store.db_path().to_path_buf(),
            bind: "127.0.0.1:49374".to_string(),
            bootstrap_lock: Arc::new(tokio::sync::Mutex::new(())),
            token_pepper: None,
        });
        (tmp, router)
    }

    async fn post_write_page(router: &Router, ws: &str, project: &str, path: &str, body: &str) {
        let req_body = serde_json::json!({
            "workspace": ws,
            "project": project,
            "path": path,
            "body": body,
            "title": "Read-back fixture",
        });
        let resp = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/write-page")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "write-page setup failed");
    }

    #[tokio::test]
    async fn read_page_path_mode_returns_full_body() {
        let (_tmp, router) = read_page_test_router();
        post_write_page(&router, "default", "audit", "notes/foo.md", "hello body").await;

        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/admin/read-page?workspace=default&project=audit&path=notes/foo.md")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["path"], "notes/foo.md");
        assert_eq!(json["title"], "Read-back fixture");
        assert!(
            json["body"]
                .as_str()
                .unwrap_or_default()
                .contains("hello body"),
            "body must round-trip; got {:?}",
            json["body"]
        );
        assert_eq!(json["workspace"], "default");
        assert_eq!(json["project"], "audit");
    }

    #[tokio::test]
    async fn read_page_missing_path_returns_404() {
        let (_tmp, router) = read_page_test_router();

        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/admin/read-page?workspace=default&project=audit&path=notes/nope.md")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn read_page_traversal_rejected_with_400() {
        let (_tmp, router) = read_page_test_router();

        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/admin/read-page?workspace=default&project=audit&path=../etc/passwd")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn read_page_without_path_or_query_returns_400() {
        let (_tmp, router) = read_page_test_router();

        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/admin/read-page?workspace=default&project=audit")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// Bug E regression: purge deletes the project's DB rows first, so by the
    /// time the admission chain fires, name-resolution from the (now gone)
    /// project row would yield an empty name and a name-based mirror would
    /// purge the wrong path. The handler must seed the admission context with
    /// the request's workspace/project names. We capture the `purge_project`
    /// webhook and assert it carries the real project name.
    #[tokio::test]
    async fn purge_project_admission_carries_the_project_name() {
        use ai_memory_wiki::{AdmissionChain, FailurePolicy, WebhookConfig};
        use axum::http::HeaderMap;
        use axum::routing::post;
        use std::sync::Mutex;

        let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let cap = captured.clone();
        let app = Router::new().route(
            "/sync",
            post(
                move |headers: HeaderMap, Json(payload): Json<serde_json::Value>| {
                    let cap = cap.clone();
                    async move {
                        if headers.get("X-Memory-Op").and_then(|v| v.to_str().ok())
                            == Some("purge_project")
                        {
                            *cap.lock().unwrap() = Some(
                                payload["ctx"]["project"]
                                    .as_str()
                                    .unwrap_or_default()
                                    .to_string(),
                            );
                        }
                        StatusCode::NO_CONTENT
                    }
                },
            ),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let chain = AdmissionChain::new(vec![WebhookConfig {
            name: "mirror".into(),
            url: format!("{base}/sync"),
            timeout_ms: 2_000,
            failure_policy: FailurePolicy::Ignore,
            events: vec![AdmissionOp::WritePage, AdmissionOp::PurgeProject],
            blocking: true,
        }])
        .unwrap();
        let wiki = Wiki::new(tmp.path(), store.writer.clone())
            .unwrap()
            .with_admission_chain(chain)
            .with_store_reader(store.reader.clone());
        let router = admin_router(AdminState {
            writer: store.writer.clone(),
            reader: store.reader.clone(),
            wiki,
            llm: None,
            embedder: None,
            provider_health: ProviderHealth::default(),
            decay_params: DecayParams::default(),
            data_dir: tmp.path().to_path_buf(),
            db_path: store.db_path().to_path_buf(),
            bind: "127.0.0.1:49374".to_string(),
            bootstrap_lock: Arc::new(tokio::sync::Mutex::new(())),
            token_pepper: None,
        });

        post_write_page(&router, "default", "doomed", "notes/x.md", "bye").await;

        let resp = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/purge-project")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&serde_json::json!({
                            "workspace": "default",
                            "project": "doomed",
                            "confirm": true,
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "purge should succeed");

        // Let the async notify land.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(
            captured.lock().unwrap().as_deref(),
            Some("doomed"),
            "purge admission must carry the real project name, not an empty/_unscoped placeholder"
        );
    }

    // ── user-management endpoints (P1.4) ──────────────────────────

    /// Test router that has a token pepper configured (so user-management
    /// endpoints don't return 503), runs the standard auth middleware
    /// upstream so `Extension<AuthLevel>` is populated, and is reachable
    /// as Root via a fixed bearer token.
    fn user_admin_test_router(root_token: &'static str) -> (TempDir, Router) {
        use ai_memory_core::ActorContext;
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();
        let pepper = ai_memory_store::TokenPepper::new("test-pepper-admin");
        let router = admin_router(AdminState {
            writer: store.writer.clone(),
            reader: store.reader.clone(),
            wiki,
            llm: None,
            embedder: None,
            provider_health: ProviderHealth::default(),
            decay_params: DecayParams::default(),
            data_dir: tmp.path().to_path_buf(),
            db_path: store.db_path().to_path_buf(),
            bind: "127.0.0.1:49374".to_string(),
            bootstrap_lock: Arc::new(tokio::sync::Mutex::new(())),
            token_pepper: Some(pepper),
        });
        // Wrap in a middleware that stamps the AuthLevel ourselves —
        // the real auth middleware lives in ai-memory-cli, and this
        // crate can't depend on it. The wrapper matches the Bearer
        // header against `root_token`: match → Root, present-but-no-
        // match → User (so we can test 403), absent → Anonymous.
        let router = router.layer(axum::middleware::from_fn(
            move |mut req: Request<Body>, next: axum::middleware::Next| async move {
                let bearer = req
                    .headers()
                    .get(axum::http::header::AUTHORIZATION)
                    .and_then(|h| h.to_str().ok())
                    .and_then(|s| s.strip_prefix("Bearer "))
                    .map(str::to_string);
                let level = match bearer.as_deref() {
                    Some(t) if t == root_token => ai_memory_core::AuthLevel::Root,
                    Some(_) => ai_memory_core::AuthLevel::User,
                    None => ai_memory_core::AuthLevel::Anonymous,
                };
                req.extensions_mut().insert(level);
                req.extensions_mut().insert(ActorContext::anonymous());
                next.run(req).await
            },
        ));
        (tmp, router)
    }

    async fn post_create_user(
        router: &Router,
        root_token: &str,
        body: serde_json::Value,
    ) -> axum::http::Response<Body> {
        router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/users")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {root_token}"))
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn create_user_happy_path_returns_token_once() {
        let (_tmp, router) = user_admin_test_router("root-token");
        let resp = post_create_user(
            &router,
            "root-token",
            serde_json::json!({
                "username": "alice",
                "name": "Alice Smith",
                "email": "Alice@Example.com"
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["user"]["username"], "alice");
        // Email was normalised to lowercase by NewUser::validate.
        assert_eq!(json["user"]["email"], "alice@example.com");
        assert_eq!(json["user"]["name"], "Alice Smith");
        // Plaintext token is surfaced exactly once — 43 chars (32 bytes
        // URL-safe-base64).
        let token = json["token"].as_str().unwrap();
        assert_eq!(token.len(), 43);
    }

    #[tokio::test]
    async fn create_user_rejects_duplicate_username_with_409() {
        let (_tmp, router) = user_admin_test_router("root-token");
        let _ = post_create_user(
            &router,
            "root-token",
            serde_json::json!({"username": "alice"}),
        )
        .await;
        let resp = post_create_user(
            &router,
            "root-token",
            serde_json::json!({"username": "alice"}),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn create_user_rejects_invalid_email_with_400() {
        let (_tmp, router) = user_admin_test_router("root-token");
        let resp = post_create_user(
            &router,
            "root-token",
            serde_json::json!({"username": "alice", "email": "not-an-email"}),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn create_user_as_anonymous_returns_401() {
        let (_tmp, router) = user_admin_test_router("root-token");
        // No Authorization header → middleware stamps Anonymous tier.
        let resp = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/users")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&serde_json::json!({"username": "alice"})).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn create_user_as_user_tier_returns_403() {
        let (_tmp, router) = user_admin_test_router("root-token");
        let resp = post_create_user(
            &router,
            "not-the-root-token",
            serde_json::json!({"username": "alice"}),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn list_users_returns_added_users_in_insertion_order() {
        let (_tmp, router) = user_admin_test_router("root-token");
        for n in ["alice", "bob", "carol"] {
            let _ =
                post_create_user(&router, "root-token", serde_json::json!({"username": n})).await;
        }
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/admin/users")
                    .header("authorization", "Bearer root-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let users = json["users"].as_array().unwrap();
        assert_eq!(users.len(), 3);
        assert_eq!(users[0]["username"], "alice");
        assert_eq!(users[1]["username"], "bob");
        assert_eq!(users[2]["username"], "carol");
        // Tokens are NEVER surfaced by the list endpoint.
        for u in users {
            assert!(u.get("token").is_none(), "list must not leak tokens");
        }
    }

    #[tokio::test]
    async fn expire_then_revive_round_trips() {
        let (_tmp, router) = user_admin_test_router("root-token");
        let _ = post_create_user(
            &router,
            "root-token",
            serde_json::json!({"username": "alice"}),
        )
        .await;

        // Expire.
        let resp = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/users/alice/expire")
                    .header("authorization", "Bearer root-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["user"]["token_expired_at"].is_i64());

        // Revive.
        let resp = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/users/alice/revive")
                    .header("authorization", "Bearer root-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["user"]["token_expired_at"].is_null());
    }

    #[tokio::test]
    async fn expire_unknown_user_returns_404() {
        let (_tmp, router) = user_admin_test_router("root-token");
        let resp = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/users/ghost/expire")
                    .header("authorization", "Bearer root-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn rotate_token_issues_a_distinct_token() {
        let (_tmp, router) = user_admin_test_router("root-token");
        let create_resp = post_create_user(
            &router,
            "root-token",
            serde_json::json!({"username": "alice"}),
        )
        .await;
        let body = to_bytes(create_resp.into_body(), usize::MAX).await.unwrap();
        let original_token = serde_json::from_slice::<serde_json::Value>(&body).unwrap()["token"]
            .as_str()
            .unwrap()
            .to_string();

        let resp = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/users/alice/rotate-token")
                    .header("authorization", "Bearer root-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let new_token = json["token"].as_str().unwrap();
        assert_eq!(new_token.len(), 43);
        assert_ne!(new_token, original_token, "rotate must change the token");
    }

    #[tokio::test]
    async fn create_user_returns_503_when_pepper_not_configured() {
        // Same as user_admin_test_router but with token_pepper = None,
        // covering the "rung 1-only" backward-compat install.
        use ai_memory_core::ActorContext;
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();
        let router = admin_router(AdminState {
            writer: store.writer.clone(),
            reader: store.reader.clone(),
            wiki,
            llm: None,
            embedder: None,
            provider_health: ProviderHealth::default(),
            decay_params: DecayParams::default(),
            data_dir: tmp.path().to_path_buf(),
            db_path: store.db_path().to_path_buf(),
            bind: "127.0.0.1:49374".to_string(),
            bootstrap_lock: Arc::new(tokio::sync::Mutex::new(())),
            token_pepper: None,
        });
        // Inject a Root level so we're past the require_root gate;
        // the 503 must come from require_pepper.
        let router = router.layer(axum::middleware::from_fn(
            |mut req: Request<Body>, next: axum::middleware::Next| async move {
                req.extensions_mut().insert(ai_memory_core::AuthLevel::Root);
                req.extensions_mut().insert(ActorContext::anonymous());
                next.run(req).await
            },
        ));

        let resp = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/users")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&serde_json::json!({"username": "alice"})).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let _ = tmp;
    }
}

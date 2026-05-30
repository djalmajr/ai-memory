//! `ai-memory serve` — MCP server with optional filesystem watcher.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use ai_memory_consolidate::{Consolidator, run_lint, run_sweep};
use ai_memory_core::{ActiveProject, ProjectId, Sanitizer, WorkspaceId};
use ai_memory_hooks::{DEFAULT_HOOK_INGEST_MAX_IN_FLIGHT, HookState, hook_router};
use ai_memory_llm::{Embedder, LlmProvider, ProviderHealth, build_embedder, build_provider};
use ai_memory_mcp::{AdminState, AiMemoryServer, admin_router};
use ai_memory_store::{EmbeddingWrite, ReaderPool, Store, WriterHandle, f32_vec_to_bytes};
use ai_memory_web;
use ai_memory_wiki::{WatcherHandle, Wiki, migrations, run_wiki_migrations};
use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{Method, Request, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use rmcp::ServiceExt;
use rmcp::transport::stdio;
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use tokio_util::sync::CancellationToken;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};
use tracing::info;

use crate::auth::{AuthState, require_bearer};
use crate::cli::{ServeArgs, TransportKind};
use crate::config::{Config, MaintenanceSettings};

/// 10 MB cap on inbound HTTP bodies. The /hook ingress accepts the
/// agent's raw payload which can include a tool output excerpt
/// (capped at 2 KB on our side via `truncate_excerpt`), but Claude
/// Code et al. send the full envelope, which can run to a few KB.
/// 10 MB is generous headroom; without a cap, axum streams unbounded
/// bodies into memory (audit critical #2).
const MAX_BODY_BYTES: usize = 10 * 1024 * 1024;
const EMBEDDING_WRITE_BATCH: usize = 100;

/// `POST /admin/bootstrap` may carry a large JSON array of sources even
/// after client-side prune; keep hooks/MCP at [`MAX_BODY_BYTES`].
const BOOTSTRAP_MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

struct ConsolidatorSetup {
    server: AiMemoryServer,
    consolidator: Option<Arc<Consolidator>>,
    admin_llm: Option<Arc<dyn LlmProvider>>,
}

/// Run the `serve` subcommand.
///
/// # Errors
/// Returns an error if the store cannot be opened, the watcher cannot
/// install, or the transport setup fails.
pub async fn run(config: &Config, args: ServeArgs) -> Result<()> {
    validate_web_ui_args(args.enable_web, args.web_ui_dir.as_deref())?;

    // Merge config + CLI CORS origins (config first, CLI adds new entries).
    // Validation runs before binding so a misconfigured origin is caught early.
    let cors_origins = merge_cors_origins(&config.cors_allow_origins, &args.cors_allow_origin);
    validate_cors_origins(&cors_origins)?;

    let store = Store::open(&config.data_dir)
        .with_context(|| format!("opening store at {}", config.data_dir.display()))?;

    // Run any outstanding wiki-structure migrations before the watcher starts
    // so file moves and renames are never raced by the reconciler.
    let wiki_root = config.data_dir.join("wiki");
    run_wiki_migrations(
        &store.writer,
        &store.reader,
        &wiki_root,
        &migrations::registry(),
    )
    .await
    .with_context(|| "applying wiki-structure migrations")?;

    let ws = store
        .writer
        .get_or_create_workspace(args.workspace.clone())
        .await?;
    let proj = store
        .writer
        .get_or_create_project(ws, args.project.clone(), None)
        .await?;
    // Build the privacy strip from config. Compile errors in
    // user-supplied regex abort startup with a clear message so
    // operators discover misconfiguration immediately.
    let sanitizer = Sanitizer::new(&config.sanitize)
        .context("compiling sanitizer.extra_patterns from config")?;
    let wiki = Wiki::new(&config.data_dir, store.writer.clone())?.with_sanitizer(sanitizer.clone());
    // Attach the admission webhook chain (operator-configured via
    // `[[admission_webhooks]]` in config.toml or `AI_MEMORY_ADMISSION_WEBHOOKS__N__*`
    // env vars). Empty config = no chain attached, zero overhead. The store
    // reader is forwarded so the chain can resolve workspace_id/project_id
    // into the human names webhooks address pages by.
    let wiki = if config.admission_webhooks.is_empty() {
        wiki
    } else {
        let chain = ai_memory_wiki::AdmissionChain::new(config.admission_webhooks.clone())
            .context("building admission webhook chain")?;
        tracing::info!(
            count = config.admission_webhooks.len(),
            "admission webhook chain attached"
        );
        wiki.with_admission_chain(chain)
            .with_store_reader(store.reader.clone())
    };
    let provider_health = ProviderHealth::default();
    let (wiki, embedder) = configure_embedder(config, &store, wiki, &provider_health).await?;

    // Keep the guard alive for the lifetime of `serve`.
    let _watcher = start_watcher(&args, &wiki)?;

    // Shared between the MCP server and the hook router: the hook
    // router publishes the cwd-resolved project on each event; the MCP
    // read tools read it as their default so a shared HTTP server
    // answers for the project the agent is actually in, not the static
    // `--project` (issue #2). In stdio mode no hook router is built, so
    // this stays empty and the baked-in default is used.
    let active_project = ActiveProject::new();
    let mut server = AiMemoryServer::new(store.reader.clone(), store.writer.clone(), ws, proj)
        .with_wiki(wiki.clone())
        .with_decay_params(config.decay)
        .with_active_project(active_project.clone())
        .with_sanitizer(sanitizer.clone());
    if let Some(e) = embedder.clone() {
        server = server.with_embedder(e);
    }
    let consolidator_setup =
        configure_consolidator(config, server, &store, &wiki, ws, proj, &provider_health)?;
    let server = consolidator_setup.server;
    let consolidator = consolidator_setup.consolidator;
    let admin_llm = consolidator_setup.admin_llm;
    let _maintenance_tasks = start_maintenance_scheduler(
        config.maintenance.clone(),
        store.reader.clone(),
        store.writer.clone(),
        wiki.clone(),
        embedder.clone(),
        admin_llm.clone(),
        ws,
        proj,
        config.decay,
    );

    match args.transport {
        TransportKind::Stdio => {
            info!("MCP server ready on stdio (Ctrl-C to stop)");
            let service = server.serve(stdio()).await?;
            service.waiting().await?;
        }
        TransportKind::Http => {
            let bind = args.bind.unwrap_or_else(|| config.bind.clone());
            let cancel = CancellationToken::new();
            let server_clone = server.clone();
            // `Host`-header allowlist for the HTTP DNS-rebinding guard.
            // Sourced from Config (which already handles the
            // `AI_MEMORY_ALLOWED_HOSTS=a,b,c` env-string vs.
            // config.toml sequence forms via the string-or-vec
            // deserializer). Logged so operators can verify the
            // effective list against what they intended.
            info!(
                allowed_hosts = ?config.allowed_hosts,
                "HTTP Host-header allowlist"
            );
            // Default to stateless Streamable HTTP: each POST is serviced
            // independently and answered as plain `application/json`, so
            // stateless clients (OpenCode `type: "remote"`, curl) work
            // without an `mcp-remote` shim (issue #3). ai-memory's tools
            // are pure request-response and project resolution rides the
            // in-process `ActiveProject` pointer, not the transport
            // session — so session mode buys us nothing. `--http-stateful`
            // restores rmcp's session+SSE behaviour for clients that want
            // it.
            info!(
                stateful = args.http_stateful,
                "MCP Streamable HTTP transport mode"
            );
            let mcp_service = StreamableHttpService::new(
                move || Ok(server_clone.clone()),
                LocalSessionManager::default().into(),
                StreamableHttpServerConfig::default()
                    .with_cancellation_token(cancel.child_token())
                    .with_allowed_hosts(config.allowed_hosts.clone())
                    .with_stateful_mode(args.http_stateful)
                    .with_json_response(!args.http_stateful),
            );
            let hooks = hook_router(HookState {
                workspace_id: ws,
                project_id: proj,
                writer: store.writer.clone(),
                reader: store.reader.clone(),
                wiki: wiki.clone(),
                consolidator: consolidator.clone(),
                sanitizer: sanitizer.clone(),
                project_cache: std::sync::Arc::new(tokio::sync::Mutex::new(
                    std::collections::HashMap::new(),
                )),
                active_project: active_project.clone(),
                ingest_semaphore: std::sync::Arc::new(tokio::sync::Semaphore::new(
                    DEFAULT_HOOK_INGEST_MAX_IN_FLIGHT,
                )),
                consolidate_on_session_end: config.consolidate_on_session_end,
            });
            let admin = admin_router(AdminState {
                writer: store.writer.clone(),
                reader: store.reader.clone(),
                wiki: wiki.clone(),
                llm: admin_llm,
                embedder: embedder.clone(),
                provider_health: provider_health.clone(),
                decay_params: config.decay,
                data_dir: config.data_dir.clone(),
                db_path: store.db_path().to_path_buf(),
                bind: bind.clone(),
                bootstrap_lock: std::sync::Arc::new(tokio::sync::Mutex::new(())),
                token_pepper: config
                    .auth
                    .token_pepper
                    .as_ref()
                    .filter(|p| !p.trim().is_empty())
                    .map(|p| ai_memory_store::TokenPepper::new(p.clone())),
            });
            // Multi-rung auth assembly:
            //   - rung 0 (no bearer_token configured) → AuthState::new
            //     stays as-is, middleware injects anonymous actor.
            //   - rung 1 (bearer_token set, no token_pepper) → root_actor
            //     stamps writes with [auth].root_* identity.
            //   - rung 2 (bearer_token + token_pepper + users in DB) →
            //     unknown bearer routes through users-table lookup.
            // The pepper is auto-generated by `ai-memory init`, so almost
            // every operator-installed server reaches rung 2; the only
            // rung-1-only setups are those whose config predates v0.8.
            let mut auth_state = AuthState::new(config.auth.bearer_token.clone());
            let root_user = config.auth.root_username.clone();
            if root_user.as_deref().is_some_and(|s| !s.trim().is_empty()) {
                auth_state = auth_state.with_root_actor(ai_memory_core::ActorContext {
                    user: root_user,
                    name: config.auth.root_name.clone(),
                    email: config.auth.root_email.clone(),
                    ..ai_memory_core::ActorContext::default()
                });
            }
            if let Some(pepper) = config
                .auth
                .token_pepper
                .as_ref()
                .filter(|p| !p.trim().is_empty())
            {
                auth_state = auth_state.with_multiuser(
                    ai_memory_store::TokenPepper::new(pepper.clone()),
                    store.reader.clone(),
                    store.writer.clone(),
                );
            }
            let auth_state = Arc::new(auth_state);
            let auth_enabled = auth_state.enabled();
            let router = axum::Router::new()
                .nest_service("/mcp", mcp_service)
                .merge(hooks)
                .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
                .merge(admin.layer(DefaultBodyLimit::max(BOOTSTRAP_MAX_BODY_BYTES)));
            let router = mount_web_router(
                router,
                args.enable_web,
                store.reader.clone(),
                wiki.clone(),
                args.web_ui_dir.as_deref(),
                &cors_origins,
            );
            let router = apply_http_layers(router, auth_state, config.allowed_hosts.clone());
            let listener = tokio::net::TcpListener::bind(&bind)
                .await
                .with_context(|| format!("binding {bind}"))?;
            info!(
                %bind,
                auth = auth_enabled,
                body_limit_mb = MAX_BODY_BYTES / 1024 / 1024,
                "MCP HTTP server ready (POST /mcp, POST /hook, Ctrl-C to stop)",
            );
            if !auth_enabled && !bind.starts_with("127.") {
                // Loud warning: a non-loopback bind with no auth is
                // the audit's critical-#1 scenario. The operator gets
                // a one-line "you sure?" instead of silent exposure.
                tracing::warn!(
                    %bind,
                    "no AI_MEMORY_AUTH_TOKEN configured AND binding to a non-loopback \
                     address — anyone on the network can call destructive MCP tools. \
                     Generate a token with `ai-memory generate-auth-token` and set \
                     AI_MEMORY_AUTH_TOKEN in the server's environment."
                );
            } else if auth_enabled && !bind.starts_with("127.") {
                // Auth IS configured but the server is reachable from
                // the network on plain HTTP. The bearer token (and
                // multi-user per-user tokens from `ai-memory user
                // add`) ride cleartext — sniffable on the LAN. Advise
                // the operator to front with a TLS proxy. One-shot
                // log at startup, not refusal to serve (operators may
                // be testing, behind their own proxy already, etc.).
                tracing::warn!(
                    %bind,
                    "AI_MEMORY_AUTH_TOKEN is set but the server is bound to a \
                     non-loopback address on plain HTTP — bearer tokens travel \
                     cleartext on the network. Front ai-memory with a TLS-terminating \
                     reverse proxy (Caddy, Cloudflare Tunnel, nginx). See \
                     docs/https-via-proxy.md for copy-paste templates."
                );
            }
            axum::serve(listener, router)
                .with_graceful_shutdown(async move {
                    let _ = tokio::signal::ctrl_c().await;
                    info!("ctrl-c received; shutting down");
                    cancel.cancel();
                })
                .await?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn start_maintenance_scheduler(
    settings: MaintenanceSettings,
    reader: ReaderPool,
    writer: WriterHandle,
    wiki: Wiki,
    embedder: Option<Arc<dyn Embedder>>,
    llm: Option<Arc<dyn LlmProvider>>,
    workspace_id: WorkspaceId,
    project_id: ProjectId,
    decay: ai_memory_store::DecayParams,
) -> Vec<tokio::task::JoinHandle<()>> {
    if !settings.enabled {
        info!("scheduled maintenance disabled");
        return Vec::new();
    }

    let forget_sweep_interval_secs = settings.forget_sweep_interval_secs;
    let lint_interval_secs = settings.lint_interval_secs;
    let embedding_backfill_interval_secs = settings.embedding_backfill_interval_secs;

    let mut tasks = Vec::new();
    if forget_sweep_interval_secs > 0 {
        let reader = reader.clone();
        let writer = writer.clone();
        tasks.push(tokio::spawn(async move {
            let interval = std::time::Duration::from_secs(forget_sweep_interval_secs);
            loop {
                tokio::time::sleep(interval).await;
                match run_sweep(&reader, &writer, workspace_id, project_id, &decay, false).await {
                    Ok(report) => info!(
                        evicted = report.evicted.len(),
                        hard_deleted = report.hard_deleted,
                        "scheduled forget sweep completed"
                    ),
                    Err(e) => tracing::warn!(error = %e, "scheduled forget sweep failed"),
                }
            }
        }));
    }

    if lint_interval_secs > 0 {
        let reader = reader.clone();
        let wiki = wiki.clone();
        let llm = llm.clone();
        tasks.push(tokio::spawn(async move {
            let interval = std::time::Duration::from_secs(lint_interval_secs);
            loop {
                tokio::time::sleep(interval).await;
                match run_lint(
                    &reader,
                    &wiki,
                    llm.as_ref(),
                    workspace_id,
                    project_id,
                    false,
                    false,
                )
                .await
                {
                    Ok(report) => info!(
                        findings = report.findings.len(),
                        "scheduled rule-based lint completed"
                    ),
                    Err(e) => tracing::warn!(error = %e, "scheduled lint failed"),
                }
            }
        }));
    }

    if embedding_backfill_interval_secs > 0 {
        if let Some(embedder) = embedder {
            let reader = reader.clone();
            let writer = writer.clone();
            let wiki = wiki.clone();
            tasks.push(tokio::spawn(async move {
                let interval = std::time::Duration::from_secs(embedding_backfill_interval_secs);
                loop {
                    tokio::time::sleep(interval).await;
                    match run_embedding_backfill(
                        &reader,
                        &writer,
                        &wiki,
                        &embedder,
                        workspace_id,
                        project_id,
                    )
                    .await
                    {
                        Ok((embedded, failed)) => {
                            info!(embedded, failed, "scheduled embedding backfill completed")
                        }
                        Err(e) => tracing::warn!(error = %e, "scheduled embedding backfill failed"),
                    }
                }
            }));
        } else {
            tracing::warn!(
                "maintenance.embedding_backfill_interval_secs is set but no embedder is configured"
            );
        }
    }

    if tasks.is_empty() {
        info!("scheduled maintenance enabled but all intervals are disabled");
    } else {
        info!(jobs = tasks.len(), "scheduled maintenance started");
    }
    tasks
}

async fn run_embedding_backfill(
    reader: &ReaderPool,
    writer: &WriterHandle,
    wiki: &Wiki,
    embedder: &Arc<dyn Embedder>,
    workspace_id: WorkspaceId,
    project_id: ProjectId,
) -> Result<(usize, usize)> {
    let provider = embedder.provider().to_string();
    let model = embedder.model().to_string();
    let dim = embedder.dim();
    let candidates = reader.decay_candidates(workspace_id, project_id).await?;
    let already: std::collections::HashSet<_> = reader
        .embedded_page_ids(
            workspace_id,
            project_id,
            provider.clone(),
            model.clone(),
            dim,
        )
        .await?
        .into_iter()
        .collect();

    let mut embedded = 0usize;
    let mut failed = 0usize;
    let mut pending = Vec::with_capacity(EMBEDDING_WRITE_BATCH);
    for cand in candidates {
        if already.contains(&cand.id) {
            continue;
        }
        let md = match wiki.read_page(workspace_id, project_id, &cand.path) {
            Ok(md) => md,
            Err(e) => {
                failed += 1;
                tracing::warn!(path = %cand.path, error = %e, "scheduled embed: unreadable page");
                continue;
            }
        };
        let vec = match embedder.embed_document(&md.body).await {
            Ok(vec) => vec,
            Err(e) => {
                failed += 1;
                tracing::warn!(path = %cand.path, error = %e, "scheduled embed: provider failed");
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
            flush_embedding_batch(writer, &mut pending, &mut embedded, &mut failed).await;
        }
    }
    flush_embedding_batch(writer, &mut pending, &mut embedded, &mut failed).await;
    Ok((embedded, failed))
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
        tracing::warn!(count, error = %e, "scheduled embed: batch store failed");
    } else {
        *embedded += count;
    }
}

async fn configure_embedder(
    config: &Config,
    store: &Store,
    wiki: Wiki,
    provider_health: &ProviderHealth,
) -> Result<(Wiki, Option<Arc<dyn Embedder>>)> {
    // M9 — pluggable embedder. Stored rows carry provider/model/dim so
    // query paths can ignore stale vectors after an embedding config change.
    let Some(cfg) = config.embedder_config()? else {
        info!("AI_MEMORY_EMBEDDING_PROVIDER unset; hybrid search disabled (FTS5-only)");
        return Ok((wiki, None));
    };
    let provider_name = cfg.provider.name().to_string();
    let model = cfg.model.clone();
    let dim = cfg.dim;
    let embedder = build_embedder(cfg).context("building embedder from config")?;
    let mismatch = store
        .reader
        .embedding_meta_for_mismatch(
            embedder.provider().into(),
            embedder.model().into(),
            embedder.dim(),
        )
        .await?;
    if !mismatch.is_empty() {
        // Mismatch handling applies to hybrid search (queries only load
        // rows matching the configured triple), not to process liveness.
        // Blocking startup made `embed --force` impossible because the
        // CLI is an HTTP client to this server.
        tracing::warn!(
            stored = ?mismatch,
            configured_provider = embedder.provider(),
            configured_model = embedder.model(),
            configured_dim = embedder.dim(),
            "stored embeddings use a different (provider, model, dim) than configured; \
             hybrid search ignores stale rows until pages are re-embedded — \
             run `ai-memory embed --force` (or wait for scheduled backfill)"
        );
    }
    info!(
        provider = embedder.provider(),
        model = embedder.model(),
        dim = embedder.dim(),
        "embedder enabled"
    );
    let embedder = provider_health.wrap_embedder(embedder, provider_name, model, dim);
    Ok((wiki.with_embedder(embedder.clone()), Some(embedder)))
}

fn start_watcher(args: &ServeArgs, wiki: &Wiki) -> Result<Option<WatcherHandle>> {
    if args.no_watcher {
        info!("watcher disabled by --no-watcher");
        return Ok(None);
    }
    info!(
        root = %wiki.root().display(),
        workspace = %args.workspace,
        project = %args.project,
        "starting wiki watcher",
    );
    Ok(Some(WatcherHandle::start(wiki.clone())?))
}

fn configure_consolidator(
    config: &Config,
    mut server: AiMemoryServer,
    store: &Store,
    wiki: &Wiki,
    workspace_id: WorkspaceId,
    project_id: ProjectId,
    provider_health: &ProviderHealth,
) -> Result<ConsolidatorSetup> {
    // Build the consolidator (if LLM configured) once, then share the
    // Arc between the MCP server (for `memory_consolidate` + lint),
    // the hook router (for PreCompact checkpointing), and the admin
    // router (for `POST /admin/bootstrap`).
    let Some(cfg) = config.llm_provider_config()? else {
        info!(
            "AI_MEMORY_LLM_PROVIDER unset; memory_consolidate disabled, PreCompact \
             falls back to rule-based checkpoint, lint runs rule-based only"
        );
        return Ok(ConsolidatorSetup {
            server,
            consolidator: None,
            admin_llm: None,
        });
    };
    let provider_name = cfg.provider.name().to_string();
    let model = cfg.model.clone();
    let retry_hint = llm_retry_hint(&provider_name, &model, cfg.base_url.as_deref());
    let llm = build_provider(cfg).context("building LLM provider from config")?;
    let llm = provider_health.wrap_llm_provider(llm, provider_name, model, Some(retry_hint));
    info!(
        provider = llm.name(),
        model = llm.model(),
        "memory_consolidate + PreCompact LLM checkpointing enabled",
    );
    let consolidator = Arc::new(Consolidator::new(
        store.reader.clone(),
        store.writer.clone(),
        wiki.clone(),
        llm.clone(),
        workspace_id,
        project_id,
    ));
    server = server.with_consolidator_arc(wiki.clone(), llm.clone(), consolidator.clone());
    Ok(ConsolidatorSetup {
        server,
        consolidator: Some(consolidator),
        admin_llm: Some(llm),
    })
}

/// Validate a list of CORS origins before the server binds.
///
/// Rules match the spec: wildcard + credentials is forbidden, each entry
/// must carry a scheme, and trailing slashes are rejected (they do not
/// match browser origins which never carry a trailing slash).
pub fn validate_cors_origins(origins: &[String]) -> Result<()> {
    for origin in origins {
        if origin == "*" {
            anyhow::bail!(
                "CORS origin `*` is not allowed: the CORS spec forbids credentials \
                 with a wildcard origin. Use explicit origins such as \
                 https://app.example.com"
            );
        }
        if !origin.starts_with("http://") && !origin.starts_with("https://") {
            anyhow::bail!(
                "CORS origin `{origin}` is missing a scheme. Each entry must start \
                 with http:// or https://"
            );
        }
        if origin.ends_with('/') {
            anyhow::bail!(
                "CORS origin `{origin}` has a trailing slash. Browser origins \
                 never carry a trailing slash — use `{}` instead",
                origin.trim_end_matches('/')
            );
        }
    }
    Ok(())
}

/// Merge config-file origins with CLI flag origins, preserving order and
/// deduplicating (config entries appear first).
fn merge_cors_origins(from_config: &[String], from_cli: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut merged = Vec::new();
    for origin in from_config.iter().chain(from_cli.iter()) {
        if seen.insert(origin.clone()) {
            merged.push(origin.clone());
        }
    }
    merged
}

fn validate_web_ui_args(enable_web: bool, web_ui_dir: Option<&Path>) -> Result<()> {
    if web_ui_dir.is_some() && !enable_web {
        anyhow::bail!("--web-ui-dir requires --enable-web");
    }

    if let Some(dir) = web_ui_dir {
        if !dir.is_dir() {
            anyhow::bail!("--web-ui-dir is not a directory: {}", dir.display());
        }
        if !dir.join("index.html").is_file() {
            anyhow::bail!("--web-ui-dir is missing index.html: {}", dir.display());
        }
    }

    Ok(())
}

fn llm_retry_hint(provider: &str, model: &str, base_url: Option<&str>) -> String {
    let mut command = format!("ai-memory llm-test --provider {provider} --model {model}");
    if let Some(base_url) = base_url {
        command.push_str(&format!(" --base-url {base_url}"));
    }
    command.push_str(" --prompt ping");
    command
}

fn mount_web_router(
    router: axum::Router,
    enable_web: bool,
    reader: ReaderPool,
    wiki: Wiki,
    web_ui_dir: Option<&Path>,
    cors_origins: &[String],
) -> axum::Router {
    if !enable_web {
        return router;
    }
    // Register the web surfaces BEFORE applying the bearer middleware. In
    // axum 0.8, `.layer()` only attaches to routes registered before the
    // call; nesting after the layer would silently bypass auth for /web/*.

    // Build the api router and optionally wrap it in a CorsLayer — the layer
    // is scoped ONLY to /api/v1 so /mcp, /hook, /admin, and /web remain
    // untouched (CORS_NOT_APPLIED_TO_OTHER_ROUTES invariant).
    let api = ai_memory_web::api_router(reader.clone(), wiki.clone());
    let api = if cors_origins.is_empty() {
        api
    } else {
        // Origins were already validated before binding, so parsing here
        // is expected to succeed; `.expect` surfaces a logic bug if it does not.
        let parsed: Vec<axum::http::HeaderValue> = cors_origins
            .iter()
            .map(|o| o.parse().expect("pre-validated origin must parse"))
            .collect();
        let cors = CorsLayer::new()
            .allow_origin(parsed)
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE])
            .allow_credentials(true)
            .max_age(Duration::from_secs(600));
        info!(origins = ?cors_origins, "CORS layer attached to /api/v1");
        api.layer(cors)
    };
    let router = router.nest("/api/v1", api);

    // Custom SPA via --web-ui-dir (SPA fallback to index.html), otherwise
    // the built-in server-side wiki browser.
    if let Some(dir) = web_ui_dir {
        let dir = dir.to_path_buf();
        let index = dir.join("index.html");
        info!(dir = %dir.display(), "custom web UI mounted at /web");
        // ServeDir already answers /web and /web/ (SPA fallback to index.html);
        // an explicit /web/ route here would conflict with the nest_service.
        return router.nest_service("/web", ServeDir::new(dir).fallback(ServeFile::new(index)));
    }

    let web_router = ai_memory_web::router(reader, wiki);
    info!("read-only wiki browser mounted at /web");
    router
        .route(
            "/web/",
            axum::routing::get(|| async { axum::response::Redirect::permanent("/web") }),
        )
        .nest("/web", web_router)
}

fn apply_http_layers(
    router: axum::Router,
    auth_state: Arc<AuthState>,
    allowed_hosts: Vec<String>,
) -> axum::Router {
    router
        .layer(axum::middleware::from_fn_with_state(
            auth_state,
            require_bearer,
        ))
        .layer(axum::middleware::from_fn_with_state(
            Arc::new(allowed_hosts),
            require_allowed_host,
        ))
}

async fn require_allowed_host(
    State(allowed_hosts): State<Arc<Vec<String>>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let Some(host) = req
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
    else {
        return (StatusCode::BAD_REQUEST, "missing Host header\n").into_response();
    };
    if host_allowed(host, &allowed_hosts) {
        return next.run(req).await;
    }
    tracing::warn!(host, allowed = ?allowed_hosts, "rejected request with disallowed Host header");
    (StatusCode::FORBIDDEN, "forbidden host\n").into_response()
}

fn host_allowed(host: &str, allowed_hosts: &[String]) -> bool {
    allowed_hosts.iter().any(|allowed| {
        host.eq_ignore_ascii_case(allowed) || host_without_port(host).eq_ignore_ascii_case(allowed)
    })
}

fn host_without_port(host: &str) -> &str {
    if let Some(rest) = host.strip_prefix('[')
        && let Some((inside, _)) = rest.split_once(']')
    {
        return inside;
    }
    match host.rsplit_once(':') {
        Some((name, port)) if !name.contains(':') && port.chars().all(|c| c.is_ascii_digit()) => {
            name
        }
        _ => host,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_memory_core::{PagePath, Tier};
    use ai_memory_llm::SyntheticEmbedder;
    use ai_memory_wiki::WritePageRequest;
    use axum::http::Request;
    use tempfile::TempDir;
    use tower::ServiceExt;

    #[test]
    fn host_allowed_accepts_host_with_port() {
        let allowed = vec!["127.0.0.1".to_string(), "localhost".to_string()];
        assert!(host_allowed("127.0.0.1:49374", &allowed));
        assert!(host_allowed("localhost", &allowed));
    }

    #[test]
    fn host_allowed_rejects_unknown_host() {
        let allowed = vec!["127.0.0.1".to_string()];
        assert!(!host_allowed("evil.example:49374", &allowed));
    }

    #[test]
    fn host_without_port_handles_ipv6_loopback() {
        assert_eq!(host_without_port("[::1]:49374"), "::1");
    }

    #[test]
    fn web_ui_dir_requires_enable_web() {
        let ui = TempDir::new().unwrap();
        std::fs::write(ui.path().join("index.html"), "custom ui").unwrap();

        let err = validate_web_ui_args(false, Some(ui.path())).unwrap_err();
        assert!(
            err.to_string()
                .contains("--web-ui-dir requires --enable-web"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn web_ui_dir_must_be_directory() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("index.html");
        std::fs::write(&file, "custom ui").unwrap();

        let err = validate_web_ui_args(true, Some(&file)).unwrap_err();
        assert!(
            err.to_string().contains("--web-ui-dir is not a directory"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn web_ui_dir_must_include_index_html() {
        let ui = TempDir::new().unwrap();

        let err = validate_web_ui_args(true, Some(ui.path())).unwrap_err();
        assert!(
            err.to_string()
                .contains("--web-ui-dir is missing index.html"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn valid_web_ui_dir_passes_validation() {
        let ui = TempDir::new().unwrap();
        std::fs::write(ui.path().join("index.html"), "custom ui").unwrap();

        validate_web_ui_args(true, Some(ui.path())).unwrap();
    }

    #[tokio::test]
    async fn web_routes_are_inside_auth_layer() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();
        let router = mount_web_router(
            axum::Router::new(),
            true,
            store.reader.clone(),
            wiki,
            None,
            &[],
        );
        let router = apply_http_layers(
            router,
            Arc::new(AuthState::new(Some("secret".to_string()))),
            vec!["localhost".to_string()],
        );

        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/web")
                    .header("Host", "localhost")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn embedder_mismatch_warns_but_keeps_server_startable() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let ws = store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();
        let proj = store
            .writer
            .get_or_create_project(ws, "scratch", None)
            .await
            .unwrap();

        let synthetic: Arc<dyn Embedder> = Arc::new(SyntheticEmbedder::new(64));
        let wiki = Wiki::new(tmp.path(), store.writer.clone())
            .unwrap()
            .with_embedder(synthetic);
        wiki.write_page(WritePageRequest {
            workspace_id: ws,
            project_id: proj,
            path: PagePath::new("notes/old-embedding.md").unwrap(),
            frontmatter: serde_json::json!({"title": "old embedding"}),
            body: "existing vector row".into(),
            tier: Tier::Semantic,
            pinned: false,
            title: None,
            admission_ctx: None,
            author_id: None,
            actor: ai_memory_core::ActorContext::anonymous(),
        })
        .await
        .unwrap();

        let cfg = Config {
            data_dir: tmp.path().to_path_buf(),
            embedding_provider: Some("openai".into()),
            runtime_env: crate::config::RuntimeEnv::with_openai_api_key_for_tests("test-key"),
            ..Config::default()
        };
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();

        let provider_health = ProviderHealth::default();
        let (_wiki, embedder) = configure_embedder(&cfg, &store, wiki, &provider_health)
            .await
            .unwrap();

        let embedder = embedder.expect("configured embedder should be enabled");
        assert_eq!(embedder.provider(), "openai");
        assert_eq!(
            provider_health.snapshot().embedding.status,
            ai_memory_llm::ProviderHealthStatus::Unknown
        );
    }

    // ── Part B: CORS validation tests ──────────────────────────────────────

    #[test]
    fn validate_cors_origins_rejects_wildcard() {
        let err = validate_cors_origins(&["*".to_string()]).unwrap_err();
        assert!(
            err.to_string().contains("wildcard"),
            "error must mention wildcard: {err}"
        );
    }

    #[test]
    fn validate_cors_origins_rejects_missing_scheme() {
        let err = validate_cors_origins(&["app.example.com".to_string()]).unwrap_err();
        assert!(
            err.to_string().contains("missing a scheme"),
            "error must mention missing scheme: {err}"
        );
    }

    #[test]
    fn validate_cors_origins_rejects_trailing_slash() {
        let err = validate_cors_origins(&["https://app.example.com/".to_string()]).unwrap_err();
        assert!(
            err.to_string().contains("trailing slash"),
            "error must mention trailing slash: {err}"
        );
    }

    #[test]
    fn validate_cors_origins_accepts_well_formed() {
        validate_cors_origins(&[
            "https://app.example.com".to_string(),
            "http://localhost:5173".to_string(),
        ])
        .unwrap();
    }

    #[test]
    fn validate_cors_origins_accepts_empty_list() {
        validate_cors_origins(&[]).unwrap();
    }

    #[test]
    fn merge_cors_origins_deduplicates_preserving_order() {
        let merged = merge_cors_origins(
            &[
                "https://a.example.com".to_string(),
                "https://b.example.com".to_string(),
            ],
            &[
                "https://b.example.com".to_string(),
                "https://c.example.com".to_string(),
            ],
        );
        assert_eq!(
            merged,
            vec![
                "https://a.example.com",
                "https://b.example.com",
                "https://c.example.com"
            ]
        );
    }

    #[tokio::test]
    async fn cors_layer_on_api_v1_allows_configured_origin() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();
        store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();

        let cors_origins = ["https://app.example.com".to_string()];
        let router = mount_web_router(
            axum::Router::new(),
            true,
            store.reader.clone(),
            wiki,
            None,
            &cors_origins,
        );
        // No auth layer so we can reach /api/v1 directly.
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/api/v1/workspaces")
                    .header("Origin", "https://app.example.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let acao = resp
            .headers()
            .get("access-control-allow-origin")
            .expect("ACAO header must be present for allowed origin")
            .to_str()
            .unwrap();
        assert_eq!(acao, "https://app.example.com");
        let acac = resp
            .headers()
            .get("access-control-allow-credentials")
            .expect("ACAC header must be present")
            .to_str()
            .unwrap();
        assert_eq!(acac, "true");
    }

    #[tokio::test]
    async fn cors_layer_on_api_v1_denies_unlisted_origin() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();
        store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();

        let cors_origins = ["https://app.example.com".to_string()];
        let router = mount_web_router(
            axum::Router::new(),
            true,
            store.reader.clone(),
            wiki,
            None,
            &cors_origins,
        );
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/api/v1/workspaces")
                    .header("Origin", "https://evil.example.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // The request is still served (CORS does not block on the server side),
        // but the ACAO header must be absent so the browser enforces the policy.
        assert!(
            resp.headers().get("access-control-allow-origin").is_none(),
            "unlisted origin must not receive ACAO header"
        );
    }

    #[tokio::test]
    async fn cors_not_applied_to_other_routes() {
        // /mcp and /admin routes must not carry CORS headers even when
        // a CORS origin list is configured (CORS_NOT_APPLIED_TO_OTHER_ROUTES
        // invariant). We verify by checking that a request to a non-/api/v1
        // path that 404s (no actual handler mounted here) still lacks ACAO.
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();

        let cors_origins = ["https://app.example.com".to_string()];
        let router = mount_web_router(
            axum::Router::new(),
            true,
            store.reader.clone(),
            wiki,
            None,
            &cors_origins,
        );
        // /web is a non-api route; sending an Origin header must not trigger CORS.
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/web")
                    .header("Origin", "https://app.example.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert!(
            resp.headers().get("access-control-allow-origin").is_none(),
            "/web must not carry CORS headers: {:?}",
            resp.headers()
        );
    }
}

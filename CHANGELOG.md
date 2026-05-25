# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Oh My Pi / OMP is now a first-class integration: `install-mcp --client pi`
  and `--client omp` write native `~/.omp/agent/mcp.json` config, while
  `install-hooks --agent omp` and `--agent pi` write the TypeScript extension
  used for lifecycle capture and handoff injection.
- Graph-aware retrieval: `memory_query` now combines FTS5, wikilink-neighbor
  expansion, optional vector RRF, and bounded raw-observation fallback.
- Observation FTS indexing and unresolved-link diagnostics surfaced through
  admin/CLI status paths.
- `_slots/` wiki pages are automatically pinned and surfaced in briefing /
  explore snapshots.
- Server-side scheduled maintenance for forget sweep and lint, with optional
  embedding backfill scheduling.
- Experimental native Windows support: PowerShell Docker wrapper,
  `ai-memory.cmd`, `.ps1` lifecycle hooks in parity with `.sh` hooks, Windows
  Tailwind hash/download support, and [`docs/windows.md`](docs/windows.md).
- Google Gemini LLM provider via `AI_MEMORY_LLM_PROVIDER=gemini`, with
  `gemini-2.5-flash` as the default hosted Google model and `GEMINI_API_KEY`
  / `GOOGLE_API_KEY` support.
- README support matrix for operating systems, agent integrations, LLM
  providers, and embedding providers.

### Changed
- Same-body page upserts are now true no-ops, avoiding periodic watcher
  reconcile writes, FTS churn, and misleading recent-page timestamps.
- Graph-neighbor expansion for hybrid search now batches all seed pages into
  one SQL query instead of issuing incoming/outgoing lookups per seed.
- Embedding backfill stores embeddings in chunks instead of one writer
  command and SQLite transaction per page.
- Hook ingestion now bounds in-flight processing and returns HTTP 429 when
  saturated instead of spawning unbounded background tasks.
- Documented the vector backend policy and the measured criteria required
  before adding `sqlite-vec`.
- Clarified Gemini CLI support docs: MCP registration, lifecycle hooks,
  SessionStart handoff injection, and SessionEnd capture are now called out
  consistently across README and install guides.
- Added OpenClaw lifecycle support via a generated native plugin package and
  updated Cursor / Claude Desktop / OpenClaw support docs against current
  upstream MCP and hook documentation.
- Docker images now bundle both POSIX and PowerShell hook scripts.

### Fixed
- `backup --to` now streams the tarball to disk instead of buffering the full
  archive in CLI memory.
- Hyphenated FTS5 queries such as `ai-memory` are normalized safely instead of
  being parsed as column operators.
- Gemini 2.5 Flash requests disable default dynamic thinking so hidden thought
  tokens do not consume `maxOutputTokens` and truncate strict JSON responses.
- `install-mcp --client claude-code` now prints the direct-edit JSON path as
  `~/.claude.json`, matching the `--apply` path and `claude mcp add` behavior.

## [0.1.3] - 2026-05-24

### Added
- `ai-memory lint --no-llm` (and `memory_lint` `no_llm` arg) to run only the
  rule-based lint pass while leaving the LLM enabled for `memory_explore` /
  `memory_consolidate` ([#4]).

### Fixed
- `memory_lint` LLM contradiction pass silently never contributed: the
  `LintFinding` struct expected `severity`/`message` but the prompt asked for
  `summary`/`detail`. The prompt is now aligned to the canonical shape and the
  struct tolerates both (defaults `severity`, aliases `summary`→`message`,
  captures optional `detail`) ([#4]).
- Reasoning models (MiniMax M2.7, DeepSeek, Qwen, Kimi) that emit
  `<think>…</think>` / `<analysis>…</analysis>` blocks before the JSON broke
  structured-output parsing (`key must be a string at line 1 column 2`). The
  openai-compat provider now strips reasoning blocks and surrounding markdown
  fences before extracting the JSON object, so lint / consolidate / bootstrap
  work with reasoning models ([#5]).
- openai-compat base URLs with non-`v1` version segments (e.g. Z.AI's `/v4`)
  or a full endpoint path no longer produce `…/v1/v1/…` 404s
  ([#6], thanks @lucasliet).

## [0.1.2] - 2026-05-24

### Changed
- HTTP transport now defaults to **stateless** mode (`json_response`, no
  `Mcp-Session-Id` required), so stateless MCP clients (OpenCode
  `type: "remote"`, `curl`) work without an `mcp-remote` stdio shim
  ([#3]). New `serve --transport http --http-stateful` flag restores the
  previous session+SSE behaviour for clients that need it.

## [0.1.1] - 2026-05-24

### Added
- Wiki-structure migration framework: `wiki_migrations` SQL table (V06),
  `WikiMigration` trait, migration registry, and `run_pending` runner
  invoked at server startup before the watcher starts.
- MCP read tools (`memory_query`, `memory_recent`, `memory_status`,
  `memory_briefing`, `memory_explore`) accept an optional `project`
  argument to target a specific project on a shared server.

### Fixed
- OpenCode hook events (`tool.execute.*`, `session.*`) were rejected with
  "missing session_id" because OpenCode sends `sessionID` (capital `ID`)
  and the extractor only matched `sessionId`. All spellings are now
  accepted ([#1]).
- MCP read tools were locked to the server's static `--project` (default
  `scratch`), so on a shared HTTP server they returned empty memory even
  while hooks populated the correct per-cwd project. The hook router now
  publishes the active project to a shared pointer that the read tools
  use as their default; an explicit `project` argument overrides it ([#2]).

## [0.1.0] - 2026-05-23

### Added
- Per-project UUID-namespaced wiki layout: pages live at
  `<wiki_root>/<workspace_id>/<project_id>/<page-path>`. Rename is now
  a single column update; purge is `remove_dir_all` on the project dir.
- CLI becomes a thin HTTP client: `bootstrap`, `status`, `search`,
  `reorg`, `lint`, `forget-sweep`, `embed`, `commit`, `backup`,
  `write-page` all delegate to the running server via `/admin/*` routes.
  The server is the sole writer of wiki + SQLite.
- `purge-project` command with cascade-delete indexes and per-project
  isolation guard (refuses to delete files claimed by sibling projects).
- `rename-project` command: column-only rename, no file moves.
- `memory_install_self_routing` MCP tool: installs the agent-routing
  snippet into CLAUDE.md / AGENTS.md / `.cursorrules` in one call.
- Read-only HTTP wiki browser (`/web`) with project tree, page view,
  and full-text search.
- Bearer token auth (`AI_MEMORY_AUTH_TOKEN` / `generate-auth-token`),
  Host-header allowlist, and 10 MB body cap for the HTTP server.
- `backup` / `restore` commands using `.tar.gz` archives with live-process
  guard (refuses to run if another `ai-memory` is active on the same data dir).
- Per-cwd project routing in hooks: observations route to the project
  matching the agent's working directory, not the server default.
- `opencode` / `openclaw` aliases for the OpenCode MCP client.
- Dockerised CLI wrapper (`bin/ai-memory`) with auto-restart for the
  local container and nudge for remote upgrades.
- `bootstrap` serialises parallel runs to prevent duplicate project creation
  and handles the case where the CWD has no git repo.
- Monthly log-md rotation to keep `log.md` from growing unbounded.
- `memory_consolidate` PreCompact checkpointing falls back to rule-based
  summarisation when no LLM is configured.
- `docs/lifecycle-ops.md`: safety matrix for state-touching commands
  (reset, restore, purge-project, rename-project).
- `docs/wiki-migrations.md`: when and how to write a wiki migration.

### Changed
- `bin/ai-memory` forwards `AI_MEMORY_SERVER_URL` and no longer creates
  `-w` mount-conflict directories.
- `bootstrap` resolves the repo root via `libgit2`, removing the
  `git` binary dependency.
- Admin routes consolidated: dry-run support, correct status codes,
  deduplicated handlers.
- Host-header allowlist sourced from `Config.allowed_hosts`; logged at
  startup so operators can verify the effective list.

### Fixed
- `AI_MEMORY_HOST_CWD` handling and dry-run no-project side effects.
- Web page view: strip leading H1 from body to prevent title duplication.
- `install-mcp` Codex config key was `bearer_token`, not
  `http_headers` / `headers`.
- Consolidator used server startup default project instead of the
  session's actual project.

[Unreleased]: https://github.com/akitaonrails/ai-memory/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.1.3
[0.1.2]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.1.2
[0.1.1]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.1.1
[0.1.0]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.1.0

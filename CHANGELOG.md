# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- New `POST /admin/delete-page` HTTP endpoint deletes a single page with
  explicit `(workspace, project)`. Like `purge-project`/`rename-project`, it
  uses no-create lookup — a delete on a typo'd or wrong scope now returns
  `404 workspace 'X' not found` instead of silently auto-creating the
  container and returning misleading `deleted: true`.
- New `ai-memory delete-page --path <P> --workspace <W> --project <P>` CLI
  subcommand, a thin client of `/admin/delete-page`. Mirrors the
  write-page/read-page CLI shape so terminal users get a complete
  delete-single-page surface for the first time.

### Fixed
- `memory_delete_page` (MCP) now accepts `workspace` alongside `project` and
  routes scope through `effective_ids_for_read_args`, the same path the read
  tools use. Previously a project name that lived in multiple workspaces
  could silently route the delete to the wrong slot and return `deleted:
  true` for a page that was never touched. Operators on shared (multi-
  workspace) servers should explicitly pass `workspace + project` to make
  the target unambiguous.

## [0.9.0] - 2026-06-02
### Added
- `openai-compat` LLM providers can now opt into strict JSON Schema structured
  output with `AI_MEMORY_LLM_COMPAT_STRICT=true`. Strict mode sends
  `response_format=json_schema` first for compatible Ollama, vLLM, LM Studio,
  llama.cpp, and gateway endpoints, while the tolerant JSON-object parser
  remains the default and the fallback for strict raw-call failures ([#70]).
- The read-only web browser now renders `[[wiki links]]` as clickable internal
  links to the target page. Supports `[[path]]`, `[[path|label]]`,
  `[[project:path]]`, and `[[workspace/project:path]]`, resolved against the
  current page's project unless the target carries its own scope; bare targets
  get a `.md` suffix. External schemes, path traversal, and links inside fenced
  or inline code are left as literal text ([#68]).
- `ai-memory serve --transport http` can host the entire HTTP surface under a
  configurable subpath with `--base-path` / `AI_MEMORY_BASE_PATH`; `/mcp`,
  `/hook`, `/admin/*`, `/api/v1`, and the web UI all move under that prefix.
  The web UI mount can also be changed with `--web-slug`, and custom
  `--web-ui-dir` SPAs receive injected `<base href>` plus
  `ai-memory-base-path` metadata for same-origin API calls behind reverse
  proxies ([#65]).
- `ai-memory move-project` can move projects across workspaces via the admin
  API. Fresh destinations use a lossless true move that keeps the same
  `project_id`, sessions, observations, handoffs, embeddings, and page history;
  existing same-named destination projects use copy-purge merge with explicit
  `on_conflict` handling. Admission webhooks can subscribe to the new
  `move_project` event and receive destination names in the context ([#60]).
- Page FTS now indexes normalized page paths, so searches can find pages by
  filename or slug even when the slug does not appear in the title/body ([#62]).
- Admission webhooks can now observe, mutate, or reject engine write/delete/
  purge operations, with authenticated actor context, loop-prevention skip
  lists for trusted re-entry, and non-blocking observer webhooks for mirrors
  and backups ([#55]).
- New `memory_delete_page` MCP tool deletes a single page by exact path,
  updates the SQLite index directly, and fires `op=delete` admission hooks
  before removal ([#55]).

### Fixed
- Backups no longer dereference symlinks under `wiki/`, preventing a planted
  wiki symlink from pulling arbitrary readable host-file contents into
  `backup.tar.gz`.
- `ai-memory restore` now validates tar entries before extraction and accepts
  only regular files/directories under the expected backup paths
  (`wiki/`, `db/memory.sqlite`, and `config.toml`), rejecting links, special
  files, unsafe paths, and unexpected archive entries.
- In multi-user mode (`[auth].token_pepper` configured), operational
  `/admin/*` endpoints now require the root token; DB-user tokens receive
  403 while single-user installs keep the historical permissive admin behavior.
- LLM provider clients now cap provider response bodies before JSON, text, or
  SSE parsing, and truncate error bodies from bounded buffers instead of
  buffering arbitrary-size responses.
- Non-blocking admission webhooks now have a process-level in-flight cap and
  webhook timeouts are clamped to a safe maximum, preventing observer hooks
  from growing unbounded background work during write bursts.
- Hook cwd/project resolution caching is now bounded with LRU-style eviction,
  preventing unbounded process-lifetime growth from streams of unique cwd
  values.
- `memory_write_page` tool description and routing prompts now steer agents
  toward writing the page title as a `# H1` on the first line of `body` and
  omitting the `title` argument. ai-memory already auto-derived the title from
  `# H1` (or path stem) when `title` was missing — the change is documentation
  only, but it eliminates a known source of MCP `JSON parsing` errors when the
  LLM failed to escape quotes/colons in `title` ([#67]).
- Custom `--web-ui-dir` frontends no longer serve raw `/index.html` without
  base-path injection; direct index requests and SPA fallback routes now return
  the injected shell, while static assets remain untouched ([#65]).
- `move-project` true moves now run through a wiki mutation gate: normal
  page writes/reindexes validate the `(workspace_id, project_id)` pair before
  touching disk, while true moves hold the exclusive side across the directory
  rename and DB re-stamp. Stale old-workspace writes now fail without creating
  orphan files, and V18 aborts if existing split-brain rows are present ([#60]).
- `move-project` copy-purge conflict detection now treats body, frontmatter,
  title, tier, and pinned status as the page identity under `on_conflict=block`,
  preventing metadata-only overwrites from slipping through ([#60]).
- `memory_write_page` calls that specify `project` without `workspace` now
  default to the active workspace published by hooks, and project-only reads use
  the same active-workspace resolution so the write can be read back without an
  explicit workspace ([#61]).
- `memory_read_page` now accepts explicit `workspace` + `project` for sibling
  projects and falls back to the stored DB body only when the markdown file is
  missing, not when the disk source of truth is corrupt or unreadable ([#63]).
- `openai-oauth` now speaks the current ChatGPT/Codex responses stream format
  for bootstrap/consolidation requests and avoids sending the unsupported
  `max_output_tokens` field on that endpoint ([#64]).
- `ai-memory write-page` now resolves an omitted `--project` through the same
  current-project heuristic as `read-page` and `search`, preventing writes from
  landing in `scratch` while the read-back targets the cwd-derived project
  ([#66]).

## [0.8.1] - 2026-05-30

### Fixed
- **Consolidation no longer fails on long sessions** (~5,000+ observations or
  multi-hour agent runs). Two bugs surfaced trying to consolidate a real
  16-hour / 7,234-observation session:
  - **Prompt confusion (regression from the v0.8 `slot_kind` work):** the
    multi-page consolidator prompt listed `slot_kind` values
    (`state` / `invariant`) immediately above the `tier` values
    (`working` / `episodic` / `semantic` / `procedural`). The LLM read them
    as one list and emitted `tier: "state"` in structured responses, which
    deserialisation rejected. Prompt now leads with `tier` (with explicit
    "EXACTLY ONE OF FOUR strings" emphasis), then `kind`, then `slot_kind`
    under its own clearly-scoped section that states "completely unrelated
    to tier" and "only for `_slots/*` paths."
  - **No token budget on the observation dump:** `build_request` and
    `build_batch_request_with_slots` dumped every observation into the
    prompt buffer, which exceeded the provider's 200k-token context on long
    sessions (the sabadell run produced a 235k-token request → 400 from
    the provider). New `window_observations_to_budget` walks the slice
    from most-recent backward, keeping each entry whose render cost fits
    in a 400k-char budget (~100k tokens), leaving room for the system
    prompt + schema + LLM output. When entries are skipped, a prepended
    note tells the LLM the context is partial so its summary doesn't
    pretend to cover the early session. Both `PreCompact` and
    `memory_consolidate` triggers benefit from the fix — both were silently
    failing into the `warn!()` catchall on sessions this long.
  - 5 unit tests guard the windowing invariants (empty input, fits-under-
    budget passthrough, most-recent-preserved, single-too-large-obs drops
    everything, observation-boundary alignment). No schema change, no
    config knob, backward-compatible for sessions that already fit.

## [0.8.0] - 2026-05-30

### Added
- **Multi-user attribution (v0.8 Phase 1, rolling out across milestones
  P1.1–P1.8).** ai-memory's data model stays single-tenant — every
  authenticated request sees every page — but writes can now be
  attributed to a named user. Five `ai-memory user` subcommands
  (`add`, `list`, `expire`, `revive`, `rotate-token`) manage a `users`
  table; the auth middleware resolves every request to one of four
  tiers (Anonymous, Root, DB user, 401), injects an
  `Extension<ActorContext>` + `Extension<AuthLevel>` for downstream
  consumers, and gates the root-only admin user-management endpoints.
  Tokens are 32 bytes of OS CSPRNG, stored only as
  `SHA-256(token || ":" || token_pepper)` (per-server pepper from
  `[auth].token_pepper`, auto-generated on `ai-memory init`); see
  [`docs/users.md`](docs/users.md) for the SHA-256-not-argon2id
  rationale, the four-rung auth ladder, and the backward-compat
  migration for pre-v0.8 installs. New v0.8 fields on `[auth]`:
  `root_username` / `root_email` / `root_name` (label for the bearer
  token's writes) and `token_pepper`. Per-page `author_id` + web UI
  surfacing lands in P1.6/P1.7; this milestone set ships P1.1
  (`ActorContext` + `UserId` in core), P1.2 (table + writer/reader
  ops + V14 migration), P1.3 (auth middleware), P1.4 (root-gated
  `POST/GET /admin/users` + `…/expire|revive|rotate-token`), and P1.5
  (CLI subcommands). **No behaviour change for existing single-user
  installs**: without `[auth].token_pepper` the multi-user lookup
  stays dormant, user-management endpoints 503 with a clear
  `multi-user not enabled` message pointing at `ai-memory init`,
  and the existing `bearer_token`-only flow keeps authenticating
  exactly as before.
- **`memory_query { global: true }` — cross-project global search** that
  reaches every project in every workspace in one call, with each hit
  annotated by its workspace + project so the agent can tell where it
  came from. Use when the agent doesn't know which project holds a
  cross-cutting note (shared infra/ops, a sibling app). Mutually
  exclusive with `scopes`/`project`/`workspace`. Routing snippet +
  `MEMORY_INSTRUCTIONS` now teach both broadening modes (`scopes` for
  named siblings, `global=true` for unknown locations) and explicitly
  warn that `memory_query` returns snippets — use `memory_read_page`
  for full bodies. The prompt-surface contradiction the original PR
  shipped ("there is no global 'search everything' mode" right after
  the bullet advertising `global=true`) was caught in the post-merge
  audit and rewritten; the prompt regression test now refuses any
  variant of that legacy phrasing
  ([#56], thanks @djalmajr).
- **Cross-project wiki links + dependency graph.** Wikilinks gain an
  explicit scope qualifier: `[[project:path.md]]` for a sibling project
  in the same workspace, `[[workspace/project:path.md]]` for another
  workspace. Bare links are unchanged (resolve within the source's own
  project). `links.to_workspace` / `links.to_project` join the primary
  key so the same `to_path` can land in two different projects without
  colliding. `memory_lint` now reports dangling cross-project refs
  (typo'd project vs missing/renamed target page), `memory_briefing`
  exposes `cross_project_dependents` / `cross_project_dependencies`
  per project, and `GET /api/v1/graph` returns the resolved cross-
  project edges for a graph view. Migration V13 rebuilds the `links`
  table preserving existing rows as `(to_workspace=NULL,
  to_project=NULL)` — same "local" semantics as before
  ([#57], thanks @djalmajr).

### Changed
- **FTS5 queries OR-join bare multi-word inputs** instead of the
  pre-existing AND default. A natural-language query like
  `"have we discussed cross project search strategy"` previously
  required every word to co-occur in one page — near-zero recall for
  multi-word queries, which the caller silently mistook for "never
  recorded". OR + BM25 ranking (callers already `ORDER BY rank`) keeps
  the best-matching pages at the top of the list, so the user-visible
  top-N is still AND-ish; OR just adds a relevant tail instead of
  returning nothing. Explicit FTS5 syntax (`OR`/`AND`/`NOT`/`NEAR`,
  quoted phrases, parens) is detected and preserved verbatim so the
  exact-match escape hatch stays available. 5 new unit tests guard the
  preservation contract (post-merge audit). Migration V12 rebuilds the
  FTS tables with `unicode61 remove_diacritics 2` so accent-free
  Portuguese queries (`"descricao da sessao"`) match accented stored
  text (`"descrição da sessão"`); contentless FTS — source rows
  untouched ([#58], thanks @djalmajr).
- **MCP write tools now honour the session's project (and create
  named projects on demand).** Three correctness fixes on
  `memory_write_page` / `memory_lint` / `memory_forget_sweep`:
  - A `memory_write_page { project: "X" }` for a project name that
    doesn't exist used to silently fall through to the session's
    active project (find-only resolution); writes meant for a fresh
    project polluted the current one. A new `write_target_ids`
    helper uses **get-or-create** for an explicit project name, so
    a named write always lands where the agent asked.
  - `memory_lint` + `memory_forget_sweep` previously always targeted
    the server's baked `--project` regardless of the session, so a
    cross-project lint or retention sweep could never reach the
    project the user was actually working in. Both now resolve
    through the same find-only `effective_ids_for_read_args` path
    the read tools use, with the hook-published active project as
    the fallback.
  - Both `lint` / `sweep` and the new `write_page` add explicit
    `workspace` + `project` args (defaulted to current session,
    documented with the v0.5.2 "**Omit unless the user explicitly
    names a *different* project.**" tail). 2 regression tests cover
    "Bug B" (explicit-project write must create + land) and
    "Bug C" (sweep must evaluate the named project, not the baked
    default) ([#59], thanks @djalmajr).

## [0.7.1] - 2026-05-29

### Fixed
- **`install-hooks --agent codex` no longer panics with `index not found`**
  when `~/.codex/config.toml` carries an `[mcp_servers]` table that has other
  MCP servers (context7, node_repl, …) but no `ai-memory` entry — a
  perfectly valid setup since ai-memory can integrate via hooks alone.
  `infer_codex_mcp_config` used `toml_edit`'s panicking `Index` impl with
  bare `[]` chains; it now walks the table via `.get()` and returns `None`
  on any missing key. Mirrors the safe pattern the JSON variant has used
  all along. Adds 4 regression tests covering missing-entry,
  missing-table, empty-doc, and bare-entry inputs
  ([#53], thanks @Otavio-Machado-Santos).
- **`install-hooks --agent claude-code` no longer silently stages 0 scripts
  and points `settings.json` at an empty directory.** On macOS — and any
  install where the binary lives outside the repo and the system package
  paths (`/usr/local/share`, `/usr/share`) are absent — `resolve_hooks_dir`
  fell through to the data-local candidate, which was *also* the staging
  destination. The wipe-then-copy flow inside `stage_hook_scripts_in` then
  deleted the very scripts it was about to read, leaving 0 copied; the
  caller proceeded to rewrite `settings.json` anyway, disabling capture
  with no error. The function now (a) canonicalizes source and destination
  paths, skips the wipe + copy when they match and verifies in-place,
  preserving any scripts a prior `setup-agent` run extracted there, and
  (b) bails with an actionable error pointing at `--hooks-dir` or
  `ai-memory setup-agent` whenever zero scripts are present in either
  branch. Adds 3 regression tests
  ([#52], thanks @Otavio-Machado-Santos).
- **macOS thin-client wrapper no longer crashes with "Permission denied" in
  the log file appender.** The `bin/ai-memory` wrapper passed
  `-u $(id -u):$(id -g)` to the one-shot helper container, which on macOS
  collides with the data volume owner (uid 1000 inside the container vs
  uid 501/502 on the host). The wrapper now skips `-u` on Darwin so the
  container runs as its default uid 1000 — Docker Desktop's file-sharing
  layer handles host ownership transparently — while Linux and other
  Unix systems continue to receive `-u`. Same change also hardens the
  `${TTY_ARGS[@]}` / `${NETWORK_ARGS[@]}` / `${ENV_ARGS[@]}` /
  `${USER_ARGS[@]}` expansions for `set -u` compatibility on macOS's
  default bash 3.2 ([#51], thanks @abnersajr; supersedes [#50]).

## [0.7.0] - 2026-05-29

### Added
- **`memory_read_page` MCP tool** (`read-only`) for fetching the FULL body of a
  wiki page — pass `path` for a direct lookup or `query` to fetch the top FTS5
  hit's full body. Complements `memory_query`'s 24-word snippets when an agent
  needs to read an entire decision page end-to-end. Also exposed as
  `GET /admin/read-page?workspace=…&project=…&path=…` (admin HTTP) and the new
  `ai-memory read-page` CLI subcommand (thin HTTP client). All three surfaces
  scope to the current project by default and route user-supplied paths through
  `PagePath::new`, so traversal attempts (`../etc/passwd`) are rejected with
  400. ARCHITECTURE.md's MCP-tool table grows from 12 to 13 rows ([#49]).
- `_slots/*.md` pages can now declare `slot_kind: state` or
  `slot_kind: invariant` frontmatter. `state` remains the default for existing
  slots; `invariant` marks high-resistance project context or preferences that
  consolidation should not rewrite unless observations directly contradict the
  existing slot content ([#47], closes [#14]).

### Fixed
- **Windows PowerShell hooks no longer hang or stall the agent.** The shared
  `hooks/lib/ai-memory-hook.ps1` read stdin via `[Console]::In.ReadToEnd()`,
  which blocks indefinitely when the agent does not close the stdin pipe
  (observed on Claude Code `PreCompact`); because the `Invoke-WebRequest`
  timeout only starts after the read returns, a stuck read meant the hook
  never POSTed anything. Stdin is now read asynchronously, guarded by
  `[Console]::IsInputRedirected` with a 2s cap, so the hook can never freeze.
  HTTP timeouts were also raised from 1s to 3s (POST) / 2s (handoff GET) to
  tolerate remote servers over higher-latency links. The full raw payload is
  still forwarded (parity with `_lib.sh`), so observation title/body stay
  intact. Affects every agent still on the PowerShell hook runner
  (Codex, Cursor, Gemini CLI, Antigravity, OpenCode on Windows) ([#48]).
- Page upserts now treat frontmatter/title/tier/pinned changes as real page
  updates instead of short-circuiting solely on unchanged body text, keeping
  the SQLite index consistent with markdown frontmatter-only edits ([#47]).

## [0.6.1] - 2026-05-28

### Added
- `Cache-Control: private, max-age=N` headers on all `/api/v1` read endpoints
  (lists/search/recent/briefing/overview: 30–60s; single-page reads: 300s).
  Errors stay uncached. A polling SPA no longer hits the DB on every request.
- **ETag + conditional GET** on the single-page read endpoint
  (`GET /api/v1/workspaces/{ws}/projects/{p}/pages/{*path}`): the response
  carries `ETag: "<sha256>"` over the markdown body, and a follow-up request
  with matching `If-None-Match` returns `304 Not Modified` with no body.
- **`--cors-allow-origin`** flag (repeatable) and
  `AI_MEMORY_CORS_ALLOW_ORIGINS=a,b,c` env var. When set, a `CorsLayer` is
  attached **only to `/api/v1`** (`/mcp`, `/hook`, `/admin`, and `/web` are
  intentionally untouched) so a separately-hosted SPA can call the API. Each
  origin must include a scheme; `*` is rejected at startup (CORS spec forbids
  credentials + wildcard). Empty list = same-origin only, unchanged behaviour.

## [0.6.0] - 2026-05-28

### Added
- Read-only **`/api/v1`** JSON surface for third-party frontends: workspaces,
  projects, pages (list + read with frontmatter, body, resolved links, and
  back-links), recent, briefing, search (GET single/global + POST multi-scope
  capped at 25 scopes), and workspace/project `overview` aggregates (handoff +
  briefing + memory-health drill-down). Mounted before the bearer +
  host-allowlist middleware so existing auth applies automatically. Read-only
  by construction — zero writer calls in the handlers ([#7]).
- **`--web-ui-dir`** flag on `ai-memory serve` to host any static SPA at
  `/web` (same origin as the API, behind the same auth), with `index.html`
  SPA fallback via `tower-http::ServeDir`. Validates the directory exists
  and contains `index.html` before binding. When the flag is absent, the
  built-in server-side `/web` browser stays the default ([#7]).
- MCP read tools (`memory_query`, `memory_recent`, `memory_status`,
  `memory_briefing`, `memory_explore`) accept optional `workspace` +
  `scopes` args for explicit multi-project queries; existing single-`project`
  behaviour is unchanged and remains the default ([#7]).
- New reader queries powering the API: per-page outgoing links + incoming
  back-links, workspace-aggregated briefing, memory-health (stale /
  duplicate / orphan) counts and drill-down lists, workspace summaries
  with last-update timestamps ([#7]).

### Fixed
- Antigravity `pre-tool-use` hook now emits the documented
  `{"decision":"allow"}` JSON contract instead of an empty `{}`, while
  keeping the `ai_memory_post_hook` call fully suppressed
  (`>/dev/null 2>&1 || true`) so the `queued` body never bleeds into the
  hook's stdout. Identical logic for `.sh` and `.ps1`; other hook scripts
  remain silent and unchanged ([#44], thanks @ArtroxGabriel).

### Docs
- New **[`docs/frontend-api.md`](docs/frontend-api.md)** integration guide
  for `/api/v1`: auth flow, response schemas (`PageHit`, `BriefingSnapshot`,
  `HealthDetail`, `PageLinks`, …), error model, limits/pagination,
  custom-UI hosting, a worked `fetch`/`curl` example, and pointers to the
  canonical source-of-truth files.

## [0.5.2] - 2026-05-28
### Added
- `ai-memory status` / `status --json` now includes passive process-scoped LLM
  and embedding provider health based on the last real provider call, without
  active probing or token spend ([#46]).

### Changed
- Agent-facing prompts (`MEMORY_INSTRUCTIONS`, the `CLAUDE.md`/`AGENTS.md`
  routing snippet, and the per-tool `project`/`cwd` arg docstrings) now lead
  with a clear "default to the current project — do not pass `project` or
  `cwd` args unless the user names a *different* project" rule, plus a
  reminder that the SessionStart auto-fetched handoff block already covers the
  current project. Reduces cross-agent friction where a fresh agent surfaced
  the wrong project's handoff because the LLM over-eagerly passed scoping
  args. Doc-only, no behaviour change.

### Fixed
- Claude Code hook installs on native Windows now render Git Bash-compatible
  `bash -c` commands that keep the POSIX `.sh` hook scripts and convert
  drive-letter paths to Git Bash paths, matching Claude Code's actual hook
  runner instead of emitting PowerShell commands ([#45]).
- `ai-memory llm-test --provider anthropic-oauth` now parses and maps to the
  Anthropic OAuth provider instead of being rejected by clap ([#43]).

## [0.5.1] - 2026-05-27
### Changed
- Docker release publishing now builds Linux x86_64 and aarch64 artifacts once,
  reuses those artifacts for Docker images, and smoke-tests both amd64 and arm64
  images after assembling the multi-arch manifest.
- The AUR `ai-memory-bin` package now supports aarch64 using the prebuilt Linux
  aarch64 release artifact.
- Docker source builds now use the vendored Tailwind CSS artifact, avoiding
  cross-architecture Tailwind CLI cache collisions during multi-arch releases.

## [0.5.0] - 2026-05-27
### Fixed
- Docker release images now publish both `linux/amd64` and `linux/arm64`
  manifests, so Apple Silicon and ARM64 Linux hosts can pull the image without
  forcing x86 emulation ([#41]).

## [0.4.0] - 2026-05-27
### Added
- `anthropic-oauth` LLM provider: use a Claude Pro/Max subscription via
  `claude setup-token` instead of an API key. In-Rust, reuses the existing
  Anthropic Messages client (incl. structured output). **Unofficial and
  against Anthropic's usage policies — use at your own risk** (docs warn
  prominently).
- Opt-in `AI_MEMORY_CONSOLIDATE_ON_SESSION_END`: when set and an LLM provider
  is configured, SessionEnd additionally runs LLM consolidation on top of the
  always-written rule-based summary page (non-fatal on failure) ([#40]).

### Changed
- Docs recommend a small/fast model (Haiku/mini class) for the OAuth /
  subscription LLM backends — consolidation/lint/explore is summarisation, not
  hard reasoning, and small models are far easier on subscription rate limits.
- Aligned every prompt surface + doc with actual SessionEnd behavior: it always
  writes a rule-based summary page + handoff; LLM consolidation runs on
  PreCompact, on demand via `memory_consolidate`, and at session end only
  behind the new opt-in flag ([#40]).

### Fixed
- Windows own-write detection: `inode_of` now returns the real NTFS file index
  (was always `0`, which collapsed the watcher's own-write set) ([#37]).
- `ai-memory upgrade` no longer fails with `invalid value 'lib' for --agent` —
  the hook-refresh loop skips the shared `lib/` helper dir ([#38]).
- Native packaging CI now supports non-root runners whose `systemd-tmpfiles`
  lacks `--dry-run`, while still operating only inside a temporary alternate
  root.

## [0.3.2] - 2026-05-27
### Fixed
- AUR release publishing now runs with `HOME=/home/aurbuild` and an explicit
  `GIT_SSH_COMMAND`, so the workflow uses the configured AUR deploy key.

## [0.3.1] - 2026-05-27
### Changed
- Reissued the release after the initial AUR publish failure. This release was
  superseded by 0.3.2 for the AUR SSH home fix.

## [0.3.0] - 2026-05-27
### Added
- Arch Linux native packaging assets: source and prebuilt AUR package
  definitions, system/user systemd units, sysusers/tmpfiles entries, native
  config/env templates, CI-safe alternate-root packaging checks, and a manual
  disposable-distrobox integration harness for validating real service startup
  before publishing.
- Tag-triggered release automation now validates that `vX.Y.Z` matches
  `Cargo.toml`, publishes a native Linux release tarball, keeps Docker image
  publishing behind Docker Hub secrets, and optionally publishes both AUR
  package bases when `AUR_SSH_PRIVATE_KEY` is configured.
- `memory_write_page` MCP tool for explicit durable annotations, so agents can
  write permanent wiki knowledge without abusing single-use handoffs.
- `openai-oauth` LLM provider for ChatGPT/Codex accounts, including
  `ai-memory auth login|logout|status` device-flow commands and token storage
  in `<data_dir>/auth.json`.
- `copilot` LLM provider for GitHub Copilot Chat accounts. It stores a GitHub
  token via `ai-memory auth login copilot`, exchanges it for a short-lived
  Copilot API token, and sends Copilot Chat requests with `vscode-chat`
  integration headers.

### Fixed
- `install-mcp`, `install-hooks`, and `setup-agent` now honor configured
  `AI_MEMORY_SERVER_URL` defaults; `install-hooks` also reuses an existing
  ai-memory MCP entry when present, preventing remote MCP setups from
  regenerating loopback-only lifecycle hooks during installs/upgrades.
- Filesystem watcher now reindexes a project when backends report only a
  parent-directory event, improving external editor capture on macOS/FSEvents.
- OpenAI strict structured-output schema normalization now strips generated
  `$ref` annotation siblings and rewrites generated enum `oneOf` schemas to
  `anyOf`, unblocking `memory_consolidate multi_page=true` on OpenAI models.
- OpenAI-compatible embedding calls now truncate oversized page bodies, surface
  provider errors returned in HTTP 200 bodies, retry bounded HTTP 429 responses,
  and may reuse `LLM_API_KEY` when a custom embedding base URL is configured.
- `ai-memory embed --force` without `--project` now re-embeds every project in
  the workspace and purges stale/superseded embedding rows in the same scope.
- Windows hook `cwd` values sent to a Linux server now resolve projects by the
  final path component instead of treating the full backslash path as the
  project name.

## [0.2.0] - 2026-05-26
### Added
- `ai-memory bootstrap` now prunes collected sources before POSTing to the
  server and supports `--chunk-input-tokens` to process large repositories via
  sequential LLM calls instead of one oversized prompt.
- Opt-in extension event metadata for `/hook`: custom integrations can
  pass `extension=<namespace>` (and optionally `source_event=<name>`) to
  preserve a validated third-party source event while storage keeps the
  canonical `ObservationKind` closed. Unknown events without an extension
  still collapse to `other` with no source-event metadata.
- `.ai-memory.toml` marker file lets a directory tree declare its
  `workspace` (required) and `project` (optional) without depending on
  `basename($cwd)`. Lifecycle hook scripts walk up from `cwd` to find
  the closest marker and forward `cwd` plus the declared names as
  query params on `POST /hook` and `GET /handoff`. Markers can also set
  `project_strategy = "repo-root"` to derive project identity from the
  main git repository root, so linked worktrees share one project. Server
  accepts the new params as optional overrides;
  absent marker means the previous behaviour (`workspace = "default"`,
  `project = basename(cwd)`) — fully backward compatible. See
  [`docs/marker-file.md`](docs/marker-file.md).
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
- Google Gemini embeddings via `AI_MEMORY_EMBEDDING_PROVIDER=google` or
  `gemini`, with `gemini-embedding-001` as the default embedding model and
  `GEMINI_API_KEY` / `GOOGLE_API_KEY` support.
- Antigravity CLI (`agy`) support for MCP config (`serverUrl`) and lifecycle
  capture through its `PreInvocation`, `PreToolUse`, `PostToolUse`, and `Stop`
  hook events.
- README support matrix for operating systems, agent integrations, LLM
  providers, and embedding providers.
- `ai-memory uninstall` — removes ai-memory's hooks, MCP registration, and
  CLAUDE.md/AGENTS.md instruction block across all detected agents (dry-run by
  default; `--apply` to execute, with timestamped backups). `--purge-data`
  wipes wiki/db/raw via the reset guard. `--only hooks|mcp|instructions` to
  narrow. MCP matching is endpoint-based by default; pass `--mcp-url` when the
  server was installed with a custom endpoint and `--mcp-name` only to narrow
  removal to one matching entry. Docker/volume teardown is printed as a hint,
  not executed.

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
- `ai-memory uninstall --purge-data` now previews the `wiki/`/`db/`/`raw/`
  wipe in dry-run (mirroring `reset`) and refuses **up front** if an
  `ai-memory` process is alive (all-or-nothing) instead of removing the
  wiring and then skipping the purge. The data wipe is now shared with
  `reset` via a single internal helper.
- `ai-memory uninstall` only deletes generated plugin/extension files after
  re-validating their ai-memory-generated content, and never treats a matching
  filename or MCP server name alone as proof of ownership.

### Fixed
- `serve` now warns and starts when stored embedding rows were created with a
  different `(provider, model, dim)` than the current config. Hybrid search
  ignores stale rows until `ai-memory embed --force` or scheduled backfill
  re-embeds them, avoiding the previous startup deadlock.
- Session capture now persists every documented agent kind (`cursor`,
  `gemini-cli`, `claude-desktop`, `openclaw`, `omp` / `pi`) instead of
  failing the `sessions.agent_kind` database CHECK for agents added after
  the initial schema.
- `memory_handoff_begin` and `memory_handoff_accept` now resolve the active
  project the same way the briefing/search tools do, so MCP handoffs land in
  the project currently reported by hooks instead of the server's baked
  default project.
- Natural-language `memory_query` text containing bare colons, such as
  `pick: handoff`, no longer trips FTS5 column syntax errors while explicit
  FTS operators like `quick OR slow` remain supported.
- Marker-file routing now reaches the generated OpenCode and OMP
  TypeScript hook integrations, not only the POSIX/PowerShell script
  hooks. POSIX helpers also preserve the outer hook `cwd` when nested
  tool payloads contain their own `cwd`, and encode `+` correctly in
  marker-derived query parameters.
- `backup --to` now streams the tarball to disk instead of buffering the full
  archive in CLI memory.
- Hyphenated FTS5 queries such as `ai-memory` are normalized safely instead of
  being parsed as column operators.
- Gemini 2.5 Flash requests disable default dynamic thinking so hidden thought
  tokens do not consume `maxOutputTokens` and truncate strict JSON responses.
- `install-mcp --client claude-code` now prints the direct-edit JSON path as
  `~/.claude.json`, matching the `--apply` path and `claude mcp add` behavior.
- Hook routing now evicts a stale project-cache entry and retries once when a
  live server sees a cached project deleted underneath it, such as after
  `purge-project`, so capture resumes without restarting the server.
- Session-start handoff hooks now include `cwd` even without a marker file, so
  default `project = basename(cwd)` projects receive pending handoffs without
  requiring `.ai-memory.toml`.
- `ai-memory uninstall` now removes only ai-memory commands from mixed nested
  hook entries, preserves third-party commands in the same matcher, and removes
  legacy Codex inline-table MCP entries.
- Generated POSIX hook commands now shell-quote script paths and env values
  with metacharacters, fixing custom hook directories containing spaces and
  preventing shell-active token/URL fragments.
- OpenClaw's generated plugin now forwards marker-file routing params just like
  the OpenCode and OMP generated integrations.
- The Linux/macOS Docker wrapper now lets thin-client commands such as
  `status` and `bootstrap` reach the local quick-start server bound on the
  host's `127.0.0.1:49374`.

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

[Unreleased]: https://github.com/akitaonrails/ai-memory/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.9.0
[0.8.1]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.8.1
[0.8.0]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.8.0
[0.7.1]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.7.1
[0.7.0]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.7.0
[0.6.1]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.6.1
[0.6.0]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.6.0
[0.5.2]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.5.2
[0.5.0]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.5.0
[0.4.0]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.4.0
[0.3.2]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.3.2
[0.3.1]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.3.1
[0.3.0]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.3.0
[0.2.0]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.2.0
[0.1.3]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.1.3
[0.1.2]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.1.2
[0.1.1]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.1.1
[0.1.0]: https://github.com/akitaonrails/ai-memory/releases/tag/v0.1.0

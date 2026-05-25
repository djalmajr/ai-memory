<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo-dark.png">
    <img alt="ai-memory" src="docs/logo.png" width="480">
  </picture>
</p>

> Long-term memory for AI coding agents. Quit Claude Code mid-task,
> start OpenAI Codex in the same directory, continue without
> re-explaining the architecture, the failed approaches, or the open
> questions.

[![status: v0.2 milestones complete](https://img.shields.io/badge/status-v0.2--complete-green)](docs/ARCHITECTURE.md)
[![Rust](https://img.shields.io/badge/rust-1.95+-blue)](rust-toolchain.toml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Support Matrix

| Area | Status | Notes |
|---|---|---|
| Linux | Supported | Primary Docker/server target and CI platform. |
| macOS | Supported | Workspace tests run in CI; native source builds are supported. |
| Windows via WSL2 | Supported | Use the Linux install path inside WSL2 when the agent runs there. |
| Native Windows | Experimental | PowerShell wrapper and `.ps1` hooks are available; real agent harness feedback still needed. See [`docs/windows.md`](docs/windows.md). |
| Claude Code | Supported | MCP config + lifecycle hooks. |
| Codex | Supported | MCP config + lifecycle hooks. |
| OpenCode | Supported | Remote MCP config + generated TypeScript plugin. |
| Cursor | Supported | MCP config + lifecycle hooks. |
| Gemini CLI | Supported | MCP config + lifecycle hooks. |
| Oh My Pi / OMP | Supported | `pi` / `omp` aliases for MCP config + TypeScript extension. |
| Claude Desktop | MCP-only | Uses `mcp-remote`; no lifecycle hooks. |
| OpenClaw | MCP-only | OpenClaw has MCP/webhook ingress but no lifecycle hook surface. |
| LLM providers | Supported | Anthropic, OpenAI, Gemini, and OpenAI-compatible endpoints. |
| Embedding providers | Supported | OpenAI and Voyage. |

## What it is

LLM coding agents lose all context when a session ends. ai-memory
gives them a shared, persistent wiki: every prompt, tool call, and
decision is captured automatically; when a session ends, the relevant
pages get rewritten as a coherent narrative; when the next agent
starts (Claude Code, Codex, OpenCode, …) it sees a handoff with
"where you left off" already prepended.

The wiki is plain markdown in a git repo - `grep`-able, openable in
Obsidian, backed up with `rsync`. No vector database to babysit, no
`write_note` ceremony, no manual context-loading. The full design is
in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); the influences and
priors are at the [bottom](#influences-and-prior-art).

## Key features

- **Zero-friction capture.** Lifecycle hooks fire-and-forget every
  prompt + tool call + session boundary. You never type `write_note`.
- **Cross-agent handoffs.** Quit Claude Code mid-task, start Codex
  in the same directory hours later - the next agent sees a
  "where you left off" block before its first prompt.
- **Per-project isolation by construction.** Each project lives at
  `<wiki_root>/<workspace_id>/<project_id>/…` keyed by stable UUIDs.
  Same page path can exist in two projects without collision; a
  rename is one column update; a purge is one `rm -rf`.
- **Karpathy-style LLM wiki.** Pages are compiled from observations
  at session-end (or PreCompact), not retrieved over raw logs.
  Supersession chain + git-versioned markdown means you can
  time-travel with `git log`.
- **Built-in `/web` browser.** Read-only HTML UI for the wiki -
  project list, folder tree, FTS5 search, markdown rendering, dark
  mode. Mounted on the same axum server as MCP.
- **Multi-agent + multi-machine ready.** Supported clients: Claude
  Code, Codex, OpenCode, Cursor, Claude Desktop (via `mcp-remote`),
  Gemini CLI, OpenClaw, and Oh My Pi / OMP (`pi` / `omp` aliases).
  Server runs local (loopback) OR on a homelab box (LAN/VPN/cloud)
  with bearer-token auth.
- **Thin-client CLI.** `ai-memory bootstrap`, `purge-project`,
  `rename-project`, `lint`, `embed`, `forget-sweep`, `backup` are
  all HTTP clients of the running server - never touch SQLite or
  wiki files directly. Server is the single source of truth.
- **LLM is opt-in.** Zero-LLM mode still gives you FTS5 search +
  rule-based summarisation. Add a provider when you want consolidated
  pages and lint contradictions.

## Use cases

- **"Quit at 4 PM, pick up at 9 AM in a different agent."** The
  classic. SessionStart hook in the next agent (any of the
  supported CLIs) prepends a typed handoff with the open questions,
  next steps, and a session summary.
- **"What did we decide about X six weeks ago?"** Type
  `memory_query X` from the agent (or `ai-memory search X` from a
  terminal) - FTS5 over the wiki. Pages are LLM-consolidated, so
  the hit is a coherent decision page, not a raw chat log.
- **"This new project has months of history before ai-memory."**
  `cd /path/to/my-project && ai-memory bootstrap` collects
  `git log`, README, `docs/`, module headers, project rules and
  one-shot-summarises them into seed wiki pages. Future sessions
  build on top.
- **"Run one ai-memory for the whole household."** Stand the server
  up on a homelab box at `0.0.0.0:49374` with a bearer token; every
  laptop/desktop talks to it. Per-cwd routing keeps each project's
  pages cleanly separated; the `/web` UI is reachable from a
  browser anywhere on the LAN.
- **"Audit what landed before sharing with a teammate."** Browse
  the wiki at `http://<server>:49374/web` - HTTP Basic dialog if
  auth is on, paste the token as password. Per-project tree view,
  rendered markdown, supersession chain visible per page.
- **"Drop an experiment, keep the rest."**
  `ai-memory purge-project --project experimental --confirm`.
  Atomic: that project's DB rows cascade away, its wiki subdir gets
  `rm -rf`'d, every sibling project is untouched by construction.

## Quick start

You need: Docker + an agent CLI (Claude Code, Codex, OpenCode, OMP, Cursor,
or anything else that speaks MCP).

The default quick-start has **no authentication** - the server binds
to loopback only, so on a single-user laptop nothing else can reach
it. Adding a bearer token is a one-line change once you're ready to
expose the server on the LAN; see [Security](#security) below.

```bash
# 1. Install the ai-memory CLI wrapper (a ~3 KB shell script that
#    runs the binary inside docker with your $HOME mounted). This is
#    the only thing that needs to live on the host filesystem.
mkdir -p ~/.local/bin
curl -fsSL https://raw.githubusercontent.com/akitaonrails/ai-memory/main/bin/ai-memory \
    -o ~/.local/bin/ai-memory
chmod +x ~/.local/bin/ai-memory
# Most distros put ~/.local/bin on PATH automatically. If `which
# ai-memory` comes up empty, add this to ~/.bashrc / ~/.zshrc:
#     export PATH="$HOME/.local/bin:$PATH"

# 2. Start the server. `--restart unless-stopped` makes it come back
#    on docker daemon restart and on machine boot (provided your
#    docker service is enabled at boot — `sudo systemctl enable
#    docker` on most distros). Loopback-only bind (`127.0.0.1:49374`)
#    so nothing outside this machine can reach it. Omit the LLM /
#    EMBEDDING lines for zero-LLM mode — FTS5 search still works
#    without any keys.
docker run -d --name ai-memory \
    --restart unless-stopped \
    -p 127.0.0.1:49374:49374 \
    -v ai-memory-data:/data \
    -e AI_MEMORY_LLM_PROVIDER=anthropic \
    -e ANTHROPIC_API_KEY=sk-ant-... \
    -e AI_MEMORY_EMBEDDING_PROVIDER=openai \
    -e OPENAI_API_KEY=sk-... \
    akitaonrails/ai-memory:latest

# 3. Wire your agent CLI in two commands. The wrapper takes care of
#    mounts + auto-detecting ~/.claude/settings.json. Re-run with
#    `--agent codex`, `--agent opencode`, `--agent gemini-cli`,
#    `--agent omp`/`pi`, `--client cursor`, `--client gemini-cli`, etc.
#    for additional agents; full list in docs/install.md.
ai-memory install-mcp   --client claude-code --apply
ai-memory install-hooks --agent  claude-code --apply
```

On Linux/macOS, that's it. Start a Claude Code session as usual - every
prompt and tool call now lands in ai-memory, and the next session you
open in this project will see a handoff with where you left off.

The `install-mcp` / `install-hooks` commands default to
`http://127.0.0.1:49374` (matching the server above) and no bearer
token. Both are idempotent - re-runs replace ai-memory's entry,
preserve every other server / hook you have configured, and write a
timestamped `.bak-<ts>` next to the file before each modifying
write. The hook scripts are staged into
`~/.local/share/ai-memory/hooks/<agent>/` automatically; re-running
overwrites them so future image updates ship updated hooks. Drop
`--apply` to print the snippet instead of mutating.

### Windows Status

Windows support has two separate modes. If your agent runs inside WSL2,
use the Linux quick start inside WSL2. If your agent runs as a native
Windows process, use the PowerShell wrapper and `.ps1` hooks. Do not mix
the two path worlds. See [`docs/windows.md`](docs/windows.md) for the
native Windows, WSL2, and current real-world testing notes.

> **Prefer docker compose?** Clone the repo and run
> `docker compose -f docker/docker-compose.yml up -d` instead of
> step 2. The bundled compose file already has
> `restart: unless-stopped`, a healthcheck, and the named volume
> wired up; step 3 is identical.

> **Server on a different machine?** Replace `-p 127.0.0.1:49374:49374` with
> `-p 0.0.0.0:49374:49374` on the server's docker run; on the client, set
> `export AI_MEMORY_SERVER_URL=http://<server-ip>:49374` and add
> `--server-url <same>` to the install-mcp / install-hooks commands.
> See [Security](#security) — anything non-loopback should also have a bearer token.

### Keeping ai-memory up to date

The wrapper checks Docker Hub at most once every 24 hours and prints
a one-line warning to stderr when a newer image is available. To
upgrade:

```bash
ai-memory upgrade
```

In order: (1) self-upgrades the wrapper script itself by re-fetching
`bin/ai-memory` from GitHub (validated against the shebang; falls
back gracefully if curl is missing or perms block the in-place
replacement), (2) `docker pull`s the latest image, (3) re-stages
hook scripts under `~/.local/share/ai-memory/hooks/<agent>/` for
every agent you've configured, and (4) tells you how to restart the
server container so the new binary is picked up. The hook refresh
is idempotent - re-running `install-hooks --apply` replaces the
seven keys ai-memory owns and leaves every other hook the user has
wired up alone. Set `AI_MEMORY_NO_VERSION_CHECK=1` to silence the
daily check, or `AI_MEMORY_WRAPPER_URL=<url>` to pin the self-upgrade
source (e.g. a fork or a tagged release).

After the upgraded server starts, ai-memory applies SQLite schema
migrations automatically (`Store::open`) and runs pending wiki-structure
migrations before the watcher starts. No manual database reset or wiki
rewrite is required for normal upgrades; derived indexes such as FTS
tables are rebuilt/backfilled by migrations or the watcher reconciliation
path.

> **If your server runs on a different host** (scenarios C/D), `ai-memory upgrade`
> only refreshes the local wrapper, your local image, and your hook scripts. You
> still need to redeploy the server separately — run `bin/deploy` on the homelab
> box (or `docker compose pull && docker compose up -d` in your deploy dir) so
> the server picks up the new binary too.

> **Inside ai-jail (or any bwrap sandbox)?** The wrapper at
> `~/.local/bin/ai-memory` works fine — the sandbox bind-mounts
> `~/.local` read-only, so the script is visible from inside, and
> `/var/run/docker.sock` is already passed through. Run the
> `install-*` commands *outside* ai-jail (they need to write to
> `~/.local/share/ai-memory/hooks/`, which the sandbox keeps
> read-only); daily use from inside the sandbox needs no binary at
> all (agents reach ai-memory over MCP).

**For everything else** - Codex, OpenCode, OMP, Cursor, Claude Desktop,
Gemini CLI, OpenClaw, the curl-based hook installer (no docker
needed), running ai-memory without docker, the full subcommand
reference, the homelab deploy pattern, security hardening - see
[**`docs/install.md`**](docs/install.md).

## Security

The default Quick start runs **without authentication** because the
server is bound to loopback (`127.0.0.1:49374`) - no process outside
this machine can reach it. That's the safest default for a personal
laptop and matches the "single-user, single-machine" use case the
project is optimised for.

### When you need bearer auth

Enable bearer auth if **any** of these are true:

- The server is exposed beyond loopback (LAN, VPN, reverse proxy, cloud).
- More than one untrusted process runs on the same machine.
- The data dir contains observations from sensitive projects you
  wouldn't want any local user to read.

### Enabling bearer auth (turn-key recipe)

```bash
# 1. Generate a token (one-time; save the output somewhere).
TOKEN=$(ai-memory generate-auth-token)
echo "$TOKEN"   # 64 hex chars

# 2. Pass it to the server on startup. Note the bind is now 0.0.0.0
#    so remote clients can reach it; only do this with a token set.
docker run -d --name ai-memory \
    --restart unless-stopped \
    -p 0.0.0.0:49374:49374 \
    -v ai-memory-data:/data \
    -e AI_MEMORY_AUTH_TOKEN="$TOKEN" \
    -e AI_MEMORY_LLM_PROVIDER=anthropic \
    -e ANTHROPIC_API_KEY=sk-ant-... \
    akitaonrails/ai-memory:latest

# 3. Set the same token in every client environment that needs to
#    reach this server:
export AI_MEMORY_AUTH_TOKEN="$TOKEN"

# 4. Re-run install-mcp / install-hooks so the agent configs pick
#    up the new token + URL.
ai-memory install-mcp   --client claude-code --apply \
    --server-url "http://192.168.0.90:49374/mcp" \
    --auth-token "$TOKEN"
ai-memory install-hooks --agent  claude-code --apply \
    --server-url "http://192.168.0.90:49374" \
    --auth-token "$TOKEN"
```

When the server has `AI_MEMORY_AUTH_TOKEN` set, every request to
`/mcp`, `/hook`, `/handoff`, `/admin/*`, and `/web/*` must present
the token. HTTP clients (MCP, hooks, CLI) send an
`Authorization: Bearer <token>` header; the `/web/*` browser flow
uses HTTP Basic auth (see below). Token comparison uses
`subtle::ConstantTimeEq` to rule out timing-based recovery.

When the server has **no** `AI_MEMORY_AUTH_TOKEN` set AND binds to a
non-loopback address, it logs a loud `warn` on startup. That's the
signal to either lock the bind back to `127.0.0.1` or set a token.

See [`docs/deploy.md`](docs/deploy.md) for the full homelab pattern
(bearer + TLS via cloudflared + reverse proxy).

### DNS-rebinding guard (`AI_MEMORY_ALLOWED_HOSTS`)

Any non-loopback bind also needs `AI_MEMORY_ALLOWED_HOSTS` set to the
host/IP the clients will use, otherwise the HTTP server rejects
external `Host` headers with 403 before they reach `/mcp`, `/hook`,
`/handoff`, `/admin/*`, or `/web/*`:

```bash
-e AI_MEMORY_ALLOWED_HOSTS="<server-ip>,localhost,127.0.0.1"
```

Clients hitting the server by IP or hostname will be accepted;
requests with an unrecognised `Host` are refused. This guards against
DNS-rebinding attacks where a malicious page tricks the browser into
sending requests to the server using a different hostname.

### Browser access to `/web`

When the server has **no** bearer token set, visit
`http://<host>:49374/web` in any browser - no prompt.

When `AI_MEMORY_AUTH_TOKEN` is set, the browser shows a native HTTP
Basic dialog on first visit. Leave the username blank (or any value)
and paste the token as the password. The server sets a cookie that
persists for 30 days so subsequent navigation doesn't re-prompt.

Browsers cannot pass a Bearer header in normal navigation, which is
why the `/web` routes use HTTP Basic rather than Bearer. MCP and hook
clients continue to use `Authorization: Bearer <token>`.

## Configuring the CLI

The `ai-memory` binary is a thin HTTP client. It never opens the
wiki or SQLite directly - every state-touching command goes through
the running server, which is the sole writer.

Configuration is two environment variables, both **optional**:

| Variable | Default | When to set it |
|---|---|---|
| `AI_MEMORY_SERVER_URL` | `http://127.0.0.1:49374` | When the server runs somewhere other than this machine (e.g. a homelab at `http://192.168.0.90:49374`). |
| `AI_MEMORY_AUTH_TOKEN` | unset (no auth) | When the server has bearer auth enabled - see [Security](#security). |

For the **single-laptop local case** (scenarios A/B) you don't need
either env var: the CLI talks to the loopback server and just works.
Scenario B (loopback + token) only requires `AI_MEMORY_AUTH_TOKEN` in
the env.

For a **remote / homelab** server (scenarios C/D), set both in your
shell rc (or a `.envrc` if you use direnv):

```bash
export AI_MEMORY_SERVER_URL="http://192.168.0.90:49374"
export AI_MEMORY_AUTH_TOKEN="b9a5075d…"   # only when server has auth enabled
```

Explicit flags (`--auth-token`, `--server-url`) on `install-mcp` /
`install-hooks` override env vars when both are set - useful when
you're generating configs for a client that talks to a different
server than the CLI default.

The `init`, `serve`, `install-*`, `generate-auth-token`, and
`setup-agent` subcommands don't need these env vars - they either
set up local files or start the server itself.

## How it works in practice

You mostly don't think about it. Hooks capture every prompt + tool
call + session boundary automatically. The agent gains awareness of
prior work without you typing anything special. A few patterns are
worth knowing:

### Cross-agent handoff

```
$ claude
> "Working on the auth refactor. JWT rotation story is broken; trying
   session cookies as an alternative."
[work for an hour]
> /exit

$ codex   # in the same directory, hours or days later
[SessionStart hook fetches the handoff; the next agent sees it.]
> "Picking up: you were investigating session cookies as an
   alternative to broken JWT rotation. Continuing?"
```

You did nothing special. Handoff created automatically on Claude
Code's session-end, surfaced automatically on Codex's session-start.

### Compaction recovery

When Claude Code or Codex compact their working context, the
`PreCompact` hook fires and ai-memory writes a fresh
`sessions/<id>.md` page summarising the session so far. After
compaction, the agent can recover the summary via `memory_recent`
even though its raw history is gone.

### Adopting ai-memory mid-project: bootstrap

If you're installing ai-memory in a project you've been working on
for months, the wiki starts empty and the first few sessions are
net-zero - you're populating, not retrieving. `ai-memory bootstrap`
solves that by LLM-summarising your existing `git log`, README,
`docs/`, and module-level doc-comments into seed wiki pages.

```bash
# Run from your project's repo root. The CLI collects sources locally
# (git log, README, docs/, module headers) and POSTs them to the server
# at AI_MEMORY_SERVER_URL, where the LLM call and wiki writes happen.
# Requires an LLM provider configured on the server. Budget caps at
# 50k input tokens (~$0.05 with Claude Haiku 4.5).
export AI_MEMORY_SERVER_URL="http://localhost:49374"
ai-memory bootstrap
```

The workspace defaults to `default` and the project defaults to the
current directory's basename - that's almost always what you want, so
omit `--workspace` / `--project` unless you're deliberately overriding.

Bootstrap produces a per-project `bootstrap.md` manifest (under
`<wiki>/<workspace>/<project>/`) listing every page generated + a
one-paragraph rationale. Run with `--dry-run` first to preview which
sources would be sent without paying for the LLM call. Re-running on
the same project requires `--force`.

See [`docs/install.md`](docs/install.md#bootstrap-mid-project) for
the full flag reference + per-source priority order.

### Spelunking your own history

```bash
docker exec ai-memory ls /data/wiki/sessions/
docker exec ai-memory cat /data/wiki/sessions/<uuid>.md

# Open in Obsidian / any markdown viewer:
docker cp ai-memory:/data/wiki ./my-ai-memory-wiki

# Time-travel:
docker exec ai-memory git -C /data/wiki log --oneline
```

### Browse the wiki in a browser

For a more navigable view, start the server with `--enable-web` and
open `http://<host>:49374/web` in any browser. Project-list homepage,
per-project page tree with breadcrumbs, rendered markdown with syntax
highlighting and metadata (tier, kind, pinned, supersedes chain),
plus FTS5 search - all read-only, no editing. Light/dark theme
follows your OS setting via `prefers-color-scheme`.

![Project list homepage with the LLM-optimised banner; four projects (distrobox-gaming, ai-memory, nes-to-sms, .config) shown as cards with page counts + last activity.](docs/web-projects-home.png)

Drill into any project for the folder tree (`concepts/`, `decisions/`,
`gotchas/`, `sessions/`) and the recent-activity column:

![distrobox-gaming project view: left sidebar shows the folder tree grouped by kind; right column lists recent pages with kind badges and relative timestamps.](docs/web-project-view.png)

```bash
ai-memory serve --transport http --bind 127.0.0.1:49374 --enable-web
# or, if you run via docker compose, add it to the command line in
# docker-compose.yml: ["serve", "--transport", "http", "--bind",
# "0.0.0.0:49374", "--enable-web"]
```

The web routes are mounted at `/web` on the same axum server as the
MCP endpoint. When the server has bearer auth enabled, the browser
shows a native HTTP Basic dialog on first visit - leave the username
blank (or any value) and paste the token as the password; the cookie
persists for 30 days so subsequent navigation doesn't re-prompt.
Loopback-bound servers with no token need no credentials at all. See
[Security → Browser access to /web](#browser-access-to-web) above.

### Rules vs facts - ai-memory tells you when something belongs in CLAUDE.md

When you type something like "don't forget to never add a function
without a unit test", that's a **durable project rule**, not a
session-level observation. Rules need to fire on every relevant
action - that's what your project's `CLAUDE.md` / `AGENTS.md` is for
(it's loaded into the agent's system prompt every turn), while
ai-memory queries only fire when the agent thinks to call them.

The consolidator now classifies each compiled observation as
`decision | fact | rule | gotcha`. Rule-tagged pages are auto-routed
to `wiki/_rules/<slug>.md`, and the next time you run `memory_lint`
the agent sees a suggestion:

> **rule_suggestion**: Page `_rules/never-ship-code-without-test.md`
> looks like a durable project rule. Consider copying it into your
> project's CLAUDE.md / AGENTS.md so the agent sees it on every
> turn, not just when it remembers to call memory_query.

ai-memory never edits your `CLAUDE.md` itself - the suggestion is
the whole UX. You copy what's useful, ignore what isn't.

### When the agent reaches for memory

Hooks handle *capture* (every prompt + tool call + session boundary)
and *handoff resume* (SessionStart auto-fetches pending handoffs)
without you typing anything. **Proactive querying** - the agent
reaching into the wiki on its own - depends on the agent knowing
*when* to call which tool. Concrete patterns once the routing
snippet is installed (see "Nudge the agent" below):

| You say… | Agent calls… | Effect |
|---|---|---|
| "Have we discussed X?" / "search memory for Y" | `memory_query` | FTS5 + link-neighbour RRF over compiled wiki pages, optional vector RRF, and bounded raw observation fallback when pages miss. |
| (before proposing architecture) implicit | `memory_query` | The routing snippet tells the agent to check prior decisions / gotchas before proposing anything, so you don't get contradicted-by-its-own-history suggestions. |
| "Catch me up" / "I've been away" | `memory_explore` | Prose digest. Verbosity auto-scales to time-since-last-activity: < 1 h → one line, > 30 days → full catchup. |
| "Where did we leave off?" | (handoff already prepended) | SessionStart hook already prepended the handoff before your first prompt; the agent just reads from that block. |
| "Save context for the next session" | `memory_handoff_begin` | Writes a terse handoff with `open_questions` + `next_steps` for whichever agent opens this project next. |
| "Consolidate this session" | `memory_consolidate` | Manual trigger of what session-end normally does automatically (compile observations into wiki pages). |
| "Audit the wiki" / "any contradictions?" | `memory_lint` | Runs the stale-page / contradiction / rule-suggestion pass. |
| "How big is the wiki?" / "stats?" | `memory_status`, `memory_briefing` | Counts + recent activity windows. |

The mapping is in the [routing snippet](#nudge-the-agent-to-use-memory-proactively)
the next section installs.

### Nudge the agent to *use* memory proactively

The capture side works automatically. For the *query* side, the
agent needs a routing table in the project's rules file
(`CLAUDE.md` for Claude Code; `AGENTS.md` for Codex / OpenCode /
Cursor / Gemini CLI). Two ways to install it - pick whichever's
easier in the moment:

**From the agent** (no terminal needed):

> "Install ai-memory routing into this project."

The agent calls the `memory_install_self_routing` MCP tool, gets
back the canonical snippet + a per-agent filename map, then uses
its own Write/Edit tool to land the block in the right rules file
(Claude Code → CLAUDE.md, everyone else → AGENTS.md). Idempotent:
the block is wrapped in `<!-- ai-memory:start -->` /
`<!-- ai-memory:end -->` markers so the agent replaces in place on
re-runs.

**From the terminal**:

```bash
ai-memory install-instructions          # auto-detects CLAUDE.md vs AGENTS.md
ai-memory install-instructions --target AGENTS.md   # force a specific file
ai-memory install-instructions --print              # preview without writing
```

The CLI's auto-detect: if `$PWD/CLAUDE.md` exists, extend that;
if `$PWD/AGENTS.md` exists, extend that; if both exist, write to
both (multi-agent project - keep both files in sync); if neither
exists, create `CLAUDE.md` and print a hint about `--target
AGENTS.md` for non-Claude agents.

Both paths produce the same block. Both replace existing markered
blocks in place rather than duplicating, so you can re-run safely
whenever the snippet evolves (e.g. when a new MCP tool ships).

## LLM provider - recommended defaults

You can run ai-memory entirely without an LLM (FTS5 search +
rule-based summaries, $0). When you *do* configure one, the
options below are ranked by fitness for ai-memory's
consolidation workload - see
[`docs/llm-provider-comparison.md`](docs/llm-provider-comparison.md)
for the empirical writeup behind this ranking.

> **TL;DR.** Use **Claude Haiku 4.5** as your default. Switch
> to **GPT-5.4-mini** if you want the same quality cheaper +
> faster. Switch to **qwen3:32b on Ollama** if you have a
> local LLM server and prefer $0 / fully-self-hosted. The
> three are interchangeable; pick once and forget.

### Option 1 - Claude Haiku 4.5 *(recommended default)*

Best balance of speed (~7 s), restraint, and classification
quality. The only model that consistently classifies durable
project rules as `kind: rule` so the consolidator auto-routes
them to `_rules/<slug>.md`. ~$0.02 per consolidation; cost
is negligible for personal use.

```bash
AI_MEMORY_LLM_PROVIDER=anthropic
AI_MEMORY_LLM_MODEL=claude-haiku-4-5
ANTHROPIC_API_KEY=sk-ant-…
```

Or via OpenRouter (handy if you already have an OpenRouter
account and want one bill):

```bash
AI_MEMORY_LLM_PROVIDER=openai-compat
AI_MEMORY_LLM_BASE_URL=https://openrouter.ai/api/v1
AI_MEMORY_LLM_MODEL=anthropic/claude-haiku-4.5
LLM_API_KEY=sk-or-v1-…
```

### Option 2 - OpenAI GPT-5.4-mini *(cheaper alternative)*

~5× cheaper than Haiku, ~2× faster (~4 s avg). Same parse
reliability, same faithfulness. One known weakness: mild
over-classification on trivial sessions (will sometimes
manufacture an extra `decisions/` page for a thin
session). Acceptable for most users.

```bash
AI_MEMORY_LLM_PROVIDER=openai
AI_MEMORY_LLM_MODEL=gpt-5.4-mini
OPENAI_API_KEY=sk-…
```

Or via OpenRouter:

```bash
AI_MEMORY_LLM_PROVIDER=openai-compat
AI_MEMORY_LLM_BASE_URL=https://openrouter.ai/api/v1
AI_MEMORY_LLM_MODEL=openai/gpt-5.4-mini
LLM_API_KEY=sk-or-v1-…
```

### Option 3 - Google Gemini 2.5 Flash *(generous free tier)*

Hosted by Google, with a free tier that's enough for most
personal consolidation loads. Structured-output is native
(`responseSchema` over OpenAPI 3 subset); the client inlines
`$ref`s and strips Draft-2020-12 keywords Gemini rejects, so
schemars-derived schemas just work. Use `gemini-2.5-flash`
for the speed/cost balance; avoid the *-thinking* variants
(see "What we don't recommend" below).

Grab a key from https://aistudio.google.com/apikey, then:

```bash
AI_MEMORY_LLM_PROVIDER=gemini
AI_MEMORY_LLM_MODEL=gemini-2.5-flash      # optional; this is the default
GEMINI_API_KEY=AIza…                      # GOOGLE_API_KEY also accepted
```

### Option 4 - Local Ollama qwen3:32b *(free / self-hosted)*

$0 per consolidation. Requires a machine with at least ~24 GB
of unified or VRAM memory to keep the Q4_K_M weights warm
(~20 GB) plus headroom. Strix Halo / Apple Silicon / a
recent NVIDIA card all work. Latency is ~90 s but
consolidation is a background job - users never see it.

One-time setup on the Ollama host:

```bash
ollama pull qwen3:32b
ollama pull nomic-embed-text   # for embeddings; see below
# Recommended Ollama env:
#   OLLAMA_KEEP_ALIVE=20m       (keep models warm between consolidations)
#   OLLAMA_FLASH_ATTENTION=1
#   OLLAMA_KV_CACHE_TYPE=q8_0   (halves KV memory)
```

ai-memory env:

```bash
AI_MEMORY_LLM_PROVIDER=openai-compat
AI_MEMORY_LLM_BASE_URL=http://<ollama-host>:11434/v1
AI_MEMORY_LLM_MODEL=qwen3:32b
LLM_API_KEY=ollama-local                  # any non-empty value; Ollama doesn't validate
```

If you bind ai-memory to a non-loopback address, also set
`AI_MEMORY_ALLOWED_HOSTS` - see [Security → DNS-rebinding guard](#dns-rebinding-guard-ai_memory_allowed_hosts).

### What we don't recommend

- **Claude Sonnet 4.5** - strictly dominated by Haiku for
  this task: same parse reliability, 3× cost, hallucinated
  details before the prompt was tightened. Use it only if
  you specifically need extended reasoning (e.g. cross-page
  lint sweeps).
- **Reasoning-mode models** (Kimi-K2.6, Claude with extended
  thinking enabled, GPT-o3, Gemini "thinking" variants) -
  these models burn `max_tokens` budget on internal
  reasoning before emitting visible content; with the
  strict-JSON consolidation prompt they hang or emit empty
  responses. If you must use one, turn reasoning off.

### Embedding provider

The LLM provider drives consolidation + lint. Embeddings are
a *separate* concern (hybrid retrieval over the wiki - BM25
+ vector RRF). Defaults when `AI_MEMORY_EMBEDDING_PROVIDER`
is set:

| Provider | Default model | Dim |
|---|---|---|
| `openai` | `text-embedding-3-small` | 1536 |
| `voyage` | `voyage-3` | 1024 |

For the local stack, point the OpenAI embedder at Ollama:

```bash
AI_MEMORY_EMBEDDING_PROVIDER=openai
AI_MEMORY_EMBEDDING_BASE_URL=http://<ollama-host>:11434/v1
AI_MEMORY_EMBEDDING_MODEL=nomic-embed-text
AI_MEMORY_EMBEDDING_DIM=768
OPENAI_API_KEY=ollama-local
```

Skipping the embedding provider entirely is fine -
`memory_query` still uses FTS5, link-neighbour expansion, and raw
fallback; you just lose vector re-ranking.

Per-tier feature breakdown + the openai-compat / Ollama setup
is in [`docs/install.md`](docs/install.md#llm-provider-tiers).

## Architecture in 60 seconds

A single Rust binary, optionally containerised. Runs as an
[MCP](https://modelcontextprotocol.io/) server over stdio + HTTP.
Owns a data directory containing:

```
<data_dir>/
├── wiki/    # markdown source of truth (git-versioned)
├── raw/     # immutable session log archive
├── db/      # SQLite (FTS5 + page_embeddings) — derived index
├── models/  # reserved for local embedding model (v0.3+)
└── logs/    # rolling daily tracing output
```

Agent lifecycle hooks fire-and-forget POST to the server's HTTP
ingress. The server queues writes through a single SQLite writer
(no `database is locked`). On session end an optional LLM-driven pass
rewrites pages atomically with supersession (`is_latest=false` +
`supersedes` chain) and opens a typed handoff for the next agent.
The scheduled retention sweep decays unused episodic content while
semantic concept pages compound forever; pinned pages and `_slots/`
memory slots are exempt. Retrieval is FTS5 + link-neighbour RRF by
default; when an embedder is configured, vector RRF over
`page_embeddings` joins the ranks. If compiled wiki search misses,
bounded raw observation FTS provides exact-detail fallback.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the canonical
data-flow diagram + crate breakdown + cross-cutting invariants.

## Docs

| File | What it is |
|---|---|
| [`docs/install.md`](docs/install.md) | **Installation cookbook.** Every agent CLI, every alternative (curl, source build, no-docker, no-auth), and the server-on-a-different-machine (homelab/LAN) walkthrough. Read after the Quick start if your setup doesn't match the happy path. |
| [`docs/windows.md`](docs/windows.md) | Windows install modes: full WSL2, native Windows with Docker Desktop, native source builds, and current hook/MCP harness caveats. |
| [`docs/mcp-install.md`](docs/mcp-install.md) | Per-client MCP config snippets (Cursor, Claude Desktop, Gemini CLI, OpenClaw, OMP). |
| [`docs/deploy.md`](docs/deploy.md) | Homelab deploy: bin/deploy, bearer-token auth, TLS via cloudflared. |
| [`docs/lifecycle-ops.md`](docs/lifecycle-ops.md) | **Read before running purge / rename / backup / restore / reset.** Safety matrix for the state-touching commands, per-project disk layout (how isolation actually works), and operator workflows for "fresh start", "snapshot before risky op", "drop one project". |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Operational summary: data flow, crate layout, cross-cutting invariants, schema. |
| [`docs/design-decisions.md`](docs/design-decisions.md) | The full v1 spec. |
| Research docs under `docs/` | Karpathy LLM Wiki notes, agentmemory / basic-memory / cognee deep-dives, lessons-learned from upstream issues. |

## Influences and prior art

- **[Karpathy LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)** - the compile-not-retrieve pattern.
- **[agentmemory](https://github.com/rohitg00/agentmemory)** - most of the right ideas; this project is the Rust successor.
- **[basic-memory](https://github.com/basicmachines-co/basic-memory)** - the markdown-on-disk source-of-truth model.
- **[cognee](https://github.com/topoteretes/cognee)** - pipeline composition and triplet embeddings.
- **[A-MEM](https://arxiv.org/abs/2502.12110)** - Zettelkasten-style atomic notes with link evolution.

## License

MIT - see [LICENSE](LICENSE).

## Acknowledgements

This codebase is being built collaboratively with Claude Code
(Anthropic Claude Opus 4.7) following the plan documented in
`docs/design-decisions.md`.

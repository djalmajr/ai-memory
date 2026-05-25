# Installation cookbook

The [README quick-start](../README.md#quick-start) covers the happy
path (docker + Claude Code). This page covers everything else:

- [Server on a different machine](#server-on-a-different-machine)
  (homelab, LAN box, remote server)
- [Configuring other agent CLIs](#configuring-other-agent-clis)
  (Codex, OpenCode, OMP, Cursor, Claude Desktop, Gemini CLI, OpenClaw)
- [Installing hooks without docker](#installing-hooks-without-docker)
  (curl-based installer)
- [Running ai-memory without docker](#running-ai-memory-without-docker)
  (cargo install, building from source)
- [LLM provider tiers + self-hosted Ollama](#llm-provider-tiers)
- [Subcommand reference](#subcommand-reference)
- [Operating without auth](#operating-without-auth) (local-only)

> **Shorthand.** Most snippets use `$TOKEN` and `homelab:49374`. If
> you're following along verbatim:
> ```bash
> export TOKEN=$(docker run --rm akitaonrails/ai-memory:latest generate-auth-token)
> ```
> and replace `homelab` with `localhost` if the server runs on the
> same machine as the agent CLI.

---

## Server on a different machine

When the ai-memory server runs on a LAN box (homelab, headless server)
and you use Claude Code / Codex / etc. on a laptop:

### Server side (the homelab host)

```bash
docker run -d --name ai-memory \
    --restart unless-stopped \
    -p 0.0.0.0:49374:49374 \
    -v ai-memory-data:/data \
    -e AI_MEMORY_AUTH_TOKEN="$TOKEN" \
    -e AI_MEMORY_ALLOWED_HOSTS="<server-ip>,localhost,127.0.0.1" \
    -e AI_MEMORY_LLM_PROVIDER=anthropic \
    -e ANTHROPIC_API_KEY=sk-ant-... \
    akitaonrails/ai-memory:latest
```

See [Security](../README.md#security) in the README for why
`AI_MEMORY_AUTH_TOKEN` and `AI_MEMORY_ALLOWED_HOSTS` are both
required for any non-loopback bind.

### Client side (the laptop)

```bash
export AI_MEMORY_SERVER_URL="http://<server-ip>:49374"
export AI_MEMORY_AUTH_TOKEN="$TOKEN"

ai-memory install-mcp   --client claude-code --apply \
    --server-url "http://<server-ip>:49374/mcp"
ai-memory install-hooks --agent  claude-code --apply \
    --server-url "http://<server-ip>:49374"
```

The CLI commands (`bootstrap`, `status`, `search`, `lint`, etc.) inherit the
two env vars automatically.

---

## Configuring other agent CLIs

> `install-mcp --server-url` takes the MCP endpoint **including** `/mcp`
> (e.g. `http://homelab:49374/mcp`) — the rendered client config expects the
> full MCP URL. `install-hooks --server-url` takes the bare server **origin**
> (e.g. `http://homelab:49374`) — hook scripts append `/hook`, `/handoff`,
> etc. themselves.

Each agent CLI needs two things:

1. **MCP registration** - so the agent can call `memory_query`,
   `memory_recent`, `memory_handoff_accept`.
2. **Lifecycle hooks** - so the server auto-captures session events.
   Without this, the agent can still query memory but capture
   becomes manual.

Claude Desktop and OpenClaw are MCP-only today. Claude Code, Codex,
OpenCode, OMP, Cursor, and Gemini CLI have lifecycle capture paths
through `install-hooks`.

> **Two-step hook install pattern.** Claude Code, Codex, Cursor, and
> Gemini CLI use shell/PowerShell hook scripts: (1) `docker cp` the
> bundled scripts to your home dir, (2) `docker run --rm install-hooks`
> to render the config snippet.
> OpenCode and OMP are different: they use generated TypeScript
> plugin/extension files, so no shell-script extraction is needed for
> those clients.

### OpenAI Codex

```bash
# MCP snippet (merge into ~/.codex/config.toml):
docker run --rm akitaonrails/ai-memory:latest \
    install-mcp --client codex \
    --server-url "http://homelab:49374/mcp" \
    --auth-token "$TOKEN"

# Hooks — extract scripts + render config:
docker cp ai-memory:/usr/local/share/ai-memory/hooks ~/.ai-memory/
docker run --rm akitaonrails/ai-memory:latest \
    install-hooks --agent codex \
        --hooks-dir ~/.ai-memory/hooks \
        --server-url "http://homelab:49374" \
        --auth-token "$TOKEN"
```

### OpenCode

```bash
docker run --rm akitaonrails/ai-memory:latest \
    install-mcp --client opencode \
    --server-url "http://homelab:49374/mcp" \
    --auth-token "$TOKEN"

# Plugin — write to ~/.config/opencode/plugins/ai-memory.ts.
# If you have the local wrapper installed, prefer `--apply`:
ai-memory install-hooks --agent opencode --apply \
    --server-url "http://homelab:49374" \
    --auth-token "$TOKEN"

# Docker-only preview path; redirect only if you want to write the file yourself:
docker run --rm akitaonrails/ai-memory:latest \
    install-hooks --agent opencode \
    --server-url "http://homelab:49374" \
    --auth-token "$TOKEN"
```

Restart OpenCode after installing or changing the plugin; plugins are
loaded at startup.

### Oh My Pi / OMP

```bash
docker run --rm akitaonrails/ai-memory:latest \
    install-mcp --client pi \
    --server-url "http://homelab:49374/mcp" \
    --auth-token "$TOKEN"

# Extension — write to ~/.omp/agent/extensions/ai-memory.ts.
# If you have the local wrapper installed, prefer `--apply`:
ai-memory install-hooks --agent omp --apply \
    --server-url "http://homelab:49374" \
    --auth-token "$TOKEN"
```

Restart OMP after installing or changing the extension; extensions are
loaded at startup. The ai-memory CLI accepts `--client pi` /
`--client omp` for MCP and `--agent omp` / `--agent pi` for hooks;
all four target the same current Oh My Pi integration surface.

### Bind mounts vs docker cp

The `setup-agent` subcommand does the extract + render in one shot
using a bind mount:

```bash
docker run --rm -v "$HOME/.ai-memory:/host" \
    akitaonrails/ai-memory:latest \
    setup-agent --agent claude-code --to /host/hooks \
        --host-prefix "$HOME/.ai-memory/hooks" \
        --server-url "http://homelab:49374" --auth-token "$TOKEN"
```

This works cleanly when the container user's UID matches the host
user's UID (e.g. the homelab where both are 1000). It **fails on
rootless Docker** and on hosts with `userns-remap` enabled - the
container can't write to a host directory that belongs to a UID
outside the user-namespace mapping.

The `docker cp` pattern recommended above sidesteps all of that
because `docker cp` is mediated by the docker daemon and outputs
files owned by the user running the command. Prefer it as the
default; reach for `setup-agent` only when your docker setup is
known not to remap UIDs.

### Cursor, Gemini CLI, Claude Desktop, OpenClaw

See [**`docs/mcp-install.md`**](mcp-install.md) for the per-client MCP
config file path and snippet, or one-shot it via:

```bash
docker run --rm akitaonrails/ai-memory:latest \
    install-mcp --client cursor          --auth-token "$TOKEN" \
    --server-url "http://homelab:49374/mcp"

docker run --rm akitaonrails/ai-memory:latest \
    install-mcp --client claude-desktop  --auth-token "$TOKEN" \
    --server-url "http://homelab:49374/mcp"

docker run --rm akitaonrails/ai-memory:latest \
    install-mcp --client gemini-cli      --auth-token "$TOKEN" \
    --server-url "http://homelab:49374/mcp"

docker run --rm akitaonrails/ai-memory:latest \
    install-hooks --agent gemini-cli     --auth-token "$TOKEN" \
    --server-url "http://homelab:49374"

docker run --rm akitaonrails/ai-memory:latest \
    install-mcp --client openclaw        --auth-token "$TOKEN" \
    --server-url "http://homelab:49374/mcp"
```

Cursor and Gemini CLI support both `install-mcp` and `install-hooks`.
Claude Desktop and OpenClaw are MCP-only here, so you'll need to nudge
the model to call `memory_query` / `memory_handoff_accept` itself. For
clients with `install-hooks` support, the capture path handles handoff
injection at session start.

---

## Installing hooks without docker

If you only need to use ai-memory *from* a machine (i.e. that
machine doesn't run the server), the curl installer pulls the seven
hook scripts straight from GitHub:

```bash
curl -sSL https://raw.githubusercontent.com/akitaonrails/ai-memory/main/scripts/install-hooks.sh \
    | bash -s -- --agent claude-code

# Then render the JSON config (still wants `ai-memory` somewhere —
# either via docker as a one-shot, or installed locally):
docker run --rm akitaonrails/ai-memory:latest \
    install-hooks --agent claude-code \
        --hooks-dir "$HOME/.ai-memory/hooks" \
        --server-url "http://homelab:49374" \
        --auth-token "$TOKEN"
```

The curl script installer supports `--agent claude-code|codex|opencode|omp|pi`
and `--to <dir>`; `--help` prints the full flag list. OpenCode and OMP do
not need this script installer because `install-hooks --agent opencode`
and `install-hooks --agent omp` / `--agent pi` generate TypeScript files instead.

This path is friction-free when:
- You have curl + bash but not docker
- You don't need to run a local ai-memory server (you're a client of
  a homelab/remote ai-memory)

---

## Running ai-memory without docker

Most users should stick to the docker wrapper from the Quick start. Build from
source only when hacking on ai-memory itself or running on a platform docker
doesn't support.

```bash
git clone https://github.com/akitaonrails/ai-memory ~/.ai-memory
cd ~/.ai-memory
cargo build --release --workspace
./target/release/ai-memory init                       # one-time
./target/release/ai-memory serve --transport http \
    --bind 127.0.0.1:49374                            # MCP + hook HTTP server
```

Data dir defaults to `~/.local/share/ai-memory` on Linux,
`~/Library/Application Support/ai-memory` on macOS, and the platform
local-data directory on Windows, typically
`%LOCALAPPDATA%\ai-memory`. Override with `AI_MEMORY_DATA_DIR=/path`.
To require bearer-token auth, set `AI_MEMORY_AUTH_TOKEN` in the
server's environment.

On Windows, see [`docs/windows.md`](windows.md). The short version: run
the install commands from the same environment that launches the agent.
WSL2-launched agents need WSL paths and POSIX `.sh` hooks; native Windows
agents need Windows paths and PowerShell `.ps1` hooks.

When run from source, `install-hooks` finds the bundled scripts in
the repo's `hooks/` automatically:

```bash
./target/release/ai-memory install-hooks --agent claude-code --auth-token "$TOKEN"
```

(No need for `setup-agent` in this case - the scripts already live
at the right host path.)

---

## LLM provider tiers

ai-memory works in three intensity tiers:

| Tier | What you get | Env vars | Cost |
|---|---|---|---|
| **Zero-LLM** (default) | FTS5 search, rule-based session summaries, auto-handoffs from prompt + tool-call history | (none) | $0 |
| **+ LLM consolidation** | LLM rewrites session pages as coherent narratives; PreCompact checkpoints; LLM-driven contradiction lint | `AI_MEMORY_LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` | ~$0.01–0.05 / session |
| **+ Hybrid retrieval** | RRF over FTS5 + vector cosine similarity. Better recall on paraphrased queries | `AI_MEMORY_EMBEDDING_PROVIDER=openai` + `OPENAI_API_KEY` | ~$0.0001 / page on backfill |

### Recommended models (chosen as defaults)

If you set only the provider, ai-memory picks a sensible default:

| Setting | Default | Why |
|---|---|---|
| `AI_MEMORY_LLM_PROVIDER=anthropic` | `claude-haiku-4-5` | **Recommended default.** Best balance of speed, restraint, and classification quality. Not a reasoning model. Consistently classifies durable project rules as `kind: rule`. |
| `AI_MEMORY_LLM_PROVIDER=openai` | `gpt-5.4-mini` | Cheaper + faster alternative. Same parse reliability; mild over-classification on thin sessions. |
| `AI_MEMORY_LLM_PROVIDER=gemini` | `gemini-2.5-flash` | Google's hosted option with a generous free tier. ai-memory disables Gemini 2.5 Flash's default dynamic thinking so hidden thought tokens do not truncate strict JSON. Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). |
| `AI_MEMORY_EMBEDDING_PROVIDER=openai` | `text-embedding-3-small` (1536-dim) | 5× cheaper than `-3-large` with marginal recall loss. |
| `AI_MEMORY_EMBEDDING_PROVIDER=voyage` | `voyage-3` (1024-dim) | Voyage's current general-purpose recommendation. |

> **What we don't recommend:** reasoning-mode models (Claude with extended
> thinking, GPT-o3, Gemini "thinking" variants) — they burn token budget on
> internal reasoning and hang or emit empty responses with the strict-JSON
> consolidation prompt. Turn reasoning off if you must use one.

### Self-hosted LLMs (Ollama / vLLM / LM Studio / OpenRouter)

```bash
docker run -d --name ai-memory \
    -p 49374:49374 \
    -v ai-memory-data:/data \
    -e AI_MEMORY_AUTH_TOKEN="$TOKEN" \
    -e AI_MEMORY_LLM_PROVIDER=openai-compat \
    -e AI_MEMORY_LLM_BASE_URL=http://host.docker.internal:11434/v1 \
    -e AI_MEMORY_LLM_MODEL=qwen2.5-coder:14b \
    akitaonrails/ai-memory:latest
```

There is no safe default model for `openai-compat`; the env var is
required. For OpenRouter (Kimi, DeepSeek, etc.):

```bash
-e AI_MEMORY_LLM_PROVIDER=openai-compat
-e AI_MEMORY_LLM_BASE_URL=https://openrouter.ai/api/v1
-e AI_MEMORY_LLM_MODEL=moonshotai/kimi-k2.6
-e LLM_API_KEY=sk-or-v1-...
```

---

## Subcommand reference

Two ways to invoke a subcommand against the docker deploy:

```bash
# A) Against the running container (stateful: status, search, backup,
#    forget-sweep, lint, embed).
docker exec ai-memory ai-memory status --json
docker exec ai-memory ai-memory search "karpathy"
docker exec ai-memory ai-memory backup --to /data/snapshot.tar.gz

# B) One-shot, no running container needed (pure-stdout: generate-
#    auth-token, install-mcp, install-hooks, setup-agent, llm-test).
docker run --rm akitaonrails/ai-memory:latest generate-auth-token
docker run --rm akitaonrails/ai-memory:latest install-mcp --client cursor
docker run --rm akitaonrails/ai-memory:latest --help     # full subcommand tree
```

| Subcommand | Pattern | What it does |
|---|---|---|
| `serve` | `docker compose up -d` (already done) | Run the HTTP MCP server |
| `status` | `docker exec` | Counts, paths, and derived-index diagnostics |
| `search "<query>"` | `docker exec` | Wiki search with FTS5 + graph/vector RRF |
| `write-page` | `docker exec` | Manual page write (atomic + indexed) |
| `backup --to` / `restore --from` | `docker exec` | Snapshot or restore the data dir |
| `forget-sweep` / `lint` / `embed` | `docker exec` | Manual maintenance; sweep + lint also run on the server schedule by default |
| `commit -m "…"` | `docker exec` | Stage + commit the wiki tree |
| `reset --confirm` | `docker exec` | Wipe data (refuses while siblings alive) |
| `generate-auth-token` | `docker run --rm` | Print a random hex bearer token |
| `install-mcp --client` | `docker run --rm` | MCP-config snippet per client |
| `install-hooks --agent` | `docker run --rm` | Hook-config snippet for an existing hooks dir |
| `setup-agent --agent --to --host-prefix` | `docker run --rm -v` | Extract bundled scripts + print config (one-shot) |
| `llm-test --provider …` | `docker run --rm -e …` | Smoke-test an LLM provider |

Data dir inside the container is `/data` (mounted via the compose
volume). Outside docker, override with `AI_MEMORY_DATA_DIR=/path`.

Scheduled maintenance is configured in `[maintenance]` in `config.toml`.
By default, rule-based lint and forget sweep run daily outside hook
latency. Embedding backfill is supported but defaults to off because it
can call a paid provider; enable it with
`embedding_backfill_interval_secs` after configuring an embedder.

---

## Bootstrap mid-project {#bootstrap-mid-project}

When you adopt ai-memory in a project that's already been around for
a while, the wiki starts empty. `ai-memory bootstrap` ingests the
project's existing history into seed pages so the first session has
warm context.

```bash
docker run --rm \
    -v ai-memory-data:/data \
    -v "$PWD:/repo" \
    -e AI_MEMORY_AUTH_TOKEN="$TOKEN" \
    -e AI_MEMORY_LLM_PROVIDER=anthropic \
    -e ANTHROPIC_API_KEY=sk-ant-... \
    akitaonrails/ai-memory:latest \
    bootstrap --repo-path /repo
```

**What gets ingested by default:**

| Source | Priority (dropped first when over budget) |
|---|---|
| `CLAUDE.md` / `AGENTS.md` (project rules) | never dropped |
| `README.md` at the repo root | very-late |
| `docs/**/*.md` | late |
| Substantive git commits (body >120 chars OR conventional-commit prefix) | mid |
| Module-level `//!` doc-comments in `**/*.rs` | first to drop |

**Flags:**

```
--repo-path <PATH>         (default: git rev-parse --show-toplevel)
--workspace <NAME>         (default: "default")
--project <NAME>           (default: "scratch")
--max-input-tokens N       (default: 50000)
--since "30 days ago"      (git log filter; supports "N days/months/years ago" + YYYY-MM-DD)
--exclude-git              (skip commit history)
--exclude-readme           (skip README)
--exclude-docs             (skip docs/**/*.md)
--exclude-code             (skip Rust module headers)
--dry-run                  (collect + estimate but don't call LLM or write)
--force                    (re-bootstrap, overwrites the prior manifest)
```

**Cost.** With Kimi 2.6 via OpenRouter ($0.73/$3.49 per M):
- 50k input tokens cap → ~$0.04 worst case input
- 1-2k generated tokens → ~$0.007 output
- Total: well under $0.20 per run.

**Idempotency.** The first run produces a per-project `bootstrap.md`
manifest (at `<wiki>/<workspace>/<project>/bootstrap.md`) listing every
page generated + a one-paragraph rationale. Re-running without `--force`
errors out. Delete the manifest (and the generated pages) if you want a
clean re-bootstrap.

**Dry-run first.** Always worth doing before the real call to see
which sources would actually be sent + how many tokens that
represents. Output is JSON to stdout.

```bash
docker run --rm -v "$PWD:/repo" ... bootstrap --repo-path /repo --dry-run
{
  "sources_collected": 117,
  "sources_sent": 22,
  "sources_dropped": 95,
  "estimated_input_tokens": 48760,
  "pages_written": [],
  "rationale": "(dry-run; LLM not invoked)",
  "dry_run": true
}
```

**Caveat: LLM-fabricated detail.** A bootstrap run can produce
plausible-but-wrong pages (the LLM doesn't know your project, it's
inferring from git history). The wiki is git-versioned precisely so
this is recoverable: review what landed, `docker exec ai-memory git
-C /data/wiki diff HEAD~1`, and revert if it's off.

## Operating without auth

For local-only / single-machine deploys you can skip the bearer
token:

```bash
docker run -d --name ai-memory \
    -p 127.0.0.1:49374:49374 \
    -v ai-memory-data:/data \
    akitaonrails/ai-memory:latest
```

Notice the bind: `127.0.0.1:49374`, not `0.0.0.0:49374`. This is the
critical pairing - **no bearer token AND loopback only** is the only
safe combination. The startup log will warn loudly if you bind to a
LAN address without setting `AI_MEMORY_AUTH_TOKEN`.

Then wire up the agent CLI. Both commands default to no auth and
`http://127.0.0.1:49374` - no extra flags needed for the local case:

```bash
ai-memory install-mcp   --client claude-code --apply
ai-memory install-hooks --agent  claude-code --apply
```

---

## See also

- [`docs/deploy.md`](deploy.md) - homelab deploy walkthrough
  (`bin/deploy`, cloudflared TLS, env-file management)
- [`docs/mcp-install.md`](mcp-install.md) - per-client MCP config
  reference for Cursor, Claude Desktop, Gemini CLI, OpenClaw, OMP
- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) - what's actually
  running inside ai-memory

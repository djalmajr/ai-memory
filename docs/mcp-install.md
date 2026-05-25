# MCP install guide - additional clients

> All snippets below default to `http://127.0.0.1:49374` (local server). For a
> remote server (homelab, LAN box) substitute the appropriate URL AND add an
> `Authorization: Bearer <token>` header to the `headers` block when bearer auth
> is enabled. The MCP wire protocol expects the `/mcp` path suffix on the URL.

> **Transport is stateless by default.** Since v0.1.2 the HTTP transport
> answers each request independently (plain JSON, no `Mcp-Session-Id`
> required), so any client that points a remote URL at `/mcp` — including
> OpenCode `type: "remote"` and plain `curl` — works without an
> `mcp-remote` stdio shim (issue #3). The `mcp-remote` bridge is still
> needed for **Claude Desktop** specifically, because its config only
> supports stdio servers — not because of session state. If you run a
> client that *requires* MCP session continuity or server-initiated SSE
> streams, start the server with `ai-memory serve --transport http
> --http-stateful` to restore rmcp's session mode.

This page documents how to register ai-memory as an MCP server with
agent CLIs beyond the README quick start.

Claude Code, OpenAI Codex, Cursor, Gemini CLI, OpenCode, and OMP have
automatic capture integrations (shell/PowerShell hooks for Claude Code /
Codex / Cursor / Gemini CLI, TypeScript plugin/extension files for
OpenCode / OMP) and are covered in the [main README](../README.md#quick-start).

Some clients on this page are **MCP-only**: they expose long-term
memory to their LLM via ai-memory's MCP tools (`memory_query`,
`memory_recent`, `memory_handoff_accept`, etc.), but they do not
auto-capture session events into ai-memory's `/hook` endpoint. The
trade-off:

| | What you get | What you don't get |
|---|---|---|
| **MCP only** | LLM can query the wiki, accept handoffs, run memory_consolidate | No automatic session-end summaries; no auto-handoff at session boundaries |
| **MCP + hooks** | All of the above *plus* every prompt/tool-call captured automatically; handoffs surface at SessionStart with no human prompting | - |

For MCP-only clients, you can still cover the session-boundary gap by
asking the LLM to call `memory_handoff_begin` manually before quitting.

> **One-shot tip:** every snippet below is also reachable from the
> CLI:
> ```bash
> ai-memory install-mcp --client gemini-cli   # or cursor / claude-desktop / openclaw / pi|omp
> ```

---

## Cursor

**Status:** ✅ MCP supported. ✅ Lifecycle hooks supported via
`ai-memory install-hooks --agent cursor --apply`.

**Config file:**
- Per-project: `.cursor/mcp.json` in the workspace root.
- Global: `~/.cursor/mcp.json`.

```json
{
  "mcpServers": {
    "ai-memory": {
      "url": "http://127.0.0.1:49374/mcp"
    }
  }
}
```

**Gotchas:**
- Cursor uses the `url` key for HTTP/SSE transports. Stdio uses
  `command` + `args` instead.
- After editing `mcp.json`, restart Cursor or toggle the server
  off+on in **Settings → MCP**. Live reload is in newer builds but
  still occasionally requires a manual restart.
- Source: <https://cursor.com/docs/context/mcp>

---

## Claude Desktop

**Status:** ✅ MCP supported (via stdio shim for HTTP). ❌ No lifecycle hooks.

**Config file:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: not officially distributed by Anthropic. Use Claude Code
  (terminal) instead.

**Important:** Claude Desktop's JSON config supports stdio MCP
servers only. To talk to ai-memory's HTTP endpoint, bridge through
the community [`mcp-remote`](https://www.npmjs.com/package/mcp-remote)
stdio shim. Requires Node.js installed on the same machine.

```json
{
  "mcpServers": {
    "ai-memory": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://127.0.0.1:49374/mcp"]
    }
  }
}
```

**Gotchas:**
- After editing the config, **fully quit and relaunch** Claude
  Desktop. "Check for Updates…" is not enough.
- If the MCP indicator doesn't appear after restart, check the logs:
  `~/Library/Logs/Claude/mcp*.log` (macOS) or `%APPDATA%\Claude\logs\`
  (Windows).
- Source: <https://modelcontextprotocol.io/quickstart/user>

---

## Gemini CLI

**Status:** ✅ MCP supported. ✅ Lifecycle hooks supported via
`ai-memory install-hooks --agent gemini-cli --apply`.

**Config file:**
- User: `~/.gemini/settings.json`
- Project: `.gemini/settings.json`

Gemini CLI uses `httpUrl` (not `url`) for streamable-HTTP MCP
endpoints. The `timeout` is in milliseconds.

```json
{
  "mcpServers": {
    "ai-memory": {
      "httpUrl": "http://127.0.0.1:49374/mcp",
      "timeout": 5000
    }
  }
}
```

**Hooks:**

```bash
ai-memory install-hooks --agent gemini-cli --apply
```

Gemini CLI's lifecycle event names differ from Claude Code's, so use
`install-hooks --agent gemini-cli` rather than copying another agent's
settings. ai-memory maps Gemini's `SessionStart`, `SessionEnd`,
`BeforeTool`, `AfterTool`, and `PreCompress` events to the shared hook
capture path; `SessionStart` also fetches pending handoffs.

**Gotchas:**
- Gemini supports stdio too via `command`/`args`, plus SSE via `url`.
  Only `httpUrl` covers streamable HTTP. Don't mix them in one entry.
- Restart the CLI session after changing `~/.gemini/settings.json` so
  both MCP servers and hooks are reloaded.
- Source: <https://github.com/google-gemini/gemini-cli/blob/main/docs/tools/mcp-server.md>

---

## OpenClaw

**Status:** ✅ MCP supported. ⚠️ Hooks exist (TypeScript plugins,
not POSIX shell). Hook integration would need a small TS plugin in
`integrations/openclaw/`; not shipped here.

**Config file:** `~/.openclaw/config.json` (the OpenClaw docs reference
this path indirectly; verify with your `openclaw config show`).

OpenClaw distinguishes transports explicitly. Use
`"transport": "streamable-http"` for ai-memory's HTTP endpoint.

```json
{
  "mcp": {
    "servers": {
      "ai-memory": {
        "url": "http://127.0.0.1:49374/mcp",
        "transport": "streamable-http"
      }
    }
  }
}
```

**Gotchas:**
- OpenClaw config keys under `mcp.*` hot-reload without a gateway
  restart. Hook config changes do require restarting the gateway.
- A TypeScript hook plugin that mirrors the Claude Code hook bundle
  would let OpenClaw auto-capture sessions; until that exists, treat
  OpenClaw as MCP-only.
- Sources: <https://docs.openclaw.ai/cli/mcp>,
  <https://docs.openclaw.ai/automation/hooks>

---

## Oh My Pi / OMP

**Status:** ✅ MCP supported via `install-mcp --client pi` or
`--client omp`. ✅ Lifecycle capture supported via
`ai-memory install-hooks --agent omp --apply` or `--agent pi --apply`.

**Config file:**
- User: `~/.omp/agent/mcp.json`
- Project: `.omp/mcp.json`

The current Oh My Pi package exposes the `omp` binary and native
`.omp` config directories. The ai-memory CLI accepts `--client pi` /
`--client omp` for MCP and `--agent omp` / `--agent pi` for lifecycle
capture; they are aliases for this same integration surface.

```json
{
  "mcpServers": {
    "ai-memory": {
      "type": "http",
      "url": "http://127.0.0.1:49374/mcp",
      "enabled": true
    }
  }
}
```

**Lifecycle extension:**

```bash
ai-memory install-hooks --agent omp --apply
# or:
ai-memory install-hooks --agent pi --apply
```

This writes `~/.omp/agent/extensions/ai-memory.ts`, which OMP discovers
as a direct TypeScript extension on startup. Restart `omp` after
installing or changing the file.

**Gotchas:**
- OMP extensions are TypeScript modules, not shell hooks; stdout is not
  used for context injection.
- The extension uses OMP lifecycle events for prompt/tool capture and
  `before_agent_start` to inject pending ai-memory handoffs.

---

## After registering MCP - verify it works

Regardless of which client you used, the first sanity check is the
same: ask the model to list its available MCP tools, or to call
`memory_status` explicitly.

```
You: List the MCP tools you can call. Use one of them to check
     ai-memory's status.

Model (any client): I can call: memory_query, memory_recent,
     memory_status, memory_briefing, memory_explore,
     memory_handoff_accept, memory_handoff_begin, memory_consolidate,
     memory_lint, memory_forget_sweep, memory_install_self_routing.
     memory_status reports: 0 pages, 0 observations, 0 sessions.
```

If the model doesn't see any of those tools, the MCP registration
isn't being picked up. Check:

1. **Is the server running?** `curl http://127.0.0.1:49374/mcp` should
   return a JSON-RPC error (not a connection refused). If refused,
   start ai-memory: `docker start ai-memory` or
   `ai-memory serve --transport http`.
2. **Did the client reload the config?** Cursor, Claude Desktop, and
   OMP need a restart. Gemini CLI and OpenClaw usually pick it up on
   next session-start.
3. **Are you on the right port?** ai-memory's default is **49374**
   (`0xC0DE` in hex). If you remapped, update the URL in every
   client's config.

If the model sees the tools but they all error, the server is
probably running in a different data dir than expected. Check
`docker logs ai-memory` or `ai-memory status --json` for the data
dir on disk.

---

## When does the auto-handoff actually work?

The cross-agent handoff feature (the "headline" pitch in the README)
requires both sides - the agent that *ends* a session, and the agent
that *starts* the next one - to play nicely with ai-memory:

| Side | What's needed | Covered by |
|---|---|---|
| **Ending side** | The agent must create a handoff, either through a true session-end hook or by calling `memory_handoff_begin`. | Built-in for Claude Code, Codex, Cursor, Gemini CLI, and OMP. OpenCode has no true session-end event, so ask it to call `memory_handoff_begin` before quitting when you need a handoff. |
| **Starting side** | Either (a) the session-start/plugin path injects the handoff via `/handoff`, OR (b) the model proactively calls `memory_handoff_accept` on first turn. | (a) is built-in for Claude Code / Codex / Cursor / Gemini CLI / OpenCode / OMP. (b) works for any MCP-capable client if you nudge the model - see [the CLAUDE.md snippet](../README.md#nudging-the-agent-to-use-memory-proactively). |

So a typical mixed workflow looks like:

- **Claude Code → Cursor.** Claude Code's `SessionEnd` creates the
  handoff automatically. Cursor doesn't have a SessionStart hook, so
  in Cursor's first chat you nudge the model: "Call
  memory_handoff_accept to resume prior work." The model fetches the
  handoff via MCP and continues.
- **Claude Desktop → Claude Code.** Claude Desktop doesn't write a
  handoff (no hooks). To resume in Claude Code, you'd have had to
  call `memory_handoff_begin` manually in Claude Desktop before
  quitting. ai-memory's wiki content via `memory_query` is still
  available either way.

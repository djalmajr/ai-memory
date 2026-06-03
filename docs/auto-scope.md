# `[auto_scope]` isolation modes

`ai-memory serve` publishes a process-shared "currently active project"
pointer that MCP read tools consult when the caller omits `workspace` /
`project`. The pointer is fed by the lifecycle hooks: every `/hook`
event that resolves a `cwd` to a real project updates the pointer so
read tools answer for the project the agent is actually in, not the
server's static `--project` default.

By default that pointer is a single process-wide slot — right for one
operator running one project at a time, but it collapses parallel
sessions on shared installs: a hook firing from `~/repo-A` overwrites
the slot that a concurrent `memory_query` (with no explicit project)
in `~/repo-B` was about to read.

The `[auto_scope]` config block selects opt-in isolation modes that
key the pointer by request identity so concurrent callers stay
separated.

## Modes

| `mode`        | Key                    | When to use                                                                                              |
|---------------|------------------------|----------------------------------------------------------------------------------------------------------|
| `single`      | (none — global slot)   | **Default.** Single operator, one project at a time. Backward-compatible with every existing install.    |
| `per_session` | `session_id`           | Single operator running concurrent agent runs in different repos (e.g. several Claude Code / Codex windows open). |
| `per_actor`   | `(user, session_id)`   | Shared engine fielding multiple authenticated users (multi-user mode, rung 2). Isolates across operators too. |

Both opt-in modes still publish to the single slot in parallel, so a
caller with no actor identity (anonymous probe, legacy code path) sees
the most recent project rather than an empty pointer — graceful
degradation, never a silent error.

## Configuration

```toml
[auto_scope]
mode = "single"           # "single" (default) | "per_session" | "per_actor"
session_ttl_secs = 3600   # TTL for per-key entries (default 1 h)
max_entries = 4096        # hard cap; oldest insertions evicted first
```

Environment-variable overrides follow the standard
`AI_MEMORY_<SECTION>__<KEY>` shape:

```bash
AI_MEMORY_AUTO_SCOPE__MODE=per_actor
AI_MEMORY_AUTO_SCOPE__SESSION_TTL_SECS=7200
AI_MEMORY_AUTO_SCOPE__MAX_ENTRIES=8192
```

## Where the actor identity comes from

| Source                                             | Populates                  |
|----------------------------------------------------|----------------------------|
| Hook payload (`/hook?event=…&agent=…`)             | `session_id`, `agent`      |
| Auth middleware (rung 1 root with `root_username`) | `user` ← root_username     |
| Auth middleware (rung 2 DB user)                   | `user` ← `users.username`  |
| Anonymous / no token                               | empty actor → single slot  |

`per_session` reads from `session_id`; `per_actor` reads from both
`user` and `session_id`. When either is missing, the lookup falls back
through the chain ending at the single slot.

## Pairing with multi-user mode

`per_actor` is most useful when the engine is in multi-user mode (see
[`docs/users.md`](users.md)) — each authenticated user has their own
`users.token_hash` row, so the auth middleware tags every request with
the right `user`. With `[auto_scope] mode = "per_actor"`, two
authenticated users running concurrent agent sessions through the same
engine no longer overwrite each other's "current project" pointer.

Single-user installs can use `per_session` alone (no `token_pepper`,
no `users` row) when one operator wants to run multiple agent windows
in parallel.

## Memory footprint

Per-key entries are tiny: two `Uuid`-sized ids + an `Instant`. With
the default `max_entries = 4096`, the map worst-cases at ~tens of KB
even on a corporate engine fielding hundreds of concurrent sessions.
The TTL ensures stale entries (closed Claude Code windows, dropped
hook clients) age out within an hour; the cap drops the oldest
insertions first if the TTL window is somehow exceeded.

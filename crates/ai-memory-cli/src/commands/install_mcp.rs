//! `ai-memory install-mcp` — print the MCP server registration
//! snippet for any supported client.
//!
//! The snippet format and the config-file location differ across
//! clients. We render the *content* the user needs to paste; we
//! deliberately do not auto-edit their config (formats are evolving
//! upstream and a bad merge is very user-visible).
//!
//! For clients that don't support remote MCP servers in their JSON
//! config (Claude Desktop today), the rendered snippet uses the
//! community-standard `npx mcp-remote` stdio shim so the same HTTP
//! endpoint still works.
//!
//! For clients that don't support MCP at all (`pi` per the upstream
//! author's stated position), we print an explanation + pointers
//! instead of fabricating a config.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde_json::json;

use crate::cli::{InstallMcpArgs, McpClient};
use crate::commands::apply_shared::{ApplyOutcome, apply_atomic, mutate_json, mutate_toml};
use crate::commands::render_shared::bearer_header_value;
use crate::config::Config;

/// Run the `install-mcp` subcommand.
///
/// # Errors
/// Returns an error if JSON serialisation fails (should never happen
/// for our handcrafted values).
pub fn run(config: &Config, args: InstallMcpArgs) -> Result<()> {
    let args = InstallMcpArgs {
        auth_token: args.auth_token.or_else(|| config.auth.bearer_token.clone()),
        ..args
    };
    if args.apply {
        if matches!(args.client, McpClient::Pi) {
            bail!(
                "--apply is not supported for `pi` — pi has no MCP config to mutate. \
                 Run without --apply to print the non-support explanation."
            );
        }
        return apply_to_config_file(&args);
    }
    let snippet = match args.client {
        McpClient::ClaudeCode => render_claude_code(&args)?,
        McpClient::Codex => render_codex(&args),
        McpClient::OpenCode => render_opencode(&args)?,
        McpClient::Cursor => render_cursor(&args)?,
        McpClient::ClaudeDesktop => render_claude_desktop(&args)?,
        McpClient::GeminiCli => render_gemini_cli(&args)?,
        McpClient::Openclaw => render_openclaw(&args)?,
        McpClient::Pi => render_pi_explanation(&args),
    };
    println!("{snippet}");
    Ok(())
}

/// Default MCP config-file path for a client (ignores any
/// `--config-file` override). Shared by install and uninstall.
///
/// # Errors
/// Returns an error for `Pi` (no MCP config), for Claude Desktop on
/// unsupported OSes, or when `$HOME` can't be resolved.
pub(crate) fn mcp_config_path(client: crate::cli::McpClient) -> Result<PathBuf> {
    use crate::cli::McpClient;
    let home = dirs::home_dir().context("could not locate $HOME for config-file auto-detect")?;
    Ok(match client {
        // Claude Code reads MCP-server registrations from `~/.claude.json`
        // (the same file `claude mcp add`/`claude mcp list` operate on).
        // `~/.claude/settings.json` is a separate file for hooks /
        // permissions / etc. — putting `mcpServers` there does NOT make
        // Claude Code load the server. (Confirmed against CC 1.x by
        // observing that `mcpServers` in settings.json is silently
        // ignored while the same entry under `~/.claude.json` shows up
        // in `claude mcp list`.)
        McpClient::ClaudeCode => home.join(".claude.json"),
        McpClient::Codex => home.join(".codex").join("config.toml"),
        McpClient::OpenCode => home.join(".config").join("opencode").join("opencode.json"),
        McpClient::Cursor => home.join(".cursor").join("mcp.json"),
        McpClient::ClaudeDesktop => {
            #[cfg(target_os = "macos")]
            {
                home.join("Library")
                    .join("Application Support")
                    .join("Claude")
                    .join("claude_desktop_config.json")
            }
            #[cfg(target_os = "windows")]
            {
                // %APPDATA% is roughly ~/AppData/Roaming.
                home.join("AppData")
                    .join("Roaming")
                    .join("Claude")
                    .join("claude_desktop_config.json")
            }
            #[cfg(not(any(target_os = "macos", target_os = "windows")))]
            {
                bail!(
                    "Claude Desktop is not officially distributed for this OS. \
                     Pass --config-file explicitly if you know where it lives."
                );
            }
        }
        McpClient::GeminiCli => home.join(".gemini").join("settings.json"),
        McpClient::Openclaw => home.join(".openclaw").join("config.json"),
        McpClient::Pi => bail!("pi has no MCP config file (MCP not supported)"),
    })
}

/// Resolve the user-config file for this client. Honours
/// `--config-file` when provided, else uses the canonical default
/// per client.
fn resolve_config_file(args: &InstallMcpArgs) -> Result<PathBuf> {
    if let Some(p) = &args.config_file {
        return Ok(p.clone());
    }
    mcp_config_path(args.client)
}

/// Mutate the resolved client config file in place. Idempotent —
/// re-runs that produce the same content are reported as no-op.
fn apply_to_config_file(args: &InstallMcpArgs) -> Result<()> {
    let path = resolve_config_file(args)?;
    let outcome = match args.client {
        McpClient::ClaudeCode
        | McpClient::ClaudeDesktop
        | McpClient::Cursor
        | McpClient::GeminiCli => apply_atomic(&path, |existing| {
            mutate_json(existing, |root| {
                let entry = build_mcp_entry(args)?;
                let servers = root
                    .entry("mcpServers")
                    .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()))
                    .as_object_mut()
                    .context("`mcpServers` is present but not an object")?;
                servers.insert(args.name.clone(), entry);
                Ok(())
            })
        })?,
        McpClient::OpenCode => apply_atomic(&path, |existing| {
            mutate_json(existing, |root| {
                let entry = build_mcp_entry_opencode(args)?;
                let mcp = root
                    .entry("mcp")
                    .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()))
                    .as_object_mut()
                    .context("`mcp` is present but not an object")?;
                mcp.insert(args.name.clone(), entry);
                Ok(())
            })
        })?,
        McpClient::Openclaw => apply_atomic(&path, |existing| {
            mutate_json(existing, |root| {
                let entry = build_mcp_entry_openclaw(args)?;
                let mcp = root
                    .entry("mcp")
                    .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()))
                    .as_object_mut()
                    .context("`mcp` is present but not an object")?;
                let servers = mcp
                    .entry("servers")
                    .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()))
                    .as_object_mut()
                    .context("`mcp.servers` is present but not an object")?;
                servers.insert(args.name.clone(), entry);
                Ok(())
            })
        })?,
        McpClient::Codex => apply_atomic(&path, |existing| {
            mutate_toml(existing, |doc| codex_upsert_mcp_server(doc, args))
        })?,
        McpClient::Pi => unreachable!("pi guarded above"),
    };
    println!(
        "✓ {} {} ({})",
        outcome.verb(),
        path.display(),
        match outcome {
            ApplyOutcome::Created => "new file",
            ApplyOutcome::Updated => "backup written next to it",
            ApplyOutcome::NoOp => "already up to date",
        }
    );
    Ok(())
}

/// JSON entry shape used by Claude Code, Claude Desktop, Cursor, and
/// Gemini CLI — they all accept `mcpServers.<name>` with `url` or
/// `httpUrl` plus optional `headers`. Returns the per-client variant.
fn build_mcp_entry(args: &InstallMcpArgs) -> Result<serde_json::Value> {
    let bearer = bearer_header_value(args.auth_token.as_deref());
    let mut entry = serde_json::Map::new();
    match args.client {
        McpClient::ClaudeCode => {
            entry.insert("type".into(), json!("http"));
            entry.insert("url".into(), json!(args.server_url));
            if let Some(b) = &bearer {
                entry.insert("headers".into(), json!({"Authorization": b}));
            }
        }
        McpClient::ClaudeDesktop => {
            // Stdio shim via mcp-remote — Claude Desktop's JSON
            // doesn't accept HTTP transport directly.
            let mut cmd_args = vec![json!("-y"), json!("mcp-remote"), json!(args.server_url)];
            if let Some(b) = &bearer {
                cmd_args.push(json!("--header"));
                cmd_args.push(json!(format!("Authorization: {b}")));
            }
            entry.insert("command".into(), json!("npx"));
            entry.insert("args".into(), serde_json::Value::Array(cmd_args));
        }
        McpClient::Cursor => {
            entry.insert("url".into(), json!(args.server_url));
            if let Some(b) = &bearer {
                entry.insert("headers".into(), json!({"Authorization": b}));
            }
        }
        McpClient::GeminiCli => {
            entry.insert("httpUrl".into(), json!(args.server_url));
            entry.insert("timeout".into(), json!(5000));
            if let Some(b) = &bearer {
                entry.insert("headers".into(), json!({"Authorization": b}));
            }
        }
        _ => bail!("internal: build_mcp_entry called for unsupported client"),
    }
    Ok(serde_json::Value::Object(entry))
}

fn build_mcp_entry_opencode(args: &InstallMcpArgs) -> Result<serde_json::Value> {
    let bearer = bearer_header_value(args.auth_token.as_deref());
    let mut entry = serde_json::Map::new();
    entry.insert("type".into(), json!("remote"));
    entry.insert("url".into(), json!(args.server_url));
    entry.insert("enabled".into(), json!(true));
    if let Some(b) = bearer {
        entry.insert("headers".into(), json!({"Authorization": b}));
    }
    Ok(serde_json::Value::Object(entry))
}

fn build_mcp_entry_openclaw(args: &InstallMcpArgs) -> Result<serde_json::Value> {
    let bearer = bearer_header_value(args.auth_token.as_deref());
    let mut entry = serde_json::Map::new();
    entry.insert("url".into(), json!(args.server_url));
    entry.insert("transport".into(), json!("streamable-http"));
    if let Some(b) = bearer {
        entry.insert("headers".into(), json!({"Authorization": b}));
    }
    Ok(serde_json::Value::Object(entry))
}

/// Insert / replace `[mcp_servers.<name>]` in a Codex `config.toml`.
///
/// Codex parses both forms (block-style `[mcp_servers.foo]` and the
/// dotted-inline `mcp_servers = { foo = { ... } }`), but its docs show
/// the block form and that's the only one humans want to read. This
/// helper canonicalises to the block form even when the file currently
/// stores `mcp_servers` as an inline table — siblings are preserved.
fn codex_upsert_mcp_server(
    doc: &mut toml_edit::DocumentMut,
    args: &InstallMcpArgs,
) -> anyhow::Result<()> {
    use toml_edit::{Item, Table, Value, value};

    // Capture sibling entries from either inline-table or block-table
    // storage so we can rebuild in block form without dropping them.
    let preserved: Vec<(String, Item)> = match doc.get("mcp_servers") {
        Some(Item::Table(t)) => t
            .iter()
            .filter(|(k, _)| *k != args.name.as_str())
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
        Some(Item::Value(Value::InlineTable(it))) => it
            .iter()
            .filter(|(k, _)| *k != args.name.as_str())
            .map(|(k, v)| (k.to_string(), Item::Value(v.clone())))
            .collect(),
        _ => Vec::new(),
    };

    // Build our `[mcp_servers.<name>]` as a block-style table.
    //
    // IMPORTANT: Codex's MCP schema (verified against
    // `openai/codex/codex-rs/config/src/mcp_types.rs`) draws a hard
    // line between transports. For STREAMABLE_HTTP (which ai-memory
    // uses — `url = "...mcp"` triggers this transport), the
    // allowed auth-related keys are:
    //
    //   bearer_token_env_var  string  env-var NAME holding the token
    //   http_headers          table   static headers map
    //   env_http_headers      table   header_name → env_var_name
    //
    // `bearer_token` (literal) is rejected with
    //   "bearer_token is not supported for streamable_http"
    // — it's a stdio-transport-only key. Confusingly the field
    // sits in the same struct, but throw_if_set guards it for
    // streamable_http.
    //
    // We use [mcp_servers.<name>.http_headers] with a literal
    // Authorization header. Static, no env-var dance required.
    //
    // History note (so the next maintainer doesn't repeat this):
    //   - v1: emitted `[mcp_servers.X.headers]` — wrong key name
    //     entirely, Codex silently ignored it and fell back to
    //     OAuth ("Run `codex mcp login <name>`").
    //   - v2: switched to top-level `bearer_token = "..."` — also
    //     wrong; Codex rejects this for streamable_http with the
    //     "bearer_token is not supported" error.
    //   - v3 (this): `[mcp_servers.X.http_headers]` with
    //     `Authorization = "Bearer ..."`. Codex schema-validates
    //     and uses it as a static auth header.
    let mut server = Table::new();
    server["url"] = value(args.server_url.clone());
    // Auto-approve ai-memory's tool calls. Without this, Codex
    // prompts on EVERY tool invocation ("approve memory_query?"
    // "approve memory_briefing?" …) which makes the MCP unusable
    // for an auto-capture workflow. The valid TOML values per
    // Codex's `AppToolApproval` enum are "auto" / "prompt" /
    // "approve" — `approve` means "no prompt, just run it". ai-
    // memory's surface is dominantly read-only (query, recent,
    // status, briefing, explore); the few writes (consolidate,
    // forget_sweep) are tagged `destructiveHint: true` upstream
    // so any agent that wants to gate THOSE specifically can
    // override per-tool — see Codex's `[mcp_servers.X.tools]`
    // map.
    server["default_tools_approval_mode"] = value("approve");
    if let Some(b) = bearer_header_value(args.auth_token.as_deref()) {
        let mut headers = Table::new();
        headers["Authorization"] = value(b);
        server["http_headers"] = Item::Table(headers);
    }

    // Replace `mcp_servers` wholesale with a fresh implicit parent
    // table. Implicit = render only the dotted `[mcp_servers.<name>]`
    // headers, never a bare `[mcp_servers]` header.
    let mut parent = Table::new();
    parent.set_implicit(true);
    for (k, v) in preserved {
        parent.insert(&k, v);
    }
    parent.insert(&args.name, Item::Table(server));

    doc.insert("mcp_servers", Item::Table(parent));
    Ok(())
}

fn render_claude_code(args: &InstallMcpArgs) -> Result<String> {
    let bearer = bearer_header_value(args.auth_token.as_deref());
    let cli_line = if let Some(b) = &bearer {
        format!(
            "claude mcp add --transport http {name} {url} \\\n    --header \"Authorization: {b}\"",
            name = args.name,
            url = args.server_url,
            b = b,
        )
    } else {
        format!(
            "claude mcp add --transport http {name} {url}",
            name = args.name,
            url = args.server_url,
        )
    };
    let mut entry = serde_json::Map::new();
    entry.insert("type".into(), json!("http"));
    entry.insert("url".into(), json!(args.server_url));
    if let Some(b) = &bearer {
        entry.insert("headers".into(), json!({"Authorization": b}));
    }
    let snippet = serde_json::to_string_pretty(&json!({
        "mcpServers": { args.name.as_str(): entry }
    }))?;
    Ok(format!(
        "# Claude Code — register the MCP server\n\
         #\n\
         # Recommended (one-shot CLI):\n\
         {cli_line}\n\
         #\n\
         # Equivalent JSON if you'd rather edit settings directly\n\
         # (~/.claude/settings.json):\n\
         {snippet}\n"
    ))
}

fn render_codex(args: &InstallMcpArgs) -> String {
    // Codex uses TOML, not JSON. Hand-render the snippet so the
    // table headers stay deterministic.
    //
    // Schema: Codex's MCP `streamable_http` transport accepts
    //   - `bearer_token_env_var = "NAME"` (env-var indirection)
    //   - `[mcp_servers.<name>.http_headers]` (static headers)
    //   - `[mcp_servers.<name>.env_http_headers]` (env-var-sourced headers)
    // — NOT a literal `bearer_token = "..."` (that's stdio-only)
    // and NOT a `[mcp_servers.<name>.headers]` sub-table (the key
    // is `http_headers`, with the `http_` prefix).
    let mut out = format!(
        "# Codex CLI — append to ~/.codex/config.toml\n\
         #\n\
         [mcp_servers.{name}]\n\
         url = \"{url}\"\n\
         # Skip per-call approval prompts on ai-memory's tools.\n\
         # ai-memory is read-mostly + writes are auto-capture; the\n\
         # approval friction makes it unusable otherwise.\n\
         default_tools_approval_mode = \"approve\"\n",
        name = args.name,
        url = args.server_url,
    );
    if let Some(b) = bearer_header_value(args.auth_token.as_deref()) {
        out.push_str(&format!(
            "\n[mcp_servers.{name}.http_headers]\n\
             Authorization = \"{b}\"\n\
             # Alternative (avoids embedding the literal token):\n\
             # bearer_token_env_var = \"AI_MEMORY_AUTH_TOKEN\"\n\
             # — and export AI_MEMORY_AUTH_TOKEN in your shell init.\n",
            name = args.name,
            b = b,
        ));
    }
    out
}

fn render_opencode(args: &InstallMcpArgs) -> Result<String> {
    let mut entry = serde_json::Map::new();
    entry.insert("type".into(), json!("remote"));
    entry.insert("url".into(), json!(args.server_url));
    entry.insert("enabled".into(), json!(true));
    if let Some(b) = bearer_header_value(args.auth_token.as_deref()) {
        entry.insert("headers".into(), json!({"Authorization": b}));
    }
    Ok(format!(
        "# OpenCode — add to ~/.config/opencode/opencode.json under \"mcp\":\n\
         {snippet}\n",
        snippet = serde_json::to_string_pretty(&json!({
            "mcp": { args.name.as_str(): entry }
        }))?,
    ))
}

fn render_cursor(args: &InstallMcpArgs) -> Result<String> {
    let mut entry = serde_json::Map::new();
    entry.insert("url".into(), json!(args.server_url));
    if let Some(b) = bearer_header_value(args.auth_token.as_deref()) {
        entry.insert("headers".into(), json!({"Authorization": b}));
    }
    Ok(format!(
        "# Cursor — write to one of:\n\
         #   - ~/.cursor/mcp.json   (global, all projects)\n\
         #   - .cursor/mcp.json     (per-project, in the workspace root)\n\
         #\n\
         # Cursor supports HTTP MCP servers via the `url` field. Restart\n\
         # Cursor (or toggle the server off+on in Settings → MCP) after\n\
         # adding a new entry; live reload landed in recent builds but\n\
         # is still flaky.\n\
         {snippet}\n",
        snippet = serde_json::to_string_pretty(&json!({
            "mcpServers": { args.name.as_str(): entry }
        }))?,
    ))
}

fn render_claude_desktop(args: &InstallMcpArgs) -> Result<String> {
    // mcp-remote's --header flag is how we plumb the Authorization
    // through Claude Desktop's stdio-only config.
    let mut cmd_args = vec![json!("-y"), json!("mcp-remote"), json!(args.server_url)];
    if let Some(b) = bearer_header_value(args.auth_token.as_deref()) {
        cmd_args.push(json!("--header"));
        cmd_args.push(json!(format!("Authorization: {b}")));
    }
    Ok(format!(
        "# Claude Desktop — write to claude_desktop_config.json:\n\
         #   - macOS:    ~/Library/Application Support/Claude/claude_desktop_config.json\n\
         #   - Windows:  %APPDATA%\\Claude\\claude_desktop_config.json\n\
         #   - Linux:    Claude Desktop is not officially distributed for Linux;\n\
         #               use Claude Code or another HTTP client instead.\n\
         #\n\
         # Claude Desktop's JSON config does not support HTTP MCP servers\n\
         # directly. We bridge through the community `mcp-remote` stdio shim\n\
         # (https://www.npmjs.com/package/mcp-remote). Requires Node.js.\n\
         # After editing, fully quit + relaunch Claude Desktop; \"Check for\n\
         # Updates\" is not enough.\n\
         {snippet}\n",
        snippet = serde_json::to_string_pretty(&json!({
            "mcpServers": {
                args.name.as_str(): {
                    "command": "npx",
                    "args": cmd_args,
                }
            }
        }))?,
    ))
}

fn render_gemini_cli(args: &InstallMcpArgs) -> Result<String> {
    let mut entry = serde_json::Map::new();
    entry.insert("httpUrl".into(), json!(args.server_url));
    entry.insert("timeout".into(), json!(5000));
    if let Some(b) = bearer_header_value(args.auth_token.as_deref()) {
        entry.insert("headers".into(), json!({"Authorization": b}));
    }
    Ok(format!(
        "# Gemini CLI — merge into ~/.gemini/settings.json:\n\
         #\n\
         # Gemini CLI uses `httpUrl` (not `url`) for streamable-HTTP\n\
         # endpoints. The `timeout` is in milliseconds.\n\
         {snippet}\n",
        snippet = serde_json::to_string_pretty(&json!({
            "mcpServers": { args.name.as_str(): entry }
        }))?,
    ))
}

fn render_openclaw(args: &InstallMcpArgs) -> Result<String> {
    let mut entry = serde_json::Map::new();
    entry.insert("url".into(), json!(args.server_url));
    entry.insert("transport".into(), json!("streamable-http"));
    if let Some(b) = bearer_header_value(args.auth_token.as_deref()) {
        entry.insert("headers".into(), json!({"Authorization": b}));
    }
    Ok(format!(
        "# OpenClaw — merge into ~/.openclaw/config.json:\n\
         #\n\
         # OpenClaw distinguishes transports explicitly. Use\n\
         # \"transport\": \"streamable-http\" for ai-memory's HTTP endpoint.\n\
         {snippet}\n",
        snippet = serde_json::to_string_pretty(&json!({
            "mcp": { "servers": { args.name.as_str(): entry } }
        }))?,
    ))
}

fn render_pi_explanation(_args: &InstallMcpArgs) -> String {
    // No JSON to print. Plain-text explanation + pointers; mirrors
    // the docs/mcp-install.md#pi section so the help is self-contained.
    "# pi (@mariozechner/pi-coding-agent) — NOT supported via MCP\n\
     #\n\
     # pi's author has stated MCP support is not on the roadmap; see\n\
     # https://mariozechner.at/posts/2025-11-30-pi-coding-agent/ for the\n\
     # design rationale (token-budget concerns). agentmemory itself uses\n\
     # pi's native extension surface instead of MCP.\n\
     #\n\
     # Options if you want ai-memory + pi:\n\
     #   1. Run ai-memory and use it from another client (Claude Code,\n\
     #      Codex, Cursor, Gemini CLI) for the cross-agent handoff;\n\
     #      keep pi for code-edit sessions you don't need to persist.\n\
     #   2. Use the third-party `pi-mcp-adapter` shim\n\
     #      (https://github.com/nicobailon/pi-mcp-adapter) — unofficial,\n\
     #      community-maintained, may break on pi updates.\n\
     #\n\
     # See docs/mcp-install.md#pi-not-supported for the full discussion.\n"
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_for(client: McpClient) -> InstallMcpArgs {
        InstallMcpArgs {
            client,
            server_url: "http://127.0.0.1:49374/mcp".into(),
            name: "ai-memory".into(),
            auth_token: None,
            apply: false,
            config_file: None,
        }
    }

    fn args_with_token(client: McpClient) -> InstallMcpArgs {
        InstallMcpArgs {
            client,
            server_url: "http://127.0.0.1:49374/mcp".into(),
            name: "ai-memory".into(),
            auth_token: Some("test-token-deadbeef".into()),
            apply: false,
            config_file: None,
        }
    }

    fn render_with_token(client: McpClient) -> String {
        let args = args_with_token(client);
        match args.client {
            McpClient::ClaudeCode => render_claude_code(&args).unwrap(),
            McpClient::Codex => render_codex(&args),
            McpClient::OpenCode => render_opencode(&args).unwrap(),
            McpClient::Cursor => render_cursor(&args).unwrap(),
            McpClient::ClaudeDesktop => render_claude_desktop(&args).unwrap(),
            McpClient::GeminiCli => render_gemini_cli(&args).unwrap(),
            McpClient::Openclaw => render_openclaw(&args).unwrap(),
            McpClient::Pi => render_pi_explanation(&args),
        }
    }

    /// With `--auth-token` set, every non-pi renderer must embed the
    /// Bearer header in its output. pi prints the same non-support
    /// message regardless.
    #[test]
    fn auth_token_threaded_into_every_client() {
        for client in [
            McpClient::ClaudeCode,
            McpClient::Codex,
            McpClient::OpenCode,
            McpClient::Cursor,
            McpClient::ClaudeDesktop,
            McpClient::GeminiCli,
            McpClient::Openclaw,
        ] {
            let out = render_with_token(client);
            // Every client embeds the token as `Authorization:
            // Bearer <token>` in some flavour of headers map — the
            // exact key path differs (Codex uses `http_headers`,
            // OpenCode uses `headers`, Cursor / Gemini / Claude
            // Desktop / Claude Code use `headers` inside their
            // server entry, etc.), but the literal `Bearer
            // <token>` substring shows up in all of them. Keep
            // the assertion uniform.
            assert!(
                out.contains("Bearer test-token-deadbeef"),
                "client {client:?} did not embed the bearer token:\n{out}"
            );
        }
    }

    /// Sanity: every supported client renders without error and the
    /// output mentions the configured server URL (or, for pi, the
    /// "not supported" verbiage).
    #[test]
    fn every_client_renders() {
        for client in [
            McpClient::ClaudeCode,
            McpClient::Codex,
            McpClient::OpenCode,
            McpClient::Cursor,
            McpClient::ClaudeDesktop,
            McpClient::GeminiCli,
            McpClient::Openclaw,
            McpClient::Pi,
        ] {
            let out = render_for_test(client);
            if matches!(client, McpClient::Pi) {
                assert!(
                    out.contains("NOT supported"),
                    "pi must explain non-support: {out}"
                );
            } else {
                assert!(
                    out.contains("http://127.0.0.1:49374/mcp"),
                    "client {client:?} did not include the server URL in output:\n{out}"
                );
            }
        }
    }

    fn render_for_test(client: McpClient) -> String {
        let args = args_for(client);
        match args.client {
            McpClient::ClaudeCode => render_claude_code(&args).unwrap(),
            McpClient::Codex => render_codex(&args),
            McpClient::OpenCode => render_opencode(&args).unwrap(),
            McpClient::Cursor => render_cursor(&args).unwrap(),
            McpClient::ClaudeDesktop => render_claude_desktop(&args).unwrap(),
            McpClient::GeminiCli => render_gemini_cli(&args).unwrap(),
            McpClient::Openclaw => render_openclaw(&args).unwrap(),
            McpClient::Pi => render_pi_explanation(&args),
        }
    }

    /// Specific shape checks — each client has a distinguishing key
    /// in its JSON snippet. This catches accidental cross-pollination
    /// between renderers (e.g. Gemini's `httpUrl` showing up under
    /// Cursor's `mcpServers`).
    #[test]
    fn client_specific_shape_keys() {
        assert!(render_for_test(McpClient::Cursor).contains("\"url\""));
        assert!(render_for_test(McpClient::GeminiCli).contains("\"httpUrl\""));
        assert!(render_for_test(McpClient::ClaudeDesktop).contains("mcp-remote"));
        assert!(render_for_test(McpClient::Openclaw).contains("\"streamable-http\""));
        assert!(render_for_test(McpClient::Codex).contains("[mcp_servers.ai-memory]"));
    }

    /// The Codex apply path must emit block-form `[mcp_servers.<name>]`
    /// headers, NOT a dotted inline-table on one line. Regression
    /// guard: M22 originally created `mcp_servers = { ai-memory = {...} }`
    /// because toml_edit auto-vivifies inline tables when you assign
    /// through `doc["foo"]["bar"]`.
    #[test]
    fn codex_apply_writes_block_form_tables() {
        let args = args_with_token(McpClient::Codex);
        let mut doc: toml_edit::DocumentMut = "".parse().unwrap();
        codex_upsert_mcp_server(&mut doc, &args).unwrap();
        let out = doc.to_string();
        assert!(
            out.contains("[mcp_servers.ai-memory]"),
            "expected block-form table header, got:\n{out}"
        );
        // Auth lives on the [mcp_servers.X.http_headers] sub-table
        // with an Authorization: Bearer <token> value. The key is
        // `http_headers` (with the `http_` prefix) per Codex's
        // streamable_http schema. Two related regressions guarded
        // here:
        //   - the legacy `headers` key (no `http_` prefix) made
        //     Codex silently fall back to OAuth login;
        //   - a top-level `bearer_token = "..."` was rejected with
        //     "bearer_token is not supported for streamable_http"
        //     (that key is stdio-transport-only).
        assert!(
            out.contains("[mcp_servers.ai-memory.http_headers]"),
            "expected `[mcp_servers.X.http_headers]` sub-table, got:\n{out}"
        );
        assert!(
            out.contains("Authorization = \"Bearer test-token-deadbeef\""),
            "expected the Authorization header in the http_headers sub-table, got:\n{out}"
        );
        assert!(
            !out.contains("[mcp_servers.ai-memory.headers]"),
            "legacy `headers` key (no `http_` prefix) must not be emitted; got:\n{out}"
        );
        assert!(
            !out.contains("\nbearer_token ="),
            "top-level `bearer_token` is rejected for streamable_http; must not be emitted; got:\n{out}"
        );
        assert!(
            !out.contains("mcp_servers = {"),
            "found inline-table form (regression):\n{out}"
        );
    }

    /// Migrating from the old M22 inline-table form to block form must
    /// be idempotent — the second apply produces identical output.
    #[test]
    fn codex_apply_migrates_inline_form_and_is_idempotent() {
        let args = args_with_token(McpClient::Codex);

        // Simulate a config.toml in the *old* inline form.
        let original = "approval_policy = \"on-request\"\n\
                        mcp_servers = { ai-memory = { url = \"http://old\", \
                        headers = { Authorization = \"Bearer old\" } } }\n\
                        \n\
                        [other]\n\
                        keep = \"this\"\n";
        let mut doc: toml_edit::DocumentMut = original.parse().unwrap();
        codex_upsert_mcp_server(&mut doc, &args).unwrap();
        let first = doc.to_string();

        // After migration the inline-table form is gone.
        assert!(!first.contains("mcp_servers = {"));
        assert!(first.contains("[mcp_servers.ai-memory]"));
        // Unrelated content survives.
        assert!(first.contains("approval_policy"));
        assert!(first.contains("[other]"));
        assert!(first.contains("keep = \"this\""));

        // Re-applying produces the same bytes (idempotency contract).
        let mut doc2: toml_edit::DocumentMut = first.parse().unwrap();
        codex_upsert_mcp_server(&mut doc2, &args).unwrap();
        let second = doc2.to_string();
        assert_eq!(
            first, second,
            "second apply must produce identical bytes; diff:\n--- first\n{first}\n--- second\n{second}"
        );
    }

    /// Sibling `[mcp_servers.<other>]` entries the user has configured
    /// (e.g. a different MCP server) must survive an --apply.
    #[test]
    fn codex_apply_preserves_sibling_mcp_servers() {
        let args = args_for(McpClient::Codex);
        let original = "[mcp_servers.other-server]\n\
                        url = \"http://other\"\n";
        let mut doc: toml_edit::DocumentMut = original.parse().unwrap();
        codex_upsert_mcp_server(&mut doc, &args).unwrap();
        let out = doc.to_string();
        assert!(out.contains("[mcp_servers.other-server]"));
        assert!(out.contains("http://other"));
        assert!(out.contains("[mcp_servers.ai-memory]"));
    }
}

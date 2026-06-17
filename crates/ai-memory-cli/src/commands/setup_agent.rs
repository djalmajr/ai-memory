//! `ai-memory setup-agent` — one-shot agent integration for the
//! docker-primary workflow.
//!
//! Solves the problem that `install-hooks` alone can't handle in a
//! docker-only deploy: the JSON snippet `install-hooks` emits
//! references absolute paths to hook scripts, and those paths must
//! exist on the host machine that runs the agent CLI (Claude Code et
//! al. shell out from the host, not inside the container).
//!
//! `setup-agent` bundles the extract + render into one command:
//!
//!     docker run --rm \
//!       -v "$HOME/.ai-memory:/host" \
//!       akitaonrails/ai-memory:latest \
//!       setup-agent \
//!         --agent claude-code \
//!         --to /host/hooks \
//!         --host-prefix "$HOME/.ai-memory/hooks" \
//!         --auth-token "$TOKEN"
//!
//! 1. Copies `/usr/local/share/ai-memory/hooks/claude-code/*.{sh,ps1}` into
//!    `/host/hooks/claude-code/` (which on the host is
//!    `$HOME/.ai-memory/hooks/claude-code/`).
//! 2. Prints the JSON config snippet whose `command` fields point at
//!    `$HOME/.ai-memory/hooks/claude-code/*.{sh,ps1}` (via `--host-prefix`)
//!    so Claude Code on the host can exec them.
//!
//! When `--host-prefix` is omitted it defaults to `--to`, which is
//! the right behaviour for a non-docker (`cargo run`) invocation
//! where the in-container path and the host path are the same thing.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::cli::{AgentChoice, SetupAgentArgs};
use crate::commands::render_shared::{
    CLAUDE_CODE_EVENTS, build_claude_code_payload, build_grok_payload,
    hook_script_for_current_platform,
};
use crate::config::{Config, DEFAULT_SERVER_URL};

/// Run the `setup-agent` subcommand.
///
/// # Errors
/// Returns an error if the source bundle can't be located, the
/// destination directory can't be created, any script copy fails,
/// or the JSON config can't be serialised.
pub fn run(config: &Config, args: SetupAgentArgs) -> Result<()> {
    let server_url = if args.server_url == DEFAULT_SERVER_URL && config.server_url_configured() {
        normalise_hook_server_url(&config.server_url)
    } else {
        normalise_hook_server_url(&args.server_url)
    };
    let args = SetupAgentArgs {
        server_url,
        auth_token: args.auth_token.or_else(|| config.auth.bearer_token.clone()),
        ..args
    };
    if matches!(
        args.agent,
        AgentChoice::OpenCode | AgentChoice::Omp | AgentChoice::Openclaw
    ) {
        emit_extension_setup_hint(&args);
        return Ok(());
    }
    let agent_sub = match args.agent {
        AgentChoice::ClaudeCode => "claude-code",
        AgentChoice::Codex => "codex",
        AgentChoice::Cursor => "cursor",
        AgentChoice::GeminiCli => "gemini-cli",
        AgentChoice::AntigravityCli => "antigravity-cli",
        AgentChoice::Grok => "grok",
        AgentChoice::OpenCode => unreachable!("opencode handled above"),
        AgentChoice::Omp => unreachable!("omp handled above"),
        AgentChoice::Openclaw => unreachable!("openclaw handled above"),
    };

    let source = resolve_source(args.source.as_deref(), agent_sub)?;
    let dest_dir = args.to.join(agent_sub);

    fs::create_dir_all(&dest_dir)
        .with_context(|| format!("creating destination {}", dest_dir.display()))?;

    let mut copied = 0_usize;
    for entry in fs::read_dir(&source)
        .with_context(|| format!("reading source bundle {}", source.display()))?
    {
        let entry = entry?;
        let from = entry.path();
        if !from.is_file() || !is_hook_script_file(&from) {
            continue;
        }
        let file_name = from
            .file_name()
            .with_context(|| format!("invalid hook script path {}", from.display()))?;
        let to = dest_dir.join(file_name);
        fs::copy(&from, &to)
            .with_context(|| format!("copying {} → {}", from.display(), to.display()))?;
        // Preserve executable bit so the agent CLI can actually run
        // the scripts. On Windows this is a no-op.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&to)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&to, perms)?;
        }
        copied += 1;
    }

    copy_support_hook_scripts(&source, &dest_dir)?;

    eprintln!(
        "✓ Extracted {copied} hook script(s) from {} to {}",
        source.display(),
        dest_dir.display(),
    );

    // The path the rendered JSON should reference. Defaults to where
    // we just copied the scripts; override with --host-prefix when
    // running inside docker against a mounted volume.
    let emit_root = args
        .host_prefix
        .as_deref()
        .unwrap_or(&args.to)
        .join(agent_sub);

    match args.agent {
        AgentChoice::ClaudeCode => emit_claude_code(&emit_root, &args)?,
        AgentChoice::Grok => emit_grok(&emit_root, &args)?,
        AgentChoice::Codex
        | AgentChoice::Cursor
        | AgentChoice::GeminiCli
        | AgentChoice::AntigravityCli => {
            emit_other(&emit_root, agent_sub, &args);
        }
        AgentChoice::OpenCode => unreachable!("opencode handled above"),
        AgentChoice::Omp => unreachable!("omp handled above"),
        AgentChoice::Openclaw => {
            unreachable!("openclaw handled by the generated-integration hint above");
        }
    }
    Ok(())
}

fn normalise_hook_server_url(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

fn emit_extension_setup_hint(args: &SetupAgentArgs) {
    let (label, agent, restart_note, mcp_client) = match args.agent {
        AgentChoice::OpenCode => (
            "OpenCode",
            "opencode",
            "Then restart OpenCode so it loads ~/.config/opencode/plugins/ai-memory.ts.",
            "opencode",
        ),
        AgentChoice::Omp => (
            "OMP",
            "omp",
            "Then restart OMP so it loads ~/.omp/agent/extensions/ai-memory.ts.",
            "pi",
        ),
        AgentChoice::Openclaw => (
            "OpenClaw",
            "openclaw",
            "Then restart the OpenClaw gateway if it does not auto-restart after plugin install.",
            "openclaw",
        ),
        _ => unreachable!("only generated-integration agents reach this hint"),
    };
    println!("# {label} uses a TypeScript extension/plugin, not extracted shell scripts.");
    println!("# Install it directly instead:");
    println!("ai-memory install-hooks --agent {agent} --apply \\");
    if args.auth_token.is_some() {
        println!("  --server-url {} \\", args.server_url);
        println!("  --auth-token <token>");
    } else {
        println!("  --server-url {}", args.server_url);
        println!("  # add --auth-token <token> if the server requires bearer auth");
    }
    println!();
    println!("{restart_note}");
    println!("Also run `ai-memory install-mcp --client {mcp_client}` to wire MCP separately.");
}

fn emit_claude_code(emit_root: &Path, args: &SetupAgentArgs) -> Result<()> {
    let payload =
        build_claude_code_payload(emit_root, &args.server_url, args.auth_token.as_deref());
    let serialized =
        serde_json::to_string_pretty(&payload).context("serializing Claude Code hook config")?;
    println!("# Claude Code — merge into ~/.claude/settings.json");
    println!("# Hook scripts (must be reachable from the host that runs Claude Code):");
    println!("#   {}", emit_root.display());
    println!("# AI-memory server: {}", args.server_url);
    if args.auth_token.is_some() {
        println!("# Auth: AI_MEMORY_AUTH_TOKEN embedded in each hook's env block.");
        println!("#       Treat ~/.claude/settings.json as sensitive (chmod 600).");
    }
    println!("# Tip: also run `ai-memory install-mcp --client claude-code --auth-token <…>`");
    println!("#      to register the MCP endpoint (separate from hooks).");
    println!();
    println!("{serialized}");
    Ok(())
}

fn emit_grok(emit_root: &Path, args: &SetupAgentArgs) -> Result<()> {
    let payload = build_grok_payload(emit_root, &args.server_url, args.auth_token.as_deref());
    let serialized =
        serde_json::to_string_pretty(&payload).context("serializing Grok hook config")?;
    println!("# Grok Build CLI — write to ~/.grok/hooks/ai-memory.json");
    println!("# Hook scripts (must be reachable from the host that runs Grok):");
    println!("#   {}", emit_root.display());
    println!("# AI-memory server: {}", args.server_url);
    if args.auth_token.is_some() {
        println!("# Auth: AI_MEMORY_AUTH_TOKEN embedded in each hook command below.");
        println!("#       Treat ~/.grok/hooks/ai-memory.json as sensitive (chmod 600).");
    }
    println!("# NOTE: Grok ignores SessionStart stdout, so this config captures");
    println!("#       lifecycle events but does not inject handoffs automatically.");
    println!("#       Recover handoffs via the MCP memory_handoff_accept tool.");
    println!();
    println!("{serialized}");
    Ok(())
}

fn emit_other(emit_root: &Path, label: &str, args: &SetupAgentArgs) {
    // These clients have hook surfaces, but their print-mode config
    // snippets are intentionally conservative: apply-mode owns the
    // exact merge/plugin generation where ai-memory knows the format.
    println!("# {label} hook scripts (manual wire-up; use install-hooks --apply when available)");
    println!("# Scripts located at: {}", emit_root.display());
    println!("# Server URL:         {}", args.server_url);
    if args.auth_token.is_some() {
        println!("# Auth: set AI_MEMORY_AUTH_TOKEN in each hook's environment to the");
        println!("#       value you passed via --auth-token (not echoed).");
    }
    println!();
    for (_, script) in CLAUDE_CODE_EVENTS {
        let script = hook_script_for_current_platform(script);
        println!("- {}", emit_root.join(script.as_ref()).display());
    }
    println!();
    println!("Set AI_MEMORY_HOOK_URL in each hook's environment to override the default.");
    println!("Also run `ai-memory install-mcp --client {label}` to wire MCP separately.");
}

fn is_hook_script_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|s| s.to_str()),
        Some("sh" | "ps1")
    )
}

fn copy_support_hook_scripts(source_dir: &Path, dest_dir: &Path) -> Result<()> {
    let Some(source_hooks_root) = source_dir.parent() else {
        return Ok(());
    };
    let source_lib = source_hooks_root.join("lib");
    if !source_lib.is_dir() {
        return Ok(());
    }
    let Some(dest_hooks_root) = dest_dir.parent() else {
        return Ok(());
    };
    let dest_lib = dest_hooks_root.join("lib");
    fs::create_dir_all(&dest_lib)
        .with_context(|| format!("creating hook support dir {}", dest_lib.display()))?;
    for entry in fs::read_dir(&source_lib)
        .with_context(|| format!("reading hook support dir {}", source_lib.display()))?
    {
        let entry = entry?;
        let from = entry.path();
        if !from.is_file() || from.extension().and_then(|s| s.to_str()) != Some("ps1") {
            continue;
        }
        let to = dest_lib.join(
            from.file_name()
                .with_context(|| format!("invalid hook support path {}", from.display()))?,
        );
        fs::copy(&from, &to)
            .with_context(|| format!("copying {} → {}", from.display(), to.display()))?;
    }
    Ok(())
}

fn resolve_source(explicit: Option<&Path>, sub: &str) -> Result<PathBuf> {
    let candidates = source_candidates(explicit, sub, std::env::current_exe().ok());
    for path in &candidates {
        if path.is_dir() {
            return Ok(path.clone());
        }
    }
    bail!(
        "could not locate hook source bundle for {sub}. \
         Tried: {candidates:?}. Pass --source <dir> to override."
    );
}

/// Ordered directories to probe for the `<sub>` hook bundle.
///
/// `exe` is the running binary's path (`std::env::current_exe()`), threaded
/// in so the derivation is unit-testable. An `explicit` `--source` is trusted
/// verbatim; otherwise we try the packaged install locations plus two
/// binary-relative spots:
///   * `<exe_dir>/hooks/<sub>` — the **release tarball** ships `hooks/` right
///     beside the binary (macOS/Windows/Linux archives), and
///   * `<exe_dir>/../../hooks/<sub>` — `cargo run` from the repo, where the
///     binary lives under `target/<profile>/`.
///
/// Without the binary-sibling entry the flat tarball layout was unreachable:
/// from `/private/tmp/<dir>/ai-memory` the `parent×3` dev fallback derived a
/// bogus `/private/hooks/<sub>` and discovery failed (issue #107).
fn source_candidates(explicit: Option<&Path>, sub: &str, exe: Option<PathBuf>) -> Vec<PathBuf> {
    if let Some(p) = explicit {
        return vec![p.join(sub)];
    }
    let mut v = vec![
        // Docker image lays them out under /usr/local/share/.
        PathBuf::from(format!("/usr/local/share/ai-memory/hooks/{sub}")),
        // Native Linux packages install hook sources under /usr/share.
        PathBuf::from(format!("/usr/share/ai-memory/hooks/{sub}")),
    ];
    if let Some(exe) = exe {
        // Release tarball: `hooks/` sits in the same dir as the binary.
        if let Some(dir) = exe.parent() {
            v.push(dir.join("hooks").join(sub));
        }
        // Repo-local fallback for `cargo run setup-agent` during dev:
        // target/<profile>/<bin> → repo root.
        if let Some(root) = exe.parent().and_then(Path::parent).and_then(Path::parent) {
            v.push(root.join("hooks").join(sub));
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_candidates_include_binary_sibling_for_flat_tarball() {
        // Flat release tarball: the binary and its `hooks/` bundle are
        // extracted side by side. On macOS the path resolves under
        // /private/..., which the old `parent×3` dev fallback turned into a
        // bogus `/private/hooks/...` (issue #107).
        let exe = PathBuf::from("/private/tmp/ai-memory-macos-aarch64/ai-memory");
        let candidates = source_candidates(None, "claude-code", Some(exe));

        assert!(
            candidates.contains(&PathBuf::from(
                "/private/tmp/ai-memory-macos-aarch64/hooks/claude-code"
            )),
            "binary-sibling hooks/ dir must be probed; got {candidates:?}"
        );
        // Packaged install locations still take precedence.
        assert_eq!(
            candidates[0],
            PathBuf::from("/usr/local/share/ai-memory/hooks/claude-code")
        );
    }

    #[test]
    fn source_candidates_preserve_cargo_run_repo_root() {
        // `cargo run`: target/<profile>/<bin> → repo root holds `hooks/`.
        let exe = PathBuf::from("/home/dev/ai-memory/target/debug/ai-memory");
        let candidates = source_candidates(None, "claude-code", Some(exe));
        assert!(
            candidates.contains(&PathBuf::from("/home/dev/ai-memory/hooks/claude-code")),
            "repo-root hooks/ dir must still be probed; got {candidates:?}"
        );
    }

    #[test]
    fn source_candidates_honour_explicit_override() {
        let candidates = source_candidates(Some(Path::new("/custom/src")), "codex", None);
        assert_eq!(candidates, vec![PathBuf::from("/custom/src/codex")]);
    }
}

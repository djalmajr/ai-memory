//! Command-line interface definition (clap derive).

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

/// Top-level CLI for the `ai-memory` binary.
#[derive(Debug, Parser)]
#[command(name = "ai-memory", version, about, long_about = None)]
pub struct Cli {
    /// Override the data directory.
    ///
    /// Defaults to a platform path under `dirs::data_local_dir()`. Also
    /// settable via the `AI_MEMORY_DATA_DIR` environment variable.
    #[arg(long, env = "AI_MEMORY_DATA_DIR", global = true)]
    pub data_dir: Option<PathBuf>,

    /// Path to an explicit config file (defaults to `<data_dir>/config.toml`).
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// Subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Initialise the data directory layout.
    Init(InitArgs),
    /// Print runtime status (counts, paths, version).
    Status(StatusArgs),
    /// Full-text search the wiki via FTS5.
    Search(SearchArgs),
    /// Write or update a wiki page atomically (also indexes it in the store).
    WritePage(WritePageArgs),
    /// Run the filesystem watcher (foreground; Ctrl-C to stop).
    Watch(WatchArgs),
    /// Run the MCP server (with watcher) over stdio or HTTP.
    Serve(ServeArgs),
    /// Wipe the data directory's wiki/, db/, raw/ contents.
    Reset(ResetArgs),
    /// Snapshot wiki/, db/, and config.toml into a gzipped tarball.
    Backup(BackupArgs),
    /// Restore a backup tarball into the data directory.
    Restore(RestoreArgs),
    /// Print (or apply) lifecycle-hook configuration for an agent CLI.
    InstallHooks(InstallHooksArgs),
    /// Print MCP server registration snippets for any supported client
    /// (Claude Code, Codex, OpenCode, Cursor, Claude Desktop, Gemini
    /// CLI, OpenClaw, pi). See docs/mcp-install.md for the full guide.
    InstallMcp(InstallMcpArgs),
    /// Stage + commit the wiki tree under git.
    Commit(CommitArgs),
    /// Smoke-test an LLM provider by sending one prompt.
    LlmTest(LlmTestArgs),
    /// Run the M8 retention sweep over episodic pages.
    ForgetSweep(ForgetSweepArgs),
    /// Run the M8 lint pass (stale / duplicates + optional LLM contradiction).
    Lint(LintArgs),
    /// Compute + store embeddings for every latest page (M9).
    Embed(EmbedArgs),
    /// Generate a random hex bearer token for AI_MEMORY_AUTH_TOKEN.
    GenerateAuthToken(GenerateAuthTokenArgs),
    /// One-shot agent setup for docker deploys: extract the bundled
    /// hook scripts to a host-mounted directory AND print the matching
    /// config snippet. Replaces the "clone the repo + cargo build"
    /// workflow for users who never want a local Rust toolchain.
    SetupAgent(SetupAgentArgs),
    /// Pre-load an existing project's history into the wiki by
    /// LLM-summarising git log, README, docs/, and module headers
    /// into seed wiki pages. Run once when adopting ai-memory in a
    /// project that's been around for a while. Requires
    /// AI_MEMORY_LLM_PROVIDER configured on the server.
    Bootstrap(BootstrapArgs),
    /// Install the ai-memory usage snippet into the project's
    /// CLAUDE.md / AGENTS.md (or any markdown file you specify).
    /// Idempotent — bracketed by `<!-- ai-memory:start -->` /
    /// `<!-- ai-memory:end -->` markers so re-running replaces the
    /// block in place without duplicating.
    InstallInstructions(InstallInstructionsArgs),
}

/// Arguments for `install-instructions`.
#[derive(Debug, Args)]
pub struct InstallInstructionsArgs {
    /// Markdown file to write into. Defaults to `$PWD/CLAUDE.md`.
    /// Use `--target AGENTS.md` for non-Claude agents, or any other
    /// path for project-rules files (`.cursor/rules`,
    /// `.windsurfrules`, etc.).
    #[arg(long, default_value = "CLAUDE.md")]
    pub target: PathBuf,
    /// Print the snippet to stdout instead of mutating the file.
    /// The default IS mutation here (the print form is also
    /// available without this command — copy the block from the
    /// README). Pass `--print` to preview what would land in
    /// the file.
    #[arg(long)]
    pub print: bool,
}

/// Arguments for `bootstrap`.
#[derive(Debug, Args)]
pub struct BootstrapArgs {
    /// Path of the project repo whose history we should ingest.
    /// Defaults to `git rev-parse --show-toplevel` resolved from the
    /// current directory (so running bootstrap from any subdir of the
    /// project works).
    #[arg(long)]
    pub repo_path: Option<PathBuf>,
    /// Workspace + project labels stamped on the generated pages.
    /// Match what your `ai-memory serve` uses so the bootstrap pages
    /// live in the same wiki silo as future captured sessions.
    #[arg(long, default_value = "default")]
    pub workspace: String,
    #[arg(long, default_value = "scratch")]
    pub project: String,
    /// Maximum total tokens of source text sent to the LLM in one
    /// run. When the collected sources exceed this, lower-priority
    /// inputs (older git commits, then code module headers, then
    /// docs) are dropped first. Default is 150K — comfortably under
    /// Haiku/Sonnet 4.5's 200K context (leaves room for the ~64K
    /// output budget) — so the model sees as much of your project
    /// as possible. Lower it explicitly only if you're cost-
    /// sensitive or running against a smaller-context provider.
    #[arg(long, default_value_t = 150_000)]
    pub max_input_tokens: usize,
    /// Skip git-commit history ingestion.
    #[arg(long)]
    pub exclude_git: bool,
    /// Skip README ingestion.
    #[arg(long)]
    pub exclude_readme: bool,
    /// Skip docs/**/*.md ingestion.
    #[arg(long)]
    pub exclude_docs: bool,
    /// Skip code module headers (Rust `//!` doc-comments at the top
    /// of `**/*.rs` files).
    #[arg(long)]
    pub exclude_code: bool,
    /// git-log time filter (passed through to `git log --since`).
    /// Useful when a repo is years old and you only want recent
    /// history (e.g. `--since "180 days ago"`). Default: no limit.
    #[arg(long)]
    pub since: Option<String>,
    /// LLM-call dry run: collects sources, builds the prompt, and
    /// prints what *would* be sent — but never calls the provider
    /// and never writes to the wiki. Useful for verifying the source
    /// selection before paying for a real run.
    #[arg(long)]
    pub dry_run: bool,
    /// Re-bootstrap a project that already has a bootstrap manifest
    /// page. Without this flag, `bootstrap` refuses to run twice on
    /// the same project (the manifest is `wiki/bootstrap.md`).
    #[arg(long)]
    pub force: bool,
}

/// Arguments for `setup-agent`.
#[derive(Debug, Args)]
pub struct SetupAgentArgs {
    /// Which agent's hook bundle to extract + render.
    #[arg(long, value_enum, default_value_t = AgentChoice::ClaudeCode)]
    pub agent: AgentChoice,
    /// Filesystem directory the hook scripts get copied into. In a
    /// docker context this is the in-container path; mount a host
    /// directory there. Example:
    ///     docker run --rm -v $HOME/.ai-memory:/host ai-memory \
    ///       setup-agent --to /host/hooks ...
    #[arg(long)]
    pub to: PathBuf,
    /// Directory the rendered config JSON should reference for the
    /// hook commands. Defaults to `--to`. Set this when the path on
    /// the host (where the agent CLI runs) differs from the in-
    /// container path. Example:
    ///     --to /host/hooks  --host-prefix $HOME/.ai-memory/hooks
    #[arg(long)]
    pub host_prefix: Option<PathBuf>,
    /// MCP / hook ingress URL the agent should POST to.
    #[arg(long, default_value = "http://127.0.0.1:49374")]
    pub server_url: String,
    /// Bearer token embedded into each hook's env block. Picked up
    /// from `AI_MEMORY_AUTH_TOKEN` if unset.
    #[arg(long, env = "AI_MEMORY_AUTH_TOKEN", hide_env_values = true)]
    pub auth_token: Option<String>,
    /// Source directory for the embedded hook bundle. Defaults to
    /// `/usr/local/share/ai-memory/hooks` (the docker image's
    /// bundled path) and falls back to a repo-local `hooks/` for
    /// `cargo run setup-agent` during development.
    #[arg(long)]
    pub source: Option<PathBuf>,
}

/// Arguments for `generate-auth-token`.
#[derive(Debug, Args)]
pub struct GenerateAuthTokenArgs {
    /// Number of random bytes of entropy. The token printed is hex-
    /// encoded, so the output length is 2× this value. 32 bytes
    /// (256 bits) is plenty for any homelab threat model.
    #[arg(long, default_value_t = 32)]
    pub bytes: usize,
}

/// Arguments for `init`.
#[derive(Debug, Args)]
pub struct InitArgs {
    /// Overwrite an existing `config.toml` if present.
    #[arg(long)]
    pub force: bool,
}

/// Arguments for `status`.
#[derive(Debug, Args)]
pub struct StatusArgs {
    /// Emit the report as JSON instead of human-readable text.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `search`.
#[derive(Debug, Args)]
pub struct SearchArgs {
    /// FTS5 query string (e.g. `"karpathy wiki"` or `quick OR slow`).
    pub query: String,
    /// Maximum number of hits to return.
    #[arg(short = 'n', long, default_value_t = 10)]
    pub limit: usize,
    /// Emit results as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `watch`.
#[derive(Debug, Args)]
pub struct WatchArgs {
    /// Workspace name to attribute discovered files to (auto-created).
    #[arg(long, default_value = "default")]
    pub workspace: String,
    /// Project name within the workspace (auto-created).
    #[arg(long, default_value = "scratch")]
    pub project: String,
}

/// Arguments for `reset`.
#[derive(Debug, Args)]
pub struct ResetArgs {
    /// Required to actually wipe data. Without this we just dry-run.
    #[arg(long)]
    pub confirm: bool,
}

/// Arguments for `backup`.
#[derive(Debug, Args)]
pub struct BackupArgs {
    /// Destination tarball (`.tar.gz`).
    #[arg(long, short = 'o')]
    pub to: PathBuf,
}

/// Arguments for `restore`.
#[derive(Debug, Args)]
pub struct RestoreArgs {
    /// Source tarball.
    #[arg(long, short = 'i')]
    pub from: PathBuf,
    /// Overwrite an existing non-empty data dir.
    #[arg(long)]
    pub force: bool,
}

/// Agent CLI to install hooks for. Only the three with lifecycle
/// hooks are listed; for MCP-only clients (Cursor, Claude Desktop,
/// Gemini CLI, OpenClaw), use `install-mcp --client <name>` instead.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum AgentChoice {
    /// Anthropic Claude Code.
    ClaudeCode,
    /// OpenAI Codex CLI.
    Codex,
    /// OpenCode (open-source coding agent).
    OpenCode,
}

/// MCP client to render configuration for. Includes both the
/// hook-capable agents (Claude Code / Codex / OpenCode — same MCP
/// surface, also covered by `install-hooks`) and the MCP-only
/// clients researched in docs/mcp-install.md.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum McpClient {
    /// Anthropic Claude Code — `claude mcp add`.
    ClaudeCode,
    /// OpenAI Codex CLI — `~/.codex/config.toml`.
    Codex,
    /// OpenCode — `opencode.json`.
    OpenCode,
    /// Cursor IDE — `~/.cursor/mcp.json` or `.cursor/mcp.json`.
    Cursor,
    /// Anthropic Claude Desktop — uses the `mcp-remote` stdio shim
    /// to talk to ai-memory's HTTP endpoint (Claude Desktop's JSON
    /// config does not register HTTP transports directly).
    ClaudeDesktop,
    /// Google Gemini CLI — `~/.gemini/settings.json`.
    GeminiCli,
    /// OpenClaw personal AI gateway — `~/.openclaw/config.json`.
    Openclaw,
    /// Mario Zechner's `pi` coding agent. NOT supported via MCP
    /// upstream; this prints the explanation + alternatives.
    Pi,
}

/// Arguments for `commit`.
#[derive(Debug, Args)]
pub struct CommitArgs {
    /// Commit message.
    #[arg(long, short = 'm', default_value = "manual commit")]
    pub message: String,
}

/// LLM provider for `llm-test`.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum LlmProviderChoice {
    /// Anthropic Messages API.
    Anthropic,
    /// OpenAI Chat Completions.
    Openai,
    /// OpenAI-compatible local (Ollama, vLLM, LM Studio).
    OpenaiCompat,
}

/// Arguments for `embed`.
#[derive(Debug, Args)]
pub struct EmbedArgs {
    /// Report what would be embedded without actually mutating.
    #[arg(long)]
    pub dry_run: bool,
    /// Re-embed pages even when they already have a row with the
    /// currently-configured `(provider, model, dim)`.
    #[arg(long)]
    pub force: bool,
    /// Workspace name (auto-created if absent).
    #[arg(long, default_value = "default")]
    pub workspace: String,
    /// Project name within the workspace (auto-created if absent).
    #[arg(long, default_value = "scratch")]
    pub project: String,
}

/// Arguments for `forget-sweep`.
#[derive(Debug, Args)]
pub struct ForgetSweepArgs {
    /// Report what would be evicted without actually mutating.
    #[arg(long)]
    pub dry_run: bool,
}

/// Arguments for `lint`.
#[derive(Debug, Args)]
pub struct LintArgs {
    /// Compute findings but don't write `wiki/_lint/<date>.md`.
    #[arg(long)]
    pub dry_run: bool,
    /// Workspace name (auto-created if absent).
    #[arg(long, default_value = "default")]
    pub workspace: String,
    /// Project name within the workspace (auto-created if absent).
    #[arg(long, default_value = "scratch")]
    pub project: String,
}

/// Arguments for `llm-test`.
#[derive(Debug, Args)]
pub struct LlmTestArgs {
    /// Provider to test.
    #[arg(long, value_enum)]
    pub provider: LlmProviderChoice,
    /// Model identifier (e.g. `claude-sonnet-4-6`, `gpt-4o-mini`, `llama3.1:8b`).
    #[arg(long)]
    pub model: String,
    /// Prompt to send.
    #[arg(long)]
    pub prompt: String,
    /// Base URL override (required for openai-compat).
    #[arg(long, env = "LLM_BASE_URL")]
    pub base_url: Option<String>,
    /// Optional API key override (otherwise pulled from env).
    #[arg(long, env = "LLM_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
}

/// Arguments for `install-hooks`.
#[derive(Debug, Args)]
pub struct InstallHooksArgs {
    /// Which agent's hooks to render.
    #[arg(long, value_enum, default_value_t = AgentChoice::ClaudeCode)]
    pub agent: AgentChoice,
    /// Filesystem root that contains the vendored hook scripts (defaults
    /// to the repo's `hooks/` if known, else `/usr/local/share/ai-memory/hooks`).
    #[arg(long)]
    pub hooks_dir: Option<PathBuf>,
    /// Server URL the hooks will POST to.
    #[arg(long, default_value = "http://127.0.0.1:49374")]
    pub server_url: String,
    /// Bearer token to embed in the hook config's `env` block. When
    /// set, every hook call carries `Authorization: Bearer <token>`,
    /// matching what the server requires when AI_MEMORY_AUTH_TOKEN
    /// is set there. Generate one with `ai-memory generate-auth-token`.
    #[arg(long, env = "AI_MEMORY_AUTH_TOKEN", hide_env_values = true)]
    pub auth_token: Option<String>,
    /// **Mutate** ~/.claude/settings.json in place instead of just
    /// printing the snippet. Idempotent — replaces the seven hook
    /// entries (SessionStart, UserPromptSubmit, …) and preserves
    /// any other hook config the user has. Only supported for
    /// `--agent claude-code` today; Codex / OpenCode hook config
    /// formats are still in flux upstream, so they keep the print
    /// path. A timestamped backup is written next to the original.
    #[arg(long)]
    pub apply: bool,
    /// Override the settings.json path (auto-detected as
    /// `~/.claude/settings.json`).
    #[arg(long)]
    pub config_file: Option<PathBuf>,
}

/// Arguments for `install-mcp`.
#[derive(Debug, Args)]
pub struct InstallMcpArgs {
    /// Which MCP client to render configuration for.
    #[arg(long, value_enum, default_value_t = McpClient::ClaudeCode)]
    pub client: McpClient,
    /// MCP HTTP endpoint URL the client should connect to.
    #[arg(long, default_value = "http://127.0.0.1:49374/mcp")]
    pub server_url: String,
    /// Friendly name the client should show for this server entry.
    #[arg(long, default_value = "ai-memory")]
    pub name: String,
    /// Bearer token to embed in the client config. When set, the
    /// rendered snippet includes an `Authorization: Bearer <token>`
    /// header so the client can authenticate against a server that
    /// requires it. Picked up from `AI_MEMORY_AUTH_TOKEN` if unset.
    #[arg(long, env = "AI_MEMORY_AUTH_TOKEN", hide_env_values = true)]
    pub auth_token: Option<String>,
    /// **Mutate** the client's config file in place instead of just
    /// printing the snippet. Idempotent: replaces any existing entry
    /// named `<name>` (default `ai-memory`); preserves every other
    /// MCP server the user has configured. A timestamped backup is
    /// written next to the original before each modifying write.
    #[arg(long)]
    pub apply: bool,
    /// Override the config-file path. Auto-detected per client when
    /// absent (e.g. `~/.claude/settings.json` for Claude Code).
    #[arg(long)]
    pub config_file: Option<PathBuf>,
}

/// Transport for the MCP server.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum TransportKind {
    /// Stdio — what `claude mcp add` uses.
    Stdio,
    /// Streamable HTTP — for HTTP clients and `mcp-inspector`.
    Http,
}

/// Arguments for `serve`.
#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Transport to expose the MCP server on.
    #[arg(long, value_enum, default_value_t = TransportKind::Stdio)]
    pub transport: TransportKind,
    /// Bind address for `--transport http` (default: from config).
    #[arg(long)]
    pub bind: Option<String>,
    /// Skip the filesystem watcher; useful for transient debugging.
    #[arg(long)]
    pub no_watcher: bool,
    /// Workspace name (auto-created).
    #[arg(long, default_value = "default")]
    pub workspace: String,
    /// Project name within the workspace (auto-created).
    #[arg(long, default_value = "scratch")]
    pub project: String,
}

/// Arguments for `write-page`.
#[derive(Debug, Args)]
pub struct WritePageArgs {
    /// Relative wiki path (e.g. `notes/foo.md`).
    #[arg(long, visible_alias = "p")]
    pub path: String,
    /// Markdown body. Use `-` to read from stdin.
    #[arg(long, visible_alias = "b")]
    pub body: String,
    /// Optional page title; otherwise derived from the first `# heading`
    /// in the body, or the path stem.
    #[arg(long)]
    pub title: Option<String>,
    /// Repeatable tag to add to the frontmatter `tags` array.
    #[arg(long, short = 't')]
    pub tag: Vec<String>,
    /// Tier (`working`, `episodic`, `semantic`, `procedural`).
    #[arg(long, default_value = "semantic")]
    pub tier: String,
    /// Pin the page so the future decay sweep skips it.
    #[arg(long)]
    pub pinned: bool,
    /// Workspace name (auto-created if absent).
    #[arg(long, default_value = "default")]
    pub workspace: String,
    /// Project name within the workspace (auto-created if absent).
    #[arg(long, default_value = "scratch")]
    pub project: String,
}

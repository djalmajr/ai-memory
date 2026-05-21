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

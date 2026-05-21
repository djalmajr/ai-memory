//! `ai-memory` binary entry point.
//!
//! Loads configuration once at startup, initialises tracing, then dispatches
//! to the requested subcommand. Domain crates take `&Config` by reference;
//! there is no global state, no `lazy_static`, no second config-read path
//! (lesson from agentmemory #456 / #469).

#![doc(html_no_source)]

use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tracing::info;

mod cli;
mod commands;
mod config;
mod logging;
mod process_guard;

use cli::{Cli, Command};
use config::Config;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let config = Arc::new(Config::load(cli.config.as_deref(), cli.data_dir.clone())?);
    let _logging_guard = logging::init(&config)?;

    info!(
        version = env!("CARGO_PKG_VERSION"),
        data_dir = %config.data_dir.display(),
        bind = %config.bind,
        "ai-memory starting",
    );

    match cli.command {
        Command::Init(args) => commands::init::run(&config, args),
        Command::Status(args) => commands::status::run(&config, args).await,
        Command::Search(args) => commands::search::run(&config, args).await,
        Command::WritePage(args) => commands::write_page::run(&config, args).await,
        Command::Watch(args) => commands::watch::run(&config, args).await,
        Command::Serve(args) => commands::serve::run(&config, args).await,
        Command::Reset(args) => commands::reset::run(&config, args),
        Command::Backup(args) => commands::backup::run(&config, args).await,
        Command::Restore(args) => commands::restore::run(&config, args),
    }
}

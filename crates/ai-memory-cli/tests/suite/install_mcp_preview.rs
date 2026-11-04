//! `install-mcp` without `--apply` and `setup-agent` print their config to
//! stdout, which a terminal's scrollback, a CI log or an agent's transcript
//! keeps. No such run may print the token, for any client and from any
//! source: `--auth-token`, the environment, the config file,
//! `[auth].bearer_token_file` or an installed MCP entry. These drive the
//! shipped binary because the renderers print directly.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_ai-memory");
const SERVER_URL: &str = "http://127.0.0.1:49374";
const TOKEN: &str = "aim_FAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfake111";
/// Part of [`TOKEN`] that survives any quoting or escaping a format applies.
const TOKEN_BODY: &str = "FAKEfake";
const PLACEHOLDER: &str = "<redacted>";
/// The stderr note `install-mcp` must print when its snippet holds the
/// placeholder.
const MCP_NOTE: &str = "[ai-memory] token omitted from this output: replace <redacted> with it, or rerun with --apply.";
/// The stderr note `setup-agent` must print when it saw a token.
const SETUP_NOTE: &str =
    "[ai-memory] token omitted from this output: replace any <redacted> with it.";

/// Every `--client` value `install-mcp` accepts, by CLI name.
const MCP_CLIENTS: &[&str] = &[
    "claude-code",
    "codex",
    "open-code",
    "opencode2",
    "cursor",
    "claude-desktop",
    "gemini-cli",
    "openclaw",
    "pi",
    "omp",
    "antigravity-cli",
    "zero",
    "zcode",
    "devin",
    "grok",
    "kimi-code",
    "kiro-cli",
    "command-code",
    "ibm-bob",
    "qwen-code",
    "copilot-cli",
    "droid",
    "swival",
    "vscode-copilot",
    "zed",
    "muse",
    "dsh",
    "hermes",
    "cline",
    "freebuff",
    "prime-agent",
];

/// Entries that never carry the token literal: they name
/// `AI_MEMORY_AUTH_TOKEN` or run the bridge that reads the install's
/// credential. For those, a given token shows neither the token nor the
/// placeholder in the snippet.
const NO_TOKEN_CLIENTS: &[&str] = &["pi", "droid", "hermes", "prime-agent"];

/// `setup-agent` agents that print a config or stage scripts. `open-code`
/// needs `--opencode-dialect`, because the container cannot probe the host
/// `opencode` binary.
const SETUP_AGENTS: &[&str] = &["zero", "zcode", "droid", "claude-code", "open-code", "pi"];
/// The agents whose printed config embeds the token, and so the
/// placeholder, when a token is given. Pi's guidance prints the literal
/// `<token>` it tells the operator to fill in, so it holds neither.
const SETUP_AGENTS_WITH_PLACEHOLDER: &[&str] =
    &["zero", "zcode", "droid", "claude-code", "open-code"];

struct Sandbox {
    _tmp: tempfile::TempDir,
    home: PathBuf,
    data_dir: PathBuf,
    config_path: PathBuf,
    bin_dir: PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().to_path_buf();
        let bin_dir = home.join("bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        fake_opencode_v1(&bin_dir);
        Self {
            data_dir: home.join("data"),
            config_path: home.join("config.toml"),
            _tmp: tmp,
            home,
            bin_dir,
        }
    }

    /// Nothing is inherited, so neither a developer's `AI_MEMORY_*` nor a
    /// harness home relocation can reach a real credential.
    fn command(&self) -> Command {
        let mut cmd = Command::new(BIN);
        cmd.env_clear()
            .current_dir(&self.home)
            // On Windows `dirs::home_dir` ignores HOME and USERPROFILE and asks
            // the OS, so this is what keeps the binary out of the real home.
            .env("AI_MEMORY_HOME", &self.home)
            .env("HOME", &self.home)
            .env("USERPROFILE", &self.home)
            .env("XDG_CONFIG_HOME", self.home.join(".config"))
            .env("XDG_DATA_HOME", self.home.join(".local/share"))
            .env("APPDATA", self.home.join("AppData/Roaming"))
            .env("LOCALAPPDATA", self.home.join("AppData/Local"))
            .env("PATH", &self.bin_dir);
        if let Some(root) = std::env::var_os("SYSTEMROOT") {
            cmd.env("SYSTEMROOT", root);
        }
        cmd.arg("--data-dir")
            .arg(&self.data_dir)
            .arg("--config")
            .arg(&self.config_path);
        cmd
    }

    fn install_mcp(&self, client: &str) -> Command {
        let mut cmd = self.command();
        cmd.args([
            "install-mcp",
            "--client",
            client,
            "--server-url",
            SERVER_URL,
        ]);
        cmd
    }

    fn setup_agent(&self, agent: &str, to: &Path) -> Command {
        let mut cmd = self.command();
        cmd.args([
            "setup-agent",
            "--agent",
            agent,
            "--to",
            to.display().to_string().as_str(),
            "--server-url",
            SERVER_URL,
            // The repo's hook bundle, so staging works wherever the test
            // binary lives.
            "--source",
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../hooks")
                .display()
                .to_string()
                .as_str(),
        ]);
        if agent == "open-code" {
            cmd.args(["--opencode-dialect", "v1"]);
        }
        cmd
    }

    fn write_config(&self, body: &str) {
        std::fs::write(&self.config_path, body).unwrap();
    }
}

/// `--client open-code` probes `opencode --version` for its plugin dialect.
fn fake_opencode_v1(bin_dir: &Path) {
    #[cfg(windows)]
    std::fs::write(bin_dir.join("opencode.cmd"), "@echo 1.18.34\r\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let script = bin_dir.join("opencode");
        std::fs::write(&script, "#!/bin/sh\nprintf '%s\\n' '1.18.34'\n").unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
}

fn combined(output: &Output) -> String {
    format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// Run a preview and fail on a leak in either stream; return `(stdout,
/// stderr)` for the shape assertions.
fn clean(what: &str, mut cmd: Command) -> (String, String) {
    let output = cmd.output().unwrap();
    let all = combined(&output);
    assert!(output.status.success(), "{what} failed: {all}");
    assert!(
        !all.contains(TOKEN_BODY),
        "{what} printed the token:\n{all}"
    );
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// The token must not appear in any file under `root`, recursively.
fn assert_no_token_under(what: &str, root: &Path) {
    if !root.is_dir() {
        return;
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let body = std::fs::read_to_string(&path).unwrap_or_default();
                assert!(
                    !body.contains(TOKEN_BODY),
                    "{what}: {path:?} carries the token"
                );
            }
        }
    }
}

/// Every client, with `--auth-token`: the token stays out of both streams.
#[test]
fn mcp_preview_keeps_the_token_out_of_every_client() {
    let sandbox = Sandbox::new();
    for client in MCP_CLIENTS {
        let what = format!("install-mcp --client {client} with --auth-token");
        let mut cmd = sandbox.install_mcp(client);
        cmd.args(["--auth-token", TOKEN]);
        clean(&what, cmd);
    }
}

/// The clients whose entry embeds the token show the placeholder and the
/// exact stderr note in its place; the entries that never carry the token
/// show neither, so the note is not printed for them.
#[test]
fn mcp_previews_show_the_placeholder_where_the_token_would_be() {
    let sandbox = Sandbox::new();
    for client in MCP_CLIENTS {
        let what = format!("install-mcp --client {client} with --auth-token");
        let mut cmd = sandbox.install_mcp(client);
        cmd.args(["--auth-token", TOKEN]);
        let (stdout, stderr) = clean(&what, cmd);
        let holds_token = !NO_TOKEN_CLIENTS.contains(client);
        assert_eq!(
            stdout.contains(PLACEHOLDER),
            holds_token,
            "{client}: the placeholder must appear exactly when the entry embeds the token:\n{stdout}"
        );
        assert_eq!(
            stderr.contains(MCP_NOTE),
            holds_token,
            "{client}: the omission note must appear exactly when the entry embeds the token:\n{stderr}"
        );
    }
}

/// The sources a token may come from, on the three clients the brief
/// names: `--auth-token` is covered by the tests above; here the
/// environment, the inline config key, the token file, and a hand-planted
/// MCP entry.
#[test]
fn mcp_previews_redact_the_token_from_every_source() {
    for client in ["claude-code", "cursor", "kimi-code"] {
        let sandbox = Sandbox::new();

        let mut cmd = sandbox.install_mcp(client);
        cmd.env("AI_MEMORY_AUTH_TOKEN", TOKEN);
        let (stdout, stderr) = clean(&format!("{client} with AI_MEMORY_AUTH_TOKEN"), cmd);
        assert!(stdout.contains(PLACEHOLDER), "{client}:\n{stdout}");
        assert!(stderr.contains(MCP_NOTE), "{client}:\n{stderr}");

        sandbox.write_config(&format!("[auth]\nbearer_token = \"{TOKEN}\"\n"));
        let (stdout, stderr) = clean(
            &format!("{client} with [auth].bearer_token"),
            sandbox.install_mcp(client),
        );
        assert!(stdout.contains(PLACEHOLDER), "{client}:\n{stdout}");
        assert!(stderr.contains(MCP_NOTE), "{client}:\n{stderr}");

        // The token file routes the entry to the bridge instead of
        // inlining a literal, so no placeholder either: the token must
        // simply stay out.
        let token_file = sandbox.home.join("root-token");
        std::fs::write(&token_file, TOKEN).unwrap();
        sandbox.write_config(&format!(
            "[auth]\nbearer_token_file = {:?}\n",
            token_file.display().to_string()
        ));
        let (stdout, stderr) = clean(
            &format!("{client} with [auth].bearer_token_file"),
            sandbox.install_mcp(client),
        );
        assert!(!stdout.contains(PLACEHOLDER), "{client}:\n{stdout}");
        assert!(!stderr.contains(MCP_NOTE), "{client}:\n{stderr}");
    }
}

/// A hand-planted MCP entry holds the token inside the sandbox's home.
/// `install-mcp` only infers from such an entry for OpenCode: the other
/// clients must not pull the token in at all, and OpenCode must redact
/// the inferred one.
#[test]
fn planted_mcp_entry_stays_out_of_the_preview() {
    let json_entry = format!(
        "{{\"mcpServers\":{{\"ai-memory\":{{\"url\":\"{SERVER_URL}/mcp\",\"headers\":{{\"Authorization\":\"Bearer {TOKEN}\"}}}}}}}}"
    );
    for (client, relative) in [
        ("claude-code", ".claude.json"),
        ("cursor", ".cursor/mcp.json"),
        ("kimi-code", ".kimi-code/mcp.json"),
    ] {
        let sandbox = Sandbox::new();
        let path = sandbox.home.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &json_entry).unwrap();

        let what = format!("{client} with a planted MCP entry");
        let (stdout, stderr) = clean(&what, sandbox.install_mcp(client));
        assert!(
            !stdout.contains(PLACEHOLDER),
            "{client}: the preview saw no token, so it must not show the placeholder:\n{stdout}"
        );
        assert!(!stderr.contains(MCP_NOTE), "{client}:\n{stderr}");
    }

    let sandbox = Sandbox::new();
    let path = sandbox
        .home
        .join(".config")
        .join("opencode")
        .join("opencode.json");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        &path,
        format!(
            "{{\"mcp\":{{\"ai-memory\":{{\"type\":\"remote\",\"url\":\"{SERVER_URL}/mcp\",\"enabled\":true,\"headers\":{{\"Authorization\":\"Bearer {TOKEN}\"}}}}}}}}"
        ),
    )
    .unwrap();
    let mut cmd = sandbox.command();
    cmd.args(["install-mcp", "--client", "open-code"]);
    let (stdout, stderr) = clean("open-code with an inferred MCP token", cmd);
    assert!(
        stdout.contains(PLACEHOLDER),
        "the inferred token must come out as the placeholder:\n{stdout}"
    );
    assert!(stderr.contains(MCP_NOTE), "{stderr}");
}

/// `setup-agent` prints the config and, for the script agents, stages
/// scripts under `--to`: the token must stay out of both streams and of
/// every staged file.
#[test]
fn setup_agent_keeps_the_token_out_of_output_and_staged_files() {
    let sandbox = Sandbox::new();
    let staged = sandbox.home.join("staged");
    for agent in SETUP_AGENTS {
        let what = format!("setup-agent --agent {agent} with --auth-token");
        let mut cmd = sandbox.setup_agent(agent, &staged);
        cmd.args(["--auth-token", TOKEN]);
        let (stdout, stderr) = clean(&what, cmd);
        assert!(
            stderr.contains(SETUP_NOTE),
            "{agent}: the omission note must say where the token went:\n{stderr}"
        );
        assert_eq!(
            stdout.contains(PLACEHOLDER),
            SETUP_AGENTS_WITH_PLACEHOLDER.contains(agent),
            "{agent}: the placeholder must appear exactly when the printed config embeds the token:\n{stdout}"
        );
    }
    assert_no_token_under("setup-agent staged files", &staged);
    assert!(
        staged.join("claude-code").is_dir(),
        "the claude-code bundle must have been staged"
    );
}

/// The environment and the config file are token sources for
/// `setup-agent` too.
#[test]
fn setup_agent_redacts_the_token_from_the_other_sources() {
    for client in ["zero", "claude-code"] {
        let sandbox = Sandbox::new();
        let staged = sandbox.home.join("staged");

        let mut cmd = sandbox.setup_agent(client, &staged);
        cmd.env("AI_MEMORY_AUTH_TOKEN", TOKEN);
        let (_stdout, stderr) = clean(
            &format!("setup-agent --agent {client} with the env token"),
            cmd,
        );
        assert!(stderr.contains(SETUP_NOTE), "{client}:\n{stderr}");

        sandbox.write_config(&format!("[auth]\nbearer_token = \"{TOKEN}\"\n"));
        let (stdout, stderr) = clean(
            &format!("setup-agent --agent {client} with [auth].bearer_token"),
            sandbox.setup_agent(client, &staged),
        );
        assert!(stderr.contains(SETUP_NOTE), "{client}:\n{stderr}");
        if client == "zero" {
            assert!(stdout.contains(PLACEHOLDER), "{client}:\n{stdout}");
        }
    }
}

/// Control: without a token, neither the placeholder nor either note
/// appears, and the preview still names the sandbox as its home.
#[test]
fn preview_without_a_token_announces_none() {
    let sandbox = Sandbox::new();
    for client in ["claude-code", "cursor", "kimi-code", "codex", "opencode2"] {
        let (stdout, stderr) = clean(client, sandbox.install_mcp(client));
        assert!(!stdout.contains(PLACEHOLDER), "{client}:\n{stdout}");
        assert!(!stderr.contains(MCP_NOTE), "{client}:\n{stderr}");
    }
    let claude = clean("claude-code", sandbox.install_mcp("claude-code")).0;
    assert!(
        claude.contains(&format!(
            "you'd rather edit {} directly",
            sandbox.home.join(".claude.json").display()
        )),
        "the binary must resolve the sandbox as its home:\n{claude}"
    );

    let staged = sandbox.home.join("staged");
    for agent in ["zero", "claude-code"] {
        let (stdout, stderr) = clean(agent, sandbox.setup_agent(agent, &staged));
        assert!(!stdout.contains(PLACEHOLDER), "{agent}:\n{stdout}");
        assert!(!stderr.contains(SETUP_NOTE), "{agent}:\n{stderr}");
    }
}

/// Control: `token_env_var` is `#[arg(skip)]` — "set by the installer,
/// never a flag" — so the CLI-reachable case is a token from
/// `[auth].bearer_token_file`: the codex snippet names the variable and
/// shows no placeholder.
#[test]
fn codex_token_env_var_shows_no_placeholder() {
    let sandbox = Sandbox::new();
    let token_file = sandbox.home.join("root-token");
    std::fs::write(&token_file, TOKEN).unwrap();
    sandbox.write_config(&format!(
        "[auth]\nbearer_token_file = {:?}\n",
        token_file.display().to_string()
    ));
    let (stdout, stderr) = clean(
        "codex with a bearer token file",
        sandbox.install_mcp("codex"),
    );
    assert!(stdout.contains("bearer_token_env_var"), "{stdout}");
    assert!(!stdout.contains(PLACEHOLDER), "{stdout}");
    assert!(!stderr.contains(MCP_NOTE), "{stderr}");
}

/// Control: `--apply` still writes the real token to the file the
/// operator named; the preview redaction must not reach the write path.
#[test]
fn apply_still_writes_the_token_to_the_named_config_file() {
    let sandbox = Sandbox::new();
    let config = sandbox.home.join("cursor-mcp.json");
    std::fs::write(&config, "{}\n").unwrap();
    let mut cmd = sandbox.install_mcp("cursor");
    cmd.args(["--auth-token", TOKEN, "--apply", "--config-file"]);
    cmd.arg(&config);
    clean("cursor --apply", cmd);
    let written = std::fs::read_to_string(&config).unwrap();
    assert!(
        written.contains(TOKEN),
        "the applied file must hold the real token:\n{written}"
    );
}

/// A client TOML whose second line is a syntax error, with the bearer on
/// that line. `toml_edit`'s own display repeats the offending line as a
/// margin snippet, so an error that carries it prints the bearer to a
/// log or transcript.
fn malformed_client_toml() -> String {
    format!(
        "[mcp_servers.ai-memory]\n\
         http_headers = {{ Authorization = \"Bearer {TOKEN}\" }} trailing\n"
    )
}

/// The assertions a refused write of a malformed client TOML owes the
/// operator: the failure keeps the position, drops the source line (and
/// with it the bearer), leaves no parser margin bar, and preserves the
/// file.
fn assert_refused_without_the_line(what: &str, config: &Path, body: &str, output: &Output) {
    assert!(
        !output.status.success(),
        "{what}: the malformed client TOML must be refused:\n{}",
        combined(output)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stdout.contains(TOKEN_BODY),
        "{what}: the token reached stdout:\n{stdout}"
    );
    assert!(
        !stderr.contains(TOKEN_BODY),
        "{what}: the token reached stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("line 2"),
        "{what}: the error must keep the line position:\n{stderr}"
    );
    assert!(
        !stderr.contains('|'),
        "{what}: the parser's margin snippet must not be echoed:\n{stderr}"
    );
    assert_eq!(
        std::fs::read_to_string(config).unwrap(),
        body,
        "{what}: the malformed file must be preserved unchanged"
    );
}

/// `install-mcp --apply` (Codex, Grok) and `install-hooks --apply`
/// (Kimi Code) refuse a client TOML that no longer parses. The error may
/// keep the file, line and column — never the offending line, which holds
/// the bearer, and never the parser's margin snippet.
#[test]
fn apply_keeps_the_bearer_out_of_a_malformed_toml_error() {
    for client in ["codex", "grok"] {
        let sandbox = Sandbox::new();
        let config = sandbox.home.join(format!("{client}-config.toml"));
        let body = malformed_client_toml();
        std::fs::write(&config, &body).unwrap();
        let mut cmd = sandbox.install_mcp(client);
        cmd.args(["--apply", "--config-file"]);
        cmd.arg(&config);
        let what = format!("install-mcp --client {client} --apply on a malformed TOML");
        assert_refused_without_the_line(&what, &config, &body, &cmd.output().unwrap());
    }

    let sandbox = Sandbox::new();
    let config = sandbox.home.join("kimi-config.toml");
    let body = malformed_client_toml();
    std::fs::write(&config, &body).unwrap();
    let mut cmd = sandbox.command();
    cmd.args([
        "install-hooks",
        "--agent",
        "kimi-code",
        "--apply",
        "--config-file",
        config.display().to_string().as_str(),
        "--hooks-dir",
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../hooks")
            .display()
            .to_string()
            .as_str(),
        "--server-url",
        SERVER_URL,
    ]);
    let what = "install-hooks --agent kimi-code --apply on a malformed TOML";
    assert_refused_without_the_line(what, &config, &body, &cmd.output().unwrap());
}

/// `auth enable` reports a client whose config no longer parses as a
/// manual change; that reason may keep the position but not the offending
/// line with its bearer.
#[test]
fn auth_enable_keeps_the_bearer_out_of_a_malformed_client_config() {
    let sandbox = Sandbox::new();
    sandbox.write_config(&format!("server_url = \"{SERVER_URL}\"\n"));
    let codex = sandbox.home.join(".codex").join("config.toml");
    std::fs::create_dir_all(codex.parent().unwrap()).unwrap();
    let body = malformed_client_toml();
    std::fs::write(&codex, &body).unwrap();

    let mut cmd = sandbox.command();
    cmd.args(["auth", "enable"]);
    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "auth enable must still succeed and list the client as manual:\n{stdout}\n{stderr}"
    );
    assert!(
        !stdout.contains(TOKEN_BODY),
        "the token reached stdout:\n{stdout}"
    );
    assert!(
        !stderr.contains(TOKEN_BODY),
        "the token reached stderr:\n{stderr}"
    );
    assert!(
        stdout.contains("needs a manual change"),
        "the malformed client must stay listed as a manual change:\n{stdout}"
    );
    assert!(
        stdout.contains("line 2"),
        "the reason must keep the line position:\n{stdout}"
    );
    assert!(
        !stdout.contains('|'),
        "the parser's margin snippet must not be echoed:\n{stdout}"
    );
}

/// Control: the same `--apply` against a valid Codex TOML still writes the
/// real token to the file; only the error path changes.
#[test]
fn apply_still_writes_the_token_to_a_valid_codex_toml() {
    let sandbox = Sandbox::new();
    let config = sandbox.home.join("codex-config.toml");
    std::fs::write(&config, "[mcp_servers]\n").unwrap();
    let mut cmd = sandbox.install_mcp("codex");
    cmd.args(["--auth-token", TOKEN, "--apply", "--config-file"]);
    cmd.arg(&config);
    clean("codex --apply on a valid TOML", cmd);
    let written = std::fs::read_to_string(&config).unwrap();
    assert!(
        written.contains(TOKEN),
        "the applied file must hold the real token:\n{written}"
    );
}

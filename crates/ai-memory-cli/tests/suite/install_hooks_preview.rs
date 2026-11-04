//! `install-hooks` without `--apply` prints to stdout, which a terminal's
//! scrollback, a CI log or an agent's transcript keeps. No such run may print
//! the token, for any agent and from any source: `--auth-token`, the
//! environment, the config file, `[auth].bearer_token_file` or an installed MCP
//! entry. These drive the shipped binary because the renderers print directly.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_ai-memory");
const SERVER_URL: &str = "http://127.0.0.1:49374";
const TOKEN: &str = "aim_FAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfake000";
/// Part of [`TOKEN`] that survives any quoting or escaping a format applies.
const TOKEN_BODY: &str = "FAKEfake";

const TS_NULL_TOKEN: &str = "const TOKEN: string | null = null;";
const TS_NOTE: &str = "the token is omitted from this preview";
const PLACEHOLDER: &str = "<redacted>";

/// Generated TypeScript integrations.
const TS_AGENTS: &[&str] = &["open-code", "opencode2", "pi", "omp", "openclaw"];
/// Hook configs that carry the token in each entry, in JSON or a JSON comment
/// dialect.
const JSON_AGENTS: &[&str] = &[
    "claude-code",
    "grok",
    "zero",
    "zcode",
    "devin",
    "kiro-cli",
    "kiro-cli-v3",
    "droid",
    "muse",
    "codebuddy",
    "ibm-bob",
    "qwen-code",
    "dsh",
];
const YAML_AGENTS: &[&str] = &["hermes", "pool"];
const TOML_AGENTS: &[&str] = &["kimi-code"];
/// Manual script lists, which only name the variable to set.
const MANUAL_AGENTS: &[&str] = &[
    "codex",
    "command-code",
    "cursor",
    "gemini-cli",
    "antigravity-cli",
    "copilot-cli",
];

struct Sandbox {
    _tmp: tempfile::TempDir,
    home: PathBuf,
    data_dir: PathBuf,
    config_path: PathBuf,
    hooks_dir: PathBuf,
    bin_dir: PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().to_path_buf();
        let hooks_dir = home.join("hook-scripts");
        let bin_dir = home.join("bin");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        std::fs::create_dir_all(&bin_dir).unwrap();
        fake_opencode_v1(&bin_dir);
        Self {
            data_dir: home.join("data"),
            config_path: home.join("config.toml"),
            _tmp: tmp,
            home,
            hooks_dir,
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

    fn install_hooks(&self, agent: &str) -> Command {
        // The per-agent script directory only has to exist; a missing script
        // is a warning, so the preview needs no real bundle.
        std::fs::create_dir_all(self.hooks_dir.join(agent)).unwrap();
        let mut cmd = self.command();
        cmd.args([
            "install-hooks",
            "--agent",
            agent,
            "--server-url",
            SERVER_URL,
        ])
        .arg("--hooks-dir")
        .arg(&self.hooks_dir);
        cmd
    }

    fn write_config(&self, body: &str) {
        std::fs::write(&self.config_path, body).unwrap();
    }
}

/// `--agent open-code` probes `opencode --version` for its plugin dialect.
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

/// Run a preview and return its stdout, failing on a leak in either stream.
fn clean_preview(what: &str, mut cmd: Command) -> String {
    let output = cmd.output().unwrap();
    let all = combined(&output);
    assert!(output.status.success(), "{what} failed: {all}");
    assert!(
        !all.contains(TOKEN_BODY),
        "{what} printed the token:\n{all}"
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn assert_ts_preview(what: &str, stdout: &str) {
    assert!(stdout.contains(TS_NULL_TOKEN), "{what}:\n{stdout}");
    assert!(stdout.contains(TS_NOTE), "{what}:\n{stdout}");
}

fn assert_placeholder_preview(what: &str, stdout: &str) {
    assert!(stdout.contains(PLACEHOLDER), "{what}:\n{stdout}");
    assert!(stdout.contains(TS_NOTE), "{what}:\n{stdout}");
    assert!(
        !stdout.contains("sensitive (chmod 600)"),
        "{what} must not call a redacted preview sensitive:\n{stdout}"
    );
}

/// Every agent in `agents` from `--auth-token`, then the first one from the
/// environment and from the config file: one decision covers all of them, and
/// the first keeps each source honest.
fn check_family(agents: &[&str], assert_shape: fn(&str, &str)) {
    let sandbox = Sandbox::new();
    for agent in agents {
        let what = format!("{agent} with --auth-token");
        let mut cmd = sandbox.install_hooks(agent);
        cmd.args(["--auth-token", TOKEN]);
        assert_shape(&what, &clean_preview(&what, cmd));
    }

    let agent = agents[0];
    let what = format!("{agent} with AI_MEMORY_AUTH_TOKEN");
    let mut cmd = sandbox.install_hooks(agent);
    cmd.env("AI_MEMORY_AUTH_TOKEN", TOKEN);
    assert_shape(&what, &clean_preview(&what, cmd));

    sandbox.write_config(&format!("[auth]\nbearer_token = \"{TOKEN}\"\n"));
    let what = format!("{agent} with [auth].bearer_token");
    assert_shape(&what, &clean_preview(&what, sandbox.install_hooks(agent)));
}

#[test]
fn typescript_previews_render_a_null_token() {
    check_family(TS_AGENTS, assert_ts_preview);
}

#[test]
fn json_hook_config_previews_show_the_placeholder() {
    check_family(JSON_AGENTS, assert_placeholder_preview);
}

#[test]
fn yaml_hook_config_previews_show_the_placeholder() {
    check_family(YAML_AGENTS, assert_placeholder_preview);
}

#[test]
fn toml_hook_config_preview_shows_the_placeholder() {
    check_family(TOML_AGENTS, assert_placeholder_preview);
}

#[test]
fn manual_script_lists_only_name_the_variable() {
    check_family(MANUAL_AGENTS, |what, stdout| {
        assert!(
            stdout.contains("omitted from this printout"),
            "{what}:\n{stdout}"
        );
    });
}

#[test]
fn cline_preview_keeps_its_placeholder() {
    check_family(&["cline"], |what, stdout| {
        assert!(stdout.contains(PLACEHOLDER), "{what}:\n{stdout}");
    });
}

#[test]
fn claude_code_project_scope_preview_holds_no_placeholder_either() {
    let sandbox = Sandbox::new();
    let mut cmd = sandbox.install_hooks("claude-code");
    cmd.args(["--auth-token", TOKEN, "--scope", "project"]);
    let stdout = clean_preview("claude-code --scope project", cmd);
    assert!(stdout.contains("NOT embedded below"), "{stdout}");
    assert!(!stdout.contains(PLACEHOLDER), "{stdout}");
}

/// A token read from an installed MCP entry, planted by hand in the sandbox:
/// this file never runs `--apply` against a default path. The placeholder or
/// note proves the inference happened, so the check is not vacuous.
#[test]
fn inferred_mcp_token_stays_out_of_every_family() {
    let json_entry = format!(
        "{{\"mcpServers\":{{\"ai-memory\":{{\"url\":\"{SERVER_URL}/mcp\",         \"headers\":{{\"Authorization\":\"Bearer {TOKEN}\"}}}}}}}}"
    );
    let toml_entry = format!(
        "[mcp_servers.ai-memory]
url = \"{SERVER_URL}/mcp\"
         [mcp_servers.ai-memory.http_headers]
Authorization = \"Bearer {TOKEN}\"
"
    );
    for (agent, relative, body) in [
        ("claude-code", ".claude.json", &json_entry),
        ("cursor", ".cursor/mcp.json", &json_entry),
        ("omp", ".omp/agent/mcp.json", &json_entry),
        ("kimi-code", ".kimi-code/mcp.json", &json_entry),
        ("codex", ".codex/config.toml", &toml_entry),
    ] {
        let sandbox = Sandbox::new();
        let path = sandbox.home.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, body).unwrap();

        let what = format!("{agent} with an inferred MCP token");
        let stdout = clean_preview(&what, sandbox.install_hooks(agent));
        assert!(
            stdout.contains(PLACEHOLDER) || stdout.contains("omitted from this"),
            "{what}: the fixture must make the preview see a token:
{stdout}"
        );
    }
}

#[test]
fn bearer_token_file_stays_out_of_previews() {
    let sandbox = Sandbox::new();
    let token_file = sandbox.home.join("root-token");
    std::fs::write(&token_file, TOKEN).unwrap();
    sandbox.write_config(&format!(
        "[auth]\nbearer_token_file = {:?}\n",
        token_file.display().to_string()
    ));
    for agent in ["opencode2", "claude-code", "hermes", "kimi-code", "codex"] {
        let what = format!("{agent} with [auth].bearer_token_file");
        let stdout = clean_preview(&what, sandbox.install_hooks(agent));
        assert!(!stdout.contains(PLACEHOLDER), "{what}:\n{stdout}");
    }
}

#[test]
fn preview_without_a_token_announces_none() {
    let sandbox = Sandbox::new();
    let claude = clean_preview("claude-code", sandbox.install_hooks("claude-code"));
    assert!(
        claude.contains(&format!(
            "merge into {}",
            sandbox.home.join(".claude").display()
        )),
        "the binary must resolve the sandbox as its home:
{claude}"
    );
    for agent in ["opencode2", "claude-code", "hermes", "kimi-code"] {
        let stdout = clean_preview(agent, sandbox.install_hooks(agent));
        assert!(!stdout.contains(PLACEHOLDER), "{agent}:\n{stdout}");
        assert!(!stdout.contains(TS_NOTE), "{agent}:\n{stdout}");
    }
}

/// Control: `--apply` still stores the same token, and the file it generates
/// stays token-less because of that store rather than because of the preview.
#[test]
fn apply_still_stores_the_token_under_the_data_dir() {
    let sandbox = Sandbox::new();
    let plugin = sandbox.home.join("ai-memory-opencode2.ts");
    let mut cmd = sandbox.install_hooks("opencode2");
    cmd.args(["--auth-token", TOKEN, "--apply", "--config-file"])
        .arg(&plugin);
    let stdout = clean_preview("opencode2 --apply", cmd);
    assert!(!stdout.contains(TS_NOTE), "{stdout}");

    let stored = std::fs::read_to_string(sandbox.data_dir.join("auth-token")).unwrap();
    assert_eq!(stored.trim(), TOKEN);
    let written = std::fs::read_to_string(&plugin).unwrap();
    assert!(written.contains(TS_NULL_TOKEN), "{written}");
}

//! `install-hooks` without `--apply` prints to stdout, which a terminal's
//! scrollback, a CI log or an agent's transcript keeps. No such run may print
//! the token, for any agent and from any source: `--auth-token`, the
//! environment, the config file, `[auth].bearer_token_file` or an installed MCP
//! entry: the preview embeds the `mask_secret` form (`<4-char prefix>...[REDACTED]`),
//! never the full credential. These drive the shipped binary because the
//! renderers print directly.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_ai-memory");
const SERVER_URL: &str = "http://127.0.0.1:49374";
const TOKEN: &str = "aim_FAKEfakeFAKEfakeFAKEfakeFAKEfakeFAKEfake000";
/// Part of [`TOKEN`] that survives any quoting or escaping a format applies.
const TOKEN_BODY: &str = "FAKEfake";

const TS_NULL_TOKEN: &str = "const TOKEN: string | null = null;";
/// The `mask_secret` form of [`TOKEN`]: a 4-character prefix plus the marker.
const MASKED_TOKEN: &str = "aim_...[REDACTED]";
/// What a generated TypeScript preview prints in place of [`TOKEN`].
const TS_MASKED_TOKEN: &str = "const TOKEN: string | null = \"aim_...[REDACTED]\";";
/// Cline's preview redacts the token with its own marker (upstream behavior).
const CLINE_REDACTED: &str = "<redacted>";

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
    assert!(stdout.contains(TS_MASKED_TOKEN), "{what}:\n{stdout}");
}

fn assert_masked_preview(what: &str, stdout: &str) {
    assert!(stdout.contains(MASKED_TOKEN), "{what}:\n{stdout}");
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
fn json_hook_config_previews_show_the_masked_token() {
    check_family(JSON_AGENTS, assert_masked_preview);
}

#[test]
fn yaml_hook_config_previews_show_the_masked_token() {
    check_family(YAML_AGENTS, assert_masked_preview);
}

#[test]
fn toml_hook_config_preview_shows_the_masked_token() {
    check_family(TOML_AGENTS, assert_masked_preview);
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
fn cline_preview_keeps_its_redaction() {
    check_family(&["cline"], |what, stdout| {
        assert!(stdout.contains(CLINE_REDACTED), "{what}:\n{stdout}");
    });
}

#[test]
fn claude_code_project_scope_preview_holds_no_placeholder_either() {
    let sandbox = Sandbox::new();
    let mut cmd = sandbox.install_hooks("claude-code");
    cmd.args(["--auth-token", TOKEN, "--scope", "project"]);
    let stdout = clean_preview("claude-code --scope project", cmd);
    assert!(stdout.contains("NOT embedded below"), "{stdout}");
    assert!(!stdout.contains("[REDACTED]"), "{stdout}");
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
            stdout.contains(MASKED_TOKEN) || stdout.contains("omitted from this"),
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
        // A file bearer never renders into a printed snippet, so neither the
        // live token nor a masked form may appear; the `--apply` control
        // below proves the file's value is still consumed.
        assert!(!stdout.contains("[REDACTED]"), "{what}:\n{stdout}");
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
        assert!(!stdout.contains("[REDACTED]"), "{agent}:\n{stdout}");
    }
}

/// A syntax error in `config.toml` used to fail with the parser's snippet
/// on stderr, echoing the offending line — which may hold
/// `[auth].bearer_token`. Both subcommands must exit 1 with the line and
/// column and none of the source line. A wrong-typed value must exit 1 with
/// the key and the expected type and none of the value — figment's own
/// display prints it (``found signed int `123456789012` ``), and a numeric
/// bearer saved without quotes is the confirmed leak.
#[test]
fn config_syntax_error_exits_one_without_echoing_the_line() {
    let installs: [(&str, &[&str]); 2] = [
        (
            "install-hooks",
            &[
                "install-hooks",
                "--agent",
                "opencode2",
                "--server-url",
                SERVER_URL,
            ],
        ),
        ("install-mcp", &["install-mcp", "--server-url", SERVER_URL]),
    ];
    let bodies = [
        (
            "trailing text after the closing quote",
            format!("[auth]\nbearer_token = \"{TOKEN}\" trailing\n"),
        ),
        (
            "missing closing quote",
            format!("[auth]\nbearer_token = \"{TOKEN}\n"),
        ),
    ];
    for (what, body) in bodies {
        let sandbox = Sandbox::new();
        sandbox.write_config(&body);
        for (subcommand, args) in installs {
            let mut cmd = sandbox.command();
            cmd.args(args);
            let output = cmd.output().unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert_eq!(
                output.status.code(),
                Some(1),
                "{what} + {subcommand}: expected exit 1:\n{stdout}\n{stderr}"
            );
            assert!(
                !stdout.contains(TOKEN_BODY) && !stderr.contains(TOKEN_BODY),
                "{what} + {subcommand} printed the token:\n{stdout}\n{stderr}"
            );
            assert!(
                !stderr.contains('|'),
                "{what} + {subcommand} printed the parser's margin:\n{stderr}"
            );
            assert!(
                stderr.contains("line 2"),
                "{what} + {subcommand} lost the line position:\n{stderr}"
            );
        }
    }

    // A wrong-typed value: the diagnostic keeps the key and the expected
    // type without the value itself.
    let type_bodies = [
        (
            "integer bearer",
            "[auth]\nbearer_token = 123456789012\n",
            "123456789012",
        ),
        (
            "sequence bearer",
            "[auth]\nbearer_token = [\"FAKEfake0123456789\"]\n",
            "FAKEfake",
        ),
    ];
    for (what, body, secret) in type_bodies {
        let sandbox = Sandbox::new();
        sandbox.write_config(body);
        for (subcommand, args) in installs {
            let mut cmd = sandbox.command();
            cmd.args(args);
            let output = cmd.output().unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert_eq!(
                output.status.code(),
                Some(1),
                "{what} + {subcommand}: expected exit 1:\n{stdout}\n{stderr}"
            );
            assert!(
                !stdout.contains(secret) && !stderr.contains(secret),
                "{what} + {subcommand} printed the value:\n{stdout}\n{stderr}"
            );
            assert!(
                stderr.contains("bearer_token") && stderr.contains("a string"),
                "{what} + {subcommand} lost the key or the expected type:\n{stderr}"
            );
            assert!(
                !stderr.contains('|'),
                "{what} + {subcommand} printed a parser margin:\n{stderr}"
            );
        }
    }

    // A numeric-looking bearer in the environment is parsed as a number by
    // the env layer and fails the type check; the rewritten diagnostic must
    // keep the value out there too.
    let sandbox = Sandbox::new();
    sandbox.write_config("");
    let mut cmd = sandbox.command();
    cmd.env("AI_MEMORY_AUTH__BEARER_TOKEN", "123456789012");
    cmd.args([
        "install-hooks",
        "--agent",
        "opencode2",
        "--server-url",
        SERVER_URL,
    ]);
    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(1),
        "a numeric env bearer must fail the type check:\n{stdout}\n{stderr}"
    );
    assert!(
        !stdout.contains("123456789012") && !stderr.contains("123456789012"),
        "the numeric env bearer must stay out of the output:\n{stdout}\n{stderr}"
    );
    assert!(
        stderr.contains("bearer_token") && stderr.contains("expected a string"),
        "the env type error keeps the key and the expected type:\n{stderr}"
    );

    // Control: a non-numeric env bearer is a valid string value, so the
    // load succeeds and the preview keeps withholding it.
    let sandbox = Sandbox::new();
    sandbox.write_config("");
    let mut cmd = sandbox.command();
    cmd.env("AI_MEMORY_AUTH__BEARER_TOKEN", "FAKEfakeEnvBearer12345");
    cmd.args([
        "install-hooks",
        "--agent",
        "opencode2",
        "--server-url",
        SERVER_URL,
    ]);
    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(0),
        "a non-numeric env bearer is a valid value:\n{stdout}\n{stderr}"
    );
    assert!(
        stdout.contains("FAKE...[REDACTED]"),
        "the preview must show the env bearer masked: {stdout}"
    );
    assert!(
        !stdout.contains("FAKEfakeEnvBearer12345") && !stderr.contains("FAKEfakeEnvBearer12345"),
        "the env bearer must stay out of the output:\n{stdout}\n{stderr}"
    );

    // Control: a valid config still loads and previews.
    let sandbox = Sandbox::new();
    sandbox.write_config(&format!("[auth]\nbearer_token = \"{TOKEN}\"\n"));
    let stdout = clean_preview("valid config control", sandbox.install_hooks("opencode2"));
    assert!(stdout.contains(TS_MASKED_TOKEN), "{stdout}");
}

/// Control for the token-file preview check above: with the token only in
/// `[auth].bearer_token_file`, `--apply` must still consume the file and
/// store its value, without printing it. A preview that merely never shows
/// the token would also pass `bearer_token_file_stays_out_of_previews` if
/// the loader ignored the file; this one cannot.
#[test]
fn bearer_token_file_apply_stores_the_file_value_without_printing_it() {
    let sandbox = Sandbox::new();
    let token_file = sandbox.home.join("root-token");
    std::fs::write(&token_file, TOKEN).unwrap();
    sandbox.write_config(&format!(
        "[auth]\nbearer_token_file = {:?}\n",
        token_file.display().to_string()
    ));
    let plugin = sandbox.home.join("ai-memory-opencode2.ts");
    let mut cmd = sandbox.install_hooks("opencode2");
    cmd.args(["--apply", "--config-file"]).arg(&plugin);
    let stdout = clean_preview("opencode2 --apply with [auth].bearer_token_file", cmd);
    assert!(!stdout.contains("[REDACTED]"), "{stdout}");

    let stored = std::fs::read_to_string(sandbox.data_dir.join("auth-token")).unwrap();
    assert_eq!(stored.trim(), TOKEN);
    let written = std::fs::read_to_string(&plugin).unwrap();
    assert!(written.contains(TS_NULL_TOKEN), "{written}");
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
    assert!(!stdout.contains("[REDACTED]"), "{stdout}");

    let stored = std::fs::read_to_string(sandbox.data_dir.join("auth-token")).unwrap();
    assert_eq!(stored.trim(), TOKEN);
    let written = std::fs::read_to_string(&plugin).unwrap();
    assert!(written.contains(TS_NULL_TOKEN), "{written}");
}

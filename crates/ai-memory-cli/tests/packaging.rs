//! Packaging asset regression tests.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate should live under crates/ai-memory-cli")
        .to_path_buf()
}

fn read_repo(path: &str) -> String {
    let path = repo_root().join(path);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn run_wrapper_on_fake_macos(args: &[&str]) -> String {
    let tmp = tempfile::tempdir().unwrap();
    let docker_args = tmp.path().join("docker-args.txt");
    let docker = tmp.path().join("docker");
    let uname = tmp.path().join("uname");
    std::fs::write(
        &docker,
        format!(
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > {}\n",
            docker_args.display()
        ),
    )
    .unwrap();
    std::fs::write(&uname, "#!/usr/bin/env bash\nprintf 'Darwin\\n'\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&docker, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::set_permissions(&uname, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    let path = format!(
        "{}:{}",
        tmp.path().display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let output = Command::new(repo_root().join("bin/ai-memory"))
        .args(args)
        .env("PATH", path)
        .env("AI_MEMORY_DOCKER", &docker)
        .env("AI_MEMORY_NO_VERSION_CHECK", "1")
        .env("AI_MEMORY_DATA_VOLUME", "test-ai-memory-data")
        .env("HOME", tmp.path())
        .env_remove("AI_MEMORY_SERVER_URL")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "wrapper failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(docker_args).unwrap()
}

#[test]
fn systemd_units_use_explicit_native_paths() {
    let system = read_repo("packaging/systemd/ai-memory.service");
    assert!(system.contains("--data-dir /var/lib/ai-memory"));
    assert!(system.contains("--config /etc/ai-memory/config.toml"));
    assert!(system.contains("EnvironmentFile=-/etc/ai-memory/env"));
    assert!(system.contains("StateDirectory=ai-memory"));
    assert!(system.contains("ReadWritePaths=/var/lib/ai-memory"));
    assert!(!system.contains("/var/local"));

    let user = read_repo("packaging/systemd/ai-memory-user.service");
    assert!(user.contains("--data-dir %h/.local/share/ai-memory"));
    assert!(user.contains("--config %h/.config/ai-memory/config.toml"));
    assert!(user.contains("EnvironmentFile=-%h/.config/ai-memory/env"));
    assert!(!user.contains("/var/lib/ai-memory"));
}

#[test]
fn aur_packages_install_all_native_assets() {
    for path in ["packaging/aur/PKGBUILD", "packaging/aur/PKGBUILD-bin"] {
        let pkgbuild = read_repo(path);
        assert!(pkgbuild.contains("/usr/bin/ai-memory"), "{path}");
        assert!(pkgbuild.contains("/usr/share/ai-memory"), "{path}");
        assert!(
            pkgbuild.contains("/usr/lib/systemd/system/ai-memory.service"),
            "{path}"
        );
        assert!(
            pkgbuild.contains("/usr/lib/systemd/user/ai-memory.service"),
            "{path}"
        );
        assert!(
            pkgbuild.contains("/usr/lib/sysusers.d/ai-memory.conf"),
            "{path}"
        );
        assert!(
            pkgbuild.contains("/usr/lib/tmpfiles.d/ai-memory.conf"),
            "{path}"
        );
        assert!(pkgbuild.contains("etc/ai-memory/config.toml"), "{path}");
        assert!(pkgbuild.contains("etc/ai-memory/env"), "{path}");
        assert!(
            pkgbuild.contains("install -Dm0640 packaging/env/ai-memory.env"),
            "{path}"
        );
    }

    let install = read_repo("packaging/aur/ai-memory.install");
    assert!(install.contains("sudo -u ai-memory ai-memory --data-dir /var/lib/ai-memory"));
    assert!(!install.contains("sudo ai-memory --data-dir /var/lib/ai-memory"));

    let bin_pkgbuild = read_repo("packaging/aur/PKGBUILD-bin");
    assert!(bin_pkgbuild.contains("source_x86_64"));
    assert!(bin_pkgbuild.contains("source_aarch64"));
    assert!(bin_pkgbuild.contains("linux-x86_64.tar.gz"));
    assert!(bin_pkgbuild.contains("linux-aarch64.tar.gz"));
}

#[test]
fn docker_source_build_uses_vendored_tailwind() {
    let dockerfile = read_repo("docker/Dockerfile");
    assert!(dockerfile.contains("TAILWIND_SKIP=1 cargo build --release -p ai-memory-cli"));
}

#[test]
fn docker_publish_jobs_use_prebuilt_binaries() {
    let dockerfile = read_repo("docker/Dockerfile");
    assert!(dockerfile.contains("FROM runtime-base AS runtime-prebuilt-amd64"));
    assert!(dockerfile.contains("FROM runtime-base AS runtime-prebuilt-arm64"));
    assert!(dockerfile.contains("dist/docker/ai-memory-linux-x86_64/ai-memory"));
    assert!(dockerfile.contains("dist/docker/ai-memory-linux-aarch64/ai-memory"));

    let release = read_repo(".github/workflows/release.yml");
    assert!(release.contains("artifact: ai-memory-linux-x86_64"));
    assert!(release.contains("artifact: ai-memory-linux-aarch64"));
    assert!(release.contains("artifact: ai-memory-macos-aarch64"));
    assert!(release.contains("artifact: ai-memory-macos-x86_64"));
    assert!(release.contains("needs: [binary, macos, windows, validate-version]"));
    assert!(release.contains("target: runtime-prebuilt-amd64"));
    assert!(release.contains("target: runtime-prebuilt-arm64"));

    let ci = read_repo(".github/workflows/ci.yml");
    assert!(ci.contains("ci-ai-memory-${{ matrix.artifact }}"));
    assert!(ci.contains("artifact: linux-x86_64"));
    assert!(ci.contains("artifact: macos-aarch64"));
    assert!(ci.contains("artifact: macos-x86_64"));
    assert!(ci.contains("runner: macos-15"));
    assert!(ci.contains("runner: macos-15-intel"));
    assert!(ci.contains("--target runtime-prebuilt-amd64"));
}

#[test]
fn macos_wrapper_routes_urls_by_real_subcommand() {
    for subcommand in ["install-mcp", "install-hooks", "setup-agent"] {
        let args = run_wrapper_on_fake_macos(&[subcommand]);
        assert!(
            !args.contains("AI_MEMORY_SERVER_URL=http://host.docker.internal:49374"),
            "{subcommand} renders host-side config and must keep loopback defaults; got {args}"
        );
    }

    let args = run_wrapper_on_fake_macos(&["status"]);
    assert!(
        args.contains("AI_MEMORY_SERVER_URL=http://host.docker.internal:49374"),
        "thin-client commands must reach the host server through Docker Desktop; got {args}"
    );

    let args = run_wrapper_on_fake_macos(&["search", "install-hooks"]);
    assert!(
        args.contains("AI_MEMORY_SERVER_URL=http://host.docker.internal:49374"),
        "only the actual subcommand should control URL routing; got {args}"
    );

    let args = run_wrapper_on_fake_macos(&["--config", "/tmp/config.toml", "install-hooks"]);
    assert!(
        !args.contains("AI_MEMORY_SERVER_URL=http://host.docker.internal:49374"),
        "global options before install-hooks must not hide the real subcommand; got {args}"
    );
}

#[test]
fn macos_docs_use_valid_install_commands_and_release_body_points_to_them() {
    let docs = read_repo("docs/macos.md");
    assert!(docs.contains("install-hooks --agent claude-code --apply"));
    assert!(docs.contains("install-mcp --client claude-code --apply"));
    assert!(
        !docs.contains("setup-agent --agent claude-code --source ./hooks"),
        "setup-agent has no --apply path; use install-hooks for native macOS docs"
    );
    assert!(
        !docs.contains("init` configures the bearer token"),
        "init writes token_pepper, not a bearer token"
    );
    assert!(docs.contains("Host-side agent config should use"));
    assert!(docs.contains("Tagged releases publish a multi-arch manifest"));

    let release = read_repo(".github/workflows/release.yml");
    assert!(release.contains("follow the bundled docs/macos.md"));
}

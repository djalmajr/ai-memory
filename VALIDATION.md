# Validation

## Scope

This contract covers Rust dependency changes in the root workspace and the
relay/importer companions. It supplements the gates in `AGENTS.md` and the
separate review. Extend it when a change touches another surface.

## Preconditions

- Use the toolchain pinned in `rust-toolchain.toml`; record `cargo --version`.
- Record `git rev-parse HEAD`, the changed lockfile hashes and any fixture hashes.
- Use synthetic data and isolated temporary stores. Do not contact a live server
  from tests; unset server/auth environment overrides for local Cargo commands.
- Use the audit/deny versions and existing advisory exceptions from
  `.github/workflows/ci.yml`; do not add an exception to make this fix pass.
- A missing tool, permission or runner makes its affected check BLOCKED.

## Test layers

- unit: `cargo test -p ai-memory-store --lib migrations`
- integration: `cargo test --locked --manifest-path companions/ai-memory-relay/Cargo.toml`
- contract: `cargo test -p ai-memory-cli --lib mcp_bridge`
- e2e: manual: the shipped relay binary tests and the legacy-store open/backup/reopen procedure in item 2; the complete CLI suite runs in CI.
- mutation: manual: the named guard-removal procedures in item 3, in temporary source copies.

## Items

1. **Required merge gates.** On the final PR SHA, require a separate review and
   the applicable Linux checks in `ci.yml`, including formatting, Clippy,
   workspace tests/docs, companion suites, dependency policy, audit, secret
   scanning and changelog checks. Run the affected local suites during iteration;
   reuse earlier receipts only while their relevant source, configuration,
   dependencies, fixtures and environment remain equivalent. The complete CI
   suite supplies the final integrated receipt; do not repeat it locally solely
   to duplicate that evidence. The importer is standalone:
   `cargo test --locked --manifest-path companions/ai-memory-importer/Cargo.toml`.
2. **Required consumer checks.** Preserve these permanent regressions:
   - Store `a_store_written_by_the_previous_driver_opens_migrates_reopens_and_backs_up`:
     the SHA-pinned V101 fixture opens under the new driver, retains its FTS
     result, validates migrations, reopens and survives online backup and
     `open_snapshot`. `version_boundaries_are_refused_without_saturating_or_migrating`
     rejects invalid numeric targets before creating migration history and
     retains a valid migration control.
   - Bridge: both oversized response tests reject through the production
     transport. Success uses `ResponseBodyTooLarge`; an oversized HTTP error
     discards its body and falls back to status. The correlated small JSON-RPC
     error and successful handshake remain legitimate controls.
   - Relay: classification and busy-retry tests preserve typed SQLite errors,
     extended codes and foreign bytes. The shipped-binary rendering test proves
     the SQLite cause appears once and non-SQLite errors keep their context.
   - Audit all five lockfiles as CI does; run `cargo deny --all-features check`
     at the root and with the relay manifest. Check each affected dependency
     graph with `cargo tree --offline --locked -e features -i libsqlite3-sys`
     and its companion manifest equivalent. Keep one bundled SQLite node per
     consumer, the intended fixed versions and the existing features.
3. **Required when the corresponding guard changes.** Execute each mutation in
   a temporary copy; retain the baseline pass and mutated runtime failure:
   - Restore target saturation and negative-version saturation separately:
     the version-boundary test fails, while a valid target still migrates.
   - Reintroduce relay queue error flattening: the two typed-error regressions
     fail. Separately restore raw error-chain rendering in `main::finish`:
     the shipped-binary presentation test fails.
   - Remove only the two `read_bounded_body` size checks in a temporary SDK
     copy: both oversized bridge tests fail at runtime and both small controls
     pass. Do not edit the installed registry or the shipped manifests/locks.
   A compilation failure is not a successful mutation proof.
4. **Platform checks.** Add the `windows` PR label for path, locking, git or
   hook changes and wait for the affected native jobs. The Linux Windows
   cross-build proves compilation only. Before a release, separately dispatch
   `ci` and `windows` on the exact candidate SHA and wait for the full matrix,
   as required by `AGENTS.md`.

## Evidence

Keep raw stdout/stderr, argv, cwd, exit status and relevant input hashes in CI
artifacts or the delivery record linked from the issue. Distinguish locally
executed checks, supplied receipts and reused evidence. Keep private logs,
temporary mutation sources and local paths out of public commits. Record any
redactions and environment restrictions without exposing credentials.

## Results

Record PASS, FAIL, NOT RUN or BLOCKED for each applicable item on each delivery.
No matching tests is NOT RUN. A required item without a proven PASS prevents an
overall PASS; record a pre-existing failure separately with its issue and
evidence. This file is a contract, not a permanent green receipt.

## What this does not prove

The legacy fixture proves compatibility, not exploitation of the SQLite
advisory. Audit covers its database at execution time. These checks do not
prove production behavior, unrelated UI behavior, native execution from a
cross-build, or a release matrix that has not run on the candidate SHA.

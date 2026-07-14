## What changed

<!-- One paragraph or bullet list: the observable behaviour before vs. after. -->

## Why

<!-- The motivation: bug fix, new feature, performance, correctness. Link related issue if any. -->

## Test plan

- [ ] `cargo fmt --all -- --check` passes
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes
- [ ] `cargo test --workspace` passes
- [ ] Manual test: <!-- describe what you ran and what you observed -->

## CHANGELOG (merge gate)

- [ ] I added a `CHANGELOG.md` `[Unreleased]` entry — **required** for any
      user-facing change: new flag / env var / endpoint / MCP tool / marker
      key, changed behaviour, or an observable bug fix. (Exempt only for
      internal refactors, dead-code removal, and test-only churn.)
- [ ] Any changed **default** (flag behaviour, config default, env var,
      response shape) is called out explicitly in "What changed" above.

Reviewers treat a missing entry as blocking — adding it up front is what
keeps your PR merging on the first pass.

## Notes for reviewers

<!-- Anything tricky, a design decision you made, or areas you'd like extra scrutiny on. -->

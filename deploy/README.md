# deploy/ — fork-exclusive image build

Builds the fork's production ai-memory image (engine + UI in ONE image)
into the fork GHCR, `ghcr.io/djalmajr/ai-memory`.

- `Dockerfile` — the production image recipe. The engine comes from the
  build context (a checkout of a validated full SHA, never a branch); the
  UI is cloned at a pinned full SHA. Same stages, user, ports and volumes
  as the image in production; no cache mounts.
- `.github/workflows/fork-image.yml` — the only build path: manual
  `workflow_dispatch` with two required 40-hex-SHA inputs.

## Building a candidate

From this repo:

```sh
gh workflow run fork-image.yml -f engine_ref=<40-hex engine SHA> -f ui_ref=<40-hex ui SHA>
```

Both refs are validated (exactly 40 hex chars) before any checkout, login
or build. The run pushes the immutable tag
`ghcr.io/djalmajr/ai-memory:candidate-<run_id>-<run_attempt>` (never
`latest`), validates the pushed digest (`sha256:` + 64 hex) and writes
image, tag, digest, both refs and the workflow SHA to the run summary.
Pin the image by digest (`image@digest`) in the deploy manifests.

## Upstream sync

Everything under `deploy/` and the `fork-image.yml` workflow is fork-only:
upstream (`akitaonrails/ai-memory`) has neither `deploy/` nor that
workflow (verified against upstream main), so merging upstream `main`
never conflicts with these paths. Sync with a plain merge of
`akitaonrails/ai-memory` `main` into this repo's `main`. If the engine
layout changes (workspace members, hooks directory), update the `COPY`
list in `deploy/Dockerfile` accordingly.

# Bundle: ai-memory (engine, Rust) + custom SPA (ai-memory-ui) served at the
# host root.
#
# Fork deploy image: ONE image with the engine and the UI. The engine comes
# from the BUILD CONTEXT (the checkout of the validated engine_ref), not as a
# git clone; the UI is still cloned at a pinned full SHA. Built only by
# .github/workflows/fork-image.yml (context: engine/, dockerfile:
# deploy-src/deploy/Dockerfile read from the workflow commit), so an old
# engine SHA that predates deploy/ still builds with the current Dockerfile.
#
# No `--mount=type=cache`: the image stays cache-free, like the one that is
# in production today. Single-arch: linux/amd64 (see the workflow).
#
# The same image serves all instances (personal/client) — only the push
# registry and the branding build-args (VITE_APP_NAME/TAGLINE) change.
#
# Example (what the workflow runs):
#   docker build --file deploy/Dockerfile --context <engine checkout> \
#     --build-arg AI_MEMORY_UI_REF=<ui: full 40-hex SHA> \
#     --build-arg VITE_APP_NAME=Memory --build-arg VITE_APP_TAGLINE="knowledge browser" \
#     -t ghcr.io/djalmajr/ai-memory:<candidate-tag>

ARG AI_MEMORY_UI_REPO=https://github.com/djalmajr/ai-memory-ui.git
# No default on purpose: the ref must be passed explicitly and is validated
# upstream (workflow fork-image) as exactly 40 hex chars BEFORE Buildx/login/
# build. It never defaults to main/latest — an empty ref fails the gate
# below instead of silently building the UI fork's default branch.
ARG AI_MEMORY_UI_REF
ARG AI_MEMORY_UI_REPO_API=https://api.github.com/repos/djalmajr/ai-memory-ui

# ---------------------------------------------------------------------------
# 1) SPA (SolidJS) → /ui/dist
# ---------------------------------------------------------------------------
FROM node:22-slim AS ui
ARG AI_MEMORY_UI_REPO
ARG AI_MEMORY_UI_REF
ARG AI_MEMORY_UI_REPO_API
ARG VITE_APP_NAME="Memory"
ARG VITE_APP_TAGLINE="knowledge browser"
# Reject an invalid ref before any network or install work. The workflow
# already validates engine_ref/ui_ref before Buildx/login/build; this is
# defense in depth so the stage can never run on a branch/tag/latest ref.
RUN echo "${AI_MEMORY_UI_REF}" | grep -Eq '^[0-9a-f]{40}$' \
    || { echo "AI_MEMORY_UI_REF must be exactly 40 hex chars (full SHA); branch names, release tags, latest and empty are not allowed" >&2; exit 1; }
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /ui
# Cache-bust from the exact immutable ref, then fetch it directly. `git fetch`
# accepts release tags and reachable 40-character SHAs; `git clone --branch`
# rejected SHAs even though the workflow explicitly permits them.
ADD ${AI_MEMORY_UI_REPO_API}/commits/${AI_MEMORY_UI_REF} /tmp/ui-tip.json
RUN git init \
 && git remote add origin "${AI_MEMORY_UI_REPO}" \
 && git fetch --depth 1 origin "${AI_MEMORY_UI_REF}" \
 && git checkout --detach FETCH_HEAD
RUN npm ci
ENV VITE_APP_NAME=${VITE_APP_NAME}
ENV VITE_APP_TAGLINE=${VITE_APP_TAGLINE}
RUN npm run build   # i18n + tsr generate + tsc + vite build → /ui/dist

# ---------------------------------------------------------------------------
# 2) ai-memory (Rust) → /usr/local/bin/ai-memory
# ---------------------------------------------------------------------------
FROM rust:1.95-slim-bookworm AS builder
RUN apt-get update \
 && apt-get install -y --no-install-recommends pkg-config build-essential curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /work
# The engine is the build context: a checkout of the validated engine_ref.
# Copy only what the workspace build and the runtime image need:
#   - the workspace manifests: cargo resolves the whole workspace before
#     building any member, including the test-only `evals` member;
#   - the crates, with their vendored assets (ai-memory-web/static/...);
#   - hooks/, installed into the runtime image below;
#   - docs/logo.png, embedded by ai-memory-web (routes/statics.rs
#     include_bytes!) in every release build.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY evals/ evals/
COPY hooks/ hooks/
COPY docs/logo.png docs/logo.png
RUN cargo build --release -p ai-memory-cli \
 && cp target/release/ai-memory /usr/local/bin/ai-memory

# ---------------------------------------------------------------------------
# 3) runtime — debian-slim, non-root, SPA em /web-ui
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 1000 --user-group --home-dir /data \
        --shell /usr/sbin/nologin ai-memory \
 && mkdir -p /data /web-ui \
 && chown -R ai-memory:ai-memory /data
COPY --from=builder /usr/local/bin/ai-memory /usr/local/bin/ai-memory
# Lifecycle hooks (setup-agent/install-hooks find them here).
COPY --from=builder /work/hooks /usr/local/share/ai-memory/hooks
# Custom SPA — served via `--enable-web --web-ui-dir /web-ui` (the default
# CMD below; instance args may override the workspace).
COPY --from=ui /ui/dist /web-ui
USER ai-memory
ENV AI_MEMORY_DATA_DIR=/data \
    AI_MEMORY_WEB_SLUG=/
WORKDIR /data
EXPOSE 49374
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["/usr/local/bin/ai-memory", "status"]
ENTRYPOINT ["/usr/local/bin/ai-memory"]
# Default standalone; instance args may override (includes --workspace).
CMD ["serve", "--transport", "http", "--bind", "0.0.0.0:49374", "--enable-web", "--web-ui-dir", "/web-ui"]

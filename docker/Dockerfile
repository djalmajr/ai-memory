# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the ai-memory MCP server.
# Builder: rust:1.95-slim with the release binary cached against
#          BuildKit's cargo registry + target caches.
# Runtime: debian:bookworm-slim with a non-root `ai-memory` user that
#          owns /data. CA certificates included for outgoing HTTPS (LLM
#          providers in M6+).

FROM rust:1.95-slim-bookworm AS builder
WORKDIR /work
RUN apt-get update \
 && apt-get install -y --no-install-recommends pkg-config build-essential \
 && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock rust-toolchain.toml deny.toml ./
COPY crates ./crates
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/work/target \
    cargo build --release -p ai-memory-cli \
 && cp target/release/ai-memory /usr/local/bin/ai-memory

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 1000 --user-group --home-dir /data \
        --shell /usr/sbin/nologin ai-memory \
 && mkdir -p /data \
 && chown -R ai-memory:ai-memory /data
COPY --from=builder /usr/local/bin/ai-memory /usr/local/bin/ai-memory
# Lifecycle-hook scripts bundled into the image so docker-only users
# can `ai-memory setup-agent` without needing a repo checkout. The
# scripts live on a path scanned by `install_hooks::resolve_hooks_dir`
# (and by `setup_agent::resolve_source`).
COPY hooks /usr/local/share/ai-memory/hooks
USER ai-memory
ENV AI_MEMORY_DATA_DIR=/data
WORKDIR /data
EXPOSE 49374
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["/usr/local/bin/ai-memory", "status"]
ENTRYPOINT ["/usr/local/bin/ai-memory"]
CMD ["serve", "--transport", "http", "--bind", "0.0.0.0:49374"]

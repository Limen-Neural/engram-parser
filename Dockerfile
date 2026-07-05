# syntax=docker/dockerfile:1.4
#
# Dockerfile for engram-parser (pure-Rust, zero-dep GGUF/MoE parser).
# Multi-stage for small image and reproducible CI builds.
# 
# Build with user's Docker CLI (as per review): use your wrapper/alias for `docker`.
# Example verification:
#   docker build --target builder -t engram-parser-builder .
#   docker run --rm engram-parser-builder cargo test --all-features
#
# See .github/workflows/docker-build.yml and issue #9 for usage in CI (GHCR on main).
# Follows corinth-canal patterns adapted for CPU-only / no CUDA.
# Avoids A && B || C anti-pattern (Codacy warning) by using explicit grouping or separate RUNs.

ARG RUST_VERSION=1.85

# Builder stage: compile with all features for verification
FROM rust:${RUST_VERSION}-slim AS builder

WORKDIR /app

# System deps for Rust (minimal for this zero-dep crate)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for better layer caching
COPY Cargo.toml Cargo.lock ./

# Create dummy main to cache deps (since no [[bin]] by default, use the lib + test)
RUN mkdir src && echo "fn main() {}" > src/main.rs && \
    cargo build --release --all-features && \
    rm -rf src

# Now copy real source
COPY . .

# Build (this layer will be cached on source changes only)
# Note: we build the lib + run tests in CI job; here we just ensure it builds
RUN cargo build --release --all-features

# Runtime / verification stage (cargo available for local verification)
FROM rust:${RUST_VERSION}-slim AS runtime

RUN useradd -m -u 10001 appuser

WORKDIR /app

# Copy the built artifacts (for if we expose a binary later, e.g. gguf_smoke)
COPY --from=builder /app/target/release /app/target/release

# For library use, the image mainly serves as a reproducible build env.
# You can also cargo install or use as base for downstream.

USER appuser

# Default: show help if a binary is present; otherwise this is a build image
CMD ["cargo", "--version"]

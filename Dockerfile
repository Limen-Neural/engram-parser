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

ARG RUST_VERSION=1.87

# Builder stage: compile with all features for verification
FROM rust:${RUST_VERSION}-slim AS builder

WORKDIR /app

# Copy manifests and lock file for reproducibility
COPY Cargo.toml Cargo.lock ./

# Copy source
COPY . .

# Build the crate (zero external deps, no system packages needed)
RUN cargo build --release --all-features

# Runtime / verification stage (minimal)
FROM debian:stable-slim AS runtime

RUN useradd -m -u 10001 appuser

WORKDIR /app

# Copy only the compiled library artifact (not the entire target/release tree)
COPY --from=builder /app/target/release/libengram_parser.rlib /app/lib/

USER appuser

# Library crate with no binary target; default to a no-op informational message.
CMD ["echo", "engram-parser: use as base image or override CMD"]

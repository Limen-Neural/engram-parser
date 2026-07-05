# engram-parser

Pure-Rust, **zero-dependency** `.gguf` deserializer and
Mixture-of-Experts per-expert weight extractor.

## What it does

- Parses the GGUF file format (magic, version 3 header, KV metadata,
  tensor directory) into an in-memory [`GgufLayout`].
- Enumerates MoE experts discovered in the checkpoint.
- Rips out the raw byte buffers for any single expert's `gate`, `up`,
  and `down` projections — supporting both the stacked
  (`blk.{B}.ffn_{role}_exps.weight`) and per-expert
  (`blk.{B}.ffn_{role}.{E}.weight`) on-disk conventions.

## What it does NOT do

- No neural-network math. No `matmul`, no `forward`, no routing,
  no softmax, no dequantization in the default build. F16→F32 bit
  conversion is available as an optional helper only.
- No CUDA, no GPU, no SIMD.
- No runtime dependencies. `[dependencies]` is intentionally empty.

## Quick start

```rust
use engram_parser::{extract_expert, list_experts, load_gguf};

let layout = load_gguf("./model.gguf")?;
println!("architecture = {}", layout.metadata.architecture());

for (block, expert) in list_experts(&layout) {
    let weights = extract_expert(&layout, block, expert)?;
    if let Some(gate) = &weights.gate {
        println!("blk.{block}.expert{expert}.gate: dims={:?} dtype={:?} bytes={}",
            gate.dims, gate.dtype, gate.bytes.len());
    }
}
# Ok::<(), engram_parser::ParserError>(())
```

## Supported dtypes

Layout-aware parsing: `F32`, `F16`, `BF16` (GGML 30), `Q8_0`, `Q4_K`,
`Q5_K`, `Q6_K`, `IQ3_S` (opaque), plus a `DType::Other(u32)` catch-all.
Only `F32` and `F16` have in-crate numeric accessors; everything else
is returned as raw `Vec<u8>`.

## Public API

`load_gguf`, `parse_bytes`, `GgufLayout`, `GgufMetadata`, `Tensor`,
`DType`, `extract_expert`, `list_experts`, `MoeExpertWeights`,
`RawTensor`, `ParserError`, `Result`.

## Ecosystem / Sibling parsers (LIM-9)

- **engram-parser** (this crate): canonical zero-dep GGUF v3 deserializer + per-expert MoE raw weight ripper.
- Safetensors extraction (header inspection, deterministic manifest, MoE router/expert candidate discovery via classify + groups + layout families) from `rmems/corinth-canal` (experimental source of inspiration) is tracked as a **separate issue** in this repo: #10 (parallel to the GGUF work in #7).
  - Source-side bootstrap/supporting: rmems/corinth-canal#116.
  - Coordination for consumers (e.g. future multi-format in cortex): Limen-Neural/cortex-tensor#9.
  - The reusable implementation will target a dedicated Limen-Neural crate (per org boundary matrix LIM-9); engram-parser charter remains GGUF-only.
- **Clarification**: one-way extraction/copy of code from inspiration. We are not adding any dependency from corinth-canal. corinth-canal keeps an unmodified reference copy (per its PROMOTION_RULES "frozen" status). See #10, #7, and the plan for full cross-links and "no dep on corinth-canal" language.

Cross-links and updates performed when #10 was created.

## Development

This is a pure-Rust, zero-dependency crate. All commands use `--all-features`.

```bash
# Format
cargo fmt --check

# Lint (fail on warnings)
cargo clippy --all-targets --all-features -- -D warnings

# Build
cargo build --all-features

# Test
cargo test --all-features

# Coverage (local; uses cargo-llvm-cov)
cargo llvm-cov --lib --all-features --locked --lcov --output-path lcov.info
```

### Repository layout hygiene

- Keep the repo single-root and workspace-clean (no duplicated crate roots / nested copies of the repo).
- Do not commit generated artifacts; CI expects the working tree to remain clean after build/test.
- If you add tools that generate files (e.g. coverage reports), ensure outputs are either ignored or cleaned up.

## Docker

The CI workflow publishes an image to GHCR on pushes to `main`:

- Image: `ghcr.io/limen-neural/engram-parser`
- Tags: `main` and the commit SHA

```bash
# Pull the published image
docker pull ghcr.io/limen-neural/engram-parser:main

# Run a verification command in the container
docker run --rm ghcr.io/limen-neural/engram-parser:main cargo test --all-features

# Build locally
docker build -t engram-parser:local .

# Run locally-built image
docker run --rm engram-parser:local cargo --version
```

## CI

- GitHub Actions: `.github/workflows/ci.yml` (harden in progress via #11; uses Codecov per https://about.codecov.io/language/rust/ )
- Azure Pipelines: `azure-pipelines.yml` (tracked in #8 for cross-platform ubuntu/mac/windows)
- Docker: `Dockerfile` + `.github/workflows/docker-build.yml` (tracked in #9 for GHCR reproducible builds; use user's Docker CLI for local verification)
- Other CI/DX issues: #12 (security), #13 (releases on tags w/ sentry option), #14 (MSRV), #15 (Dependabot no auto-merge), #16 (layout clean)

See the issue bodies for full ACs and corinth-canal inspiration patterns (one-way copy only; no dep on corinth-canal).

Cross-reference: #11, #8, #9, #7, #5, LIM-9.

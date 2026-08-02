# engram-parser

[![CI](https://github.com/Limen-Neural/engram-parser/actions/workflows/ci.yml/badge.svg)](https://github.com/Limen-Neural/engram-parser/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

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

## Scope / Boundaries

This crate **owns**:

- GGUF v3 deserialization (header, KV metadata, tensor directory).
- MoE expert enumeration (`list_experts`).
- Per-expert raw weight extraction (`extract_expert` — gate/up/down byte
  buffers with shape and dtype metadata).
- Zero-dependency, layout-aware dtype handling (F32/F16/BF16 plus opaque
  quant types as raw bytes).

This crate **does not own**:

- Neural-network math (matmul, forward, routing, softmax, dequantization
  in the default build).
- CUDA/GPU/SIMD execution.
- Tokenization, inference orchestration, or SNN dynamics.
- Full checkpoint routing or model-family adapters (see
  [`cortex-tensor`](https://github.com/Limen-Neural/cortex-tensor)).

**Allowed dependencies:** none — `[dependencies]` stays empty.

**Forbidden dependencies:** inference engines, GPU backends, domain-specific
adapters.

| Crate | Role |
|-------|------|
| `engram-parser` | GGUF parse + per-expert weight extraction |
| [`cortex-tensor`](https://github.com/Limen-Neural/cortex-tensor) | Tensor math + MoE routing on extracted weights |
| [`hybrid-fusion`](https://github.com/Limen-Neural/hybrid-fusion) | ANN→SNN orchestration |
| [`neuromod`](https://github.com/Limen-Neural/neuromod) | SNN neuron dynamics (downstream consumer) |

See [LIM-9](https://linear.app/saaq-spiking-adaptive-activity/issue/LIM-9/plan-rust-runtime-and-deployment-repo-boundary-matrix)
for the full Rust runtime/deployment boundary matrix and
[issue #4](https://github.com/Limen-Neural/engram-parser/issues/4) for
this repo's tracking issue.


## Origin / modularization (#7)

GGUF layout parsing and MoE expert **raw byte** extraction were expanded
using one-way inspiration from the experimental
[`rmems/corinth-canal`](https://github.com/rmems/corinth-canal) reference
implementation (**no** runtime dependency on corinth-canal).

- Tracking: [engram-parser#7](https://github.com/Limen-Neural/engram-parser/issues/7)
- Corinth migration companion: [corinth-canal#115](https://github.com/rmems/corinth-canal/issues/115)
- Cortex coordination: [cortex-tensor#8](https://github.com/Limen-Neural/cortex-tensor/issues/8)
- Linear: [LIM-123](https://linear.app/rpd-34/issue/LIM-123), [LIM-88](https://linear.app/rpd-34/issue/LIM-88)

Wire-type labels follow the corinth-canal `ggml` table (e.g. type **31** is
historical `Q4_0_4_4`, not the HuggingFace “IQ3_M” preset). MoE extraction
remains free functions (`list_experts` / `extract_expert`); traits are out
of scope for #7.

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

Layout-aware parsing (byte sizes only — **no dequant**) for GGUF wire types:
`F32`, `F16`, `BF16` (30), `F64`, `I8`–`I64`, `Q4_0`/`Q4_1`,
`Q5_0`/`Q5_1`, `Q8_0`/`Q8_1`, `Q2_K`–`Q8_K`, and IQ wire layouts
`IQ2_XXS`/`IQ2_XS`/`IQ2_S`, `IQ3_XXS`/`IQ3_S`, `IQ1_S`/`IQ1_M`,
`IQ4_NL`/`IQ4_XS`. Remaining codes use `DType::Other(u32)` (including
historical **wire type 31 = `Q4_0_4_4`**, which is **not** HF “IQ3_M”
and fails closed without a modeled size).

Only `F32` and `F16` have in-crate numeric accessors; everything else
is returned as raw `Vec<u8>`. Unknown layouts fail closed at parse time
when element count cannot be converted to a byte length.

`GgufMetadata::quantization()` prefers `general.quantization_type`, then
falls back to `general.file_type` (`0→F32`, `1→F16`, else `GGUF(n)`).

## Public API

`load_gguf`, `parse_bytes`, `GgufLayout`, `GgufMetadata`, `Tensor`,
`DType`, `ggml_type_label`, `extract_expert`, `list_experts`,
`MoeExpertWeights`, `RawTensor`, `ParserError`, `Result`, plus public
`GGML_TYPE_*` and `GGUF_VALUE_TYPE_*` constants.

## Ecosystem / Sibling parsers (LIM-9)

- **engram-parser** (this crate): canonical zero-dep GGUF v3 deserializer + per-expert MoE raw weight ripper.
- Safetensors extraction (header inspection, deterministic manifest, MoE router/expert candidate discovery via classify + groups + layout families) from `rmems/corinth-canal` (experimental source of inspiration) is tracked as a **separate issue** in this repo: #10 (parallel to the GGUF work in #7).
  - Source-side bootstrap/supporting: rmems/corinth-canal#116.
  - Coordination for consumers (e.g. future multi-format in cortex): Limen-Neural/cortex-tensor#9.
  - The reusable implementation will target a dedicated Limen-Neural crate (per org boundary matrix LIM-9); engram-parser charter remains GGUF-only.
- **Clarification**: one-way extraction/copy of code from inspiration. We are not adding any dependency from corinth-canal. corinth-canal keeps an unmodified reference copy (per its PROMOTION_RULES "frozen" status). See #10, #7, and the plan for full cross-links and "no dep on corinth-canal" language.

Cross-links and updates performed when #10 was created.

## Development

This is a pure-Rust, zero-dependency crate. Build, lint, and test commands use `--all-features`.

```bash
# Format
cargo fmt --check

# Lint (fail on warnings)
cargo clippy --all-targets --all-features -- -D warnings

# Build
cargo build --all-features

# Test
cargo test --all-features

# Coverage (local; requires cargo-llvm-cov: cargo install cargo-llvm-cov)
cargo llvm-cov --all-targets --all-features --locked --lcov --output-path lcov.info

# Real GGUF pilots (xai-dissect style; not CI — needs weights on disk)
# Full-file load (no mmap): one ENGRAM_GGUF per process; free RAM ≥ file size + margin
ENGRAM_GGUF=~/.models/gguf/.../model.gguf cargo test --test real_gguf -- --ignored --nocapture
# Large MoE: ENGRAM_EXPECT_MOE=1 ENGRAM_MOE_SAMPLES=3 (see REVIEW.md T1 large MoE)
cargo run --example inspect_gguf -- ~/.models/gguf/.../model.gguf
```

GPU experiments on real models live in **`~/rmems/blackwell-kernel-lab`**
(and production kernels in `myelin-accelerator`) — not as deps of this crate.
See [REVIEW.md](REVIEW.md) for the T0/T1/T2 quality-gate layout.

## Docker

```bash
# Build the image locally (includes build + test verification)
docker build -t engram-parser .

# Run tests in the container
docker run --rm engram-parser

# Pull from GHCR (published on merges to main)
docker pull ghcr.io/limen-neural/engram-parser:main
```

## CI

- GitHub Actions: `.github/workflows/ci.yml` (hardened via #11; uses Codecov per <https://about.codecov.io/language/rust/>)
- Security: `.github/workflows/security.yml` (RustSec audit always runs; Snyk SCA+SAST opt-in via `SNYK_TOKEN` secret, see #12)
- Azure Pipelines: `azure-pipelines.yml` (tracked in #8 for cross-platform ubuntu/mac/windows)
- Docker: `Dockerfile` + `.github/workflows/docker-build.yml` (tracked in #9 for GHCR reproducible builds; use user's Docker CLI for local verification)
- Other CI/DX issues: #13 (releases on tags w/ sentry option), #14 (MSRV), #15 (Dependabot no auto-merge), #16 (layout clean)

See the issue bodies for full ACs and corinth-canal inspiration patterns (one-way copy only; no dep on corinth-canal).

Cross-reference: #11, #8, #9, #7, #5, LIM-9.

## MSRV (Minimum Supported Rust Version)

**MSRV: 1.97** (current stable floor as of 2026-07)

This crate guarantees compatibility with Rust 1.97 and later. The MSRV is:

- Declared in `Cargo.toml` via `rust-version = "1.97"`
- Tested in CI on every PR and push (see `msrv` job in `.github/workflows/ci.yml`)
- Verified alongside **stable** (always latest) in the `validate` job so both toolchains pass

Local development defaults to the toolchain in [`rust-toolchain.toml`](rust-toolchain.toml)
(`stable` + `rustfmt` / `clippy`).

**MSRV Policy:**
- MSRV bumps will be documented in release notes
- Bumps are considered breaking changes and follow semver conventions
- Justification is required when bumping MSRV (e.g., dependency requirements, critical features)

See [issue #14](https://github.com/Limen-Neural/engram-parser/issues/14) for the full MSRV policy discussion.


## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))
- MIT license ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))

at your option.

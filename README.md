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

## Development

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))
- MIT license ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))

at your option.
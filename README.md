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
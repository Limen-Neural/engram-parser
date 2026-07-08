# engram-parser

[![CI](https://github.com/Limen-Neural/engram-parser/actions/workflows/ci.yml/badge.svg)](https://github.com/Limen-Neural/engram-parser/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Limen-Neural/engram-parser/branch/main/graph/badge.svg)](https://codecov.io/gh/Limen-Neural/engram-parser)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

A pure-Rust, zero-dependency parser for GGUF (GPT-Generated Unified Format) v3 files with Mixture of Experts (MoE) support.

## Features

- **Zero dependencies**: Pure Rust implementation, no external crates
- **Complete GGUF v3 parsing**: Headers, metadata, tensor directories
- **Full GGML type coverage**: 32 type constants (F32, F16, Q4_0-Q8_K, IQ1_S-IQ3_M, etc.)
- **Human-readable type labels**: `ggml_type_label()` function for all GGML types
- **Metadata helpers**: Architecture-aware convenience methods (block_count, expert_count, etc.)
- **MoE support**: Expert weight extraction from stacked and per-expert tensor formats
- **Byte-level accuracy**: Precise byte length calculations for all quantization types

## Quick Start

```rust
use engram_parser::{load_gguf, ggml_type_label};

let layout = load_gguf("model.gguf")?;

// Access metadata with helper methods
println!("Architecture: {}", layout.metadata.architecture());
println!("Quantization: {}", layout.metadata.quantization());
println!("Block count: {:?}", layout.metadata.block_count());
println!("Expert count: {:?}", layout.metadata.expert_count());

// List and extract MoE experts
for (block, expert) in engram_parser::list_experts(&layout) {
    let weights = engram_parser::extract_expert(&layout, block, expert)?;
    println!("Expert {block}.{expert}: gate={:?}, up={:?}, down={:?}",
        weights.gate.is_some(), weights.up.is_some(), weights.down.is_some());
}

// Use type labels for human-readable output
for (name, tensor) in &layout.tensors {
    println!("{}: type={}, dims={:?}",
        name, ggml_type_label(tensor.ggml_type), tensor.dims);
}
```

## Supported GGML Types

The parser supports all 32 GGML tensor type constants:

### Floating Point Types
- `GGML_TYPE_F32` (0): 32-bit float
- `GGML_TYPE_F16` (1): 16-bit float
- `GGML_TYPE_F64` (28): 64-bit float
- `GGML_TYPE_BF16` (30): Brain float 16

### Integer Types
- `GGML_TYPE_I8` (24): 8-bit integer
- `GGML_TYPE_I16` (25): 16-bit integer
- `GGML_TYPE_I32` (26): 32-bit integer
- `GGML_TYPE_I64` (27): 64-bit integer

### Quantized Types
- `GGML_TYPE_Q4_0` (2), `GGML_TYPE_Q4_1` (3): 4-bit quantization
- `GGML_TYPE_Q5_0` (6), `GGML_TYPE_Q5_1` (7): 5-bit quantization
- `GGML_TYPE_Q8_0` (8), `GGML_TYPE_Q8_1` (9): 8-bit quantization
- `GGML_TYPE_Q2_K` through `GGML_TYPE_Q8_K` (10-15): K-quant types
- `GGML_TYPE_IQ1_S` (19), `GGML_TYPE_IQ1_M` (29): 1-bit i-quant
- `GGML_TYPE_IQ2_XXS` (16), `GGML_TYPE_IQ2_XS` (17), `GGML_TYPE_IQ2_S` (22): 2-bit i-quant
- `GGML_TYPE_IQ3_XXS` (18), `GGML_TYPE_IQ3_S` (21), `GGML_TYPE_IQ3_M` (31): 3-bit i-quant
- `GGML_TYPE_IQ4_NL` (20), `GGML_TYPE_IQ4_XS` (23): 4-bit i-quant

All types have:
- Public constants for matching (e.g., `GGML_TYPE_IQ3_M`)
- Human-readable labels via `ggml_type_label()`
- Precise byte length calculations where applicable
- `DType` enum representation for type-safe code

## Metadata Helper Methods

The `GgufMetadata` struct provides architecture-aware convenience methods:

```rust
// Basic metadata
metadata.architecture()           // e.g., "qwen2moe"
metadata.quantization()           // e.g., "Q4_K_M"

// Model dimensions (architecture-aware)
metadata.block_count()            // {arch}.block_count
metadata.expert_count()           // {arch}.expert_count or num_experts
metadata.expert_used_count()      // {arch}.expert_used_count or num_experts_per_tok
metadata.embedding_length()       // {arch}.embedding_length
metadata.head_count()             // {arch}.attention.head_count

// Generic accessors
metadata.numeric("custom.key")    // Any numeric value
metadata.string("custom.key")     // Any string value
metadata.float32("custom.key")    // Any f32 value
metadata.float64("custom.key")    // Any f64 value
```

## MoE Expert Extraction

Extract weights for Mixture of Experts models:

```rust
use engram_parser::{extract_expert, list_experts};

// List all experts in the model
for (block, expert) in list_experts(&layout) {
    println!("Found expert: block={}, expert={}", block, expert);
}

// Extract weights for a specific expert
let weights = extract_expert(&layout, 0, 0)?;

// Access gate, up, and down projection weights
if let Some(gate) = weights.gate {
    println!("Gate weight: {:?} bytes", gate.bytes.len());
}
if let Some(up) = weights.up {
    println!("Up weight: {:?} bytes", up.bytes.len());
}
if let Some(down) = weights.down {
    println!("Down weight: {:?} bytes", down.bytes.len());
}
```

Supports both:
- **Stacked format**: `blk.{B}.ffn_{role}_exps.weight` (all experts in one tensor)
- **Per-expert format**: `blk.{B}.ffn_{role}.{E}.weight` (separate tensors)

## Type Labels

Convert GGML type IDs to human-readable strings:

```rust
use engram_parser::ggml_type_label;

assert_eq!(ggml_type_label(0), "F32");
assert_eq!(ggml_type_label(1), "F16");
assert_eq!(ggml_type_label(31), "IQ3_M");
assert_eq!(ggml_type_label(999), "unknown");
```

## API Reference

### Core Functions

- `load_gguf(path)`: Load and parse a GGUF file from disk
- `parse_bytes(bytes, path)`: Parse GGUF data from a byte vector
- `list_experts(layout)`: List all MoE experts in the model
- `extract_expert(layout, block, expert)`: Extract weights for a specific expert

### Core Types

- `GgufLayout`: Parsed GGUF file with metadata and tensor directory
- `GgufMetadata`: Architecture and model configuration
- `Tensor`: Tensor directory entry with shape and type information
- `MoeExpertWeights`: Extracted weights for a single MoE expert
- `RawTensor`: Raw tensor bytes with metadata
- `DType`: Type-safe representation of GGML tensor types
- `ParserError`: Error type for all parser operations

### Constants

- `GGML_TYPE_F32` through `GGML_TYPE_IQ3_M`: 32 GGML type constants
- `GGUF_VALUE_TYPE_*`: Metadata value type constants

## Development

```bash
# Run all tests
cargo test --all-features

# Run clippy with strict warnings
cargo clippy --all-features --all-targets -- -D warnings

# Generate documentation
cargo doc --all-features --open
```

## License

Licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

## Contributing

Contributions are welcome! Please ensure:
- All tests pass (`cargo test --all-features`)
- No clippy warnings (`cargo clippy --all-features --all-targets -- -D warnings`)
- New features include comprehensive tests
- Documentation is updated for public APIs

## Related Projects

- **[corinth-canal](https://github.com/rmems/corinth-canal)**: Reference implementation for GGUF parsing and MoE extraction
- **[cortex-tensor](https://github.com/Limen-Neural/cortex-tensor)**: Tensor operations library that consumes engram-parser output

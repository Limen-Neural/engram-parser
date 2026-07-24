// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure-Rust, zero-dependency GGUF parser with MoE support.
//!
//! This crate parses GGUF (GPT-Generated Unified Format) v3 files,
//! extracts metadata and tensor information, and provides utilities
//! for Mixture of Experts (MoE) model analysis.
//!
//! # Features
//!
//! - **Zero dependencies**: Pure Rust implementation with no external crates
//! - **GGUF v3 support**: Full parsing of headers, metadata, and tensor directories
//! - **Comprehensive dtype support**: All GGML tensor types including F32, F16, BF16,
//!   Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K through Q8_K, IQ1_S through IQ4_XS,
//!   integers (I8–I64, F64), plus historical wire type 31 as labeled `Q4_0_4_4`
//!   via [`DType::Other`]
//! - **Type labels**: Human-readable names for all GGML types via [`ggml_type_label`]
//! - **MoE support**: Extract expert weights and analyze mixture-of-experts architectures
//! - **Metadata helpers**: Architecture-aware convenience methods for common fields
//!
//! # Example
//!
//! ```no_run
//! use engram_parser::{load_gguf, ggml_type_label};
//!
//! let layout = load_gguf("model.gguf").unwrap();
//! println!("Architecture: {}", layout.metadata.architecture());
//! println!("Quantization: {}", layout.metadata.quantization());
//!
//! if let Some(block_count) = layout.metadata.block_count() {
//!     println!("Blocks: {}", block_count);
//! }
//!
//! for (name, tensor) in &layout.tensors {
//!     println!("{}: {:?} (type: {})",
//!         name, tensor.dims,
//!         ggml_type_label(tensor.ggml_type));
//! }
//! ```

pub mod error;
pub mod gguf;
pub mod moe;

// Re-export commonly used types at the crate root for convenience.
pub use error::ParserError;
pub use gguf::{
    DType,
    // GGML type constants
    GGML_TYPE_BF16,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_F64,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_I64,
    GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ4_XS,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_0_4_4,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_1,
    GGML_TYPE_Q8_K,
    // Metadata value type constants
    GGUF_VALUE_TYPE_ARRAY,
    GGUF_VALUE_TYPE_BOOL,
    GGUF_VALUE_TYPE_FLOAT32,
    GGUF_VALUE_TYPE_FLOAT64,
    GGUF_VALUE_TYPE_INT8,
    GGUF_VALUE_TYPE_INT16,
    GGUF_VALUE_TYPE_INT32,
    GGUF_VALUE_TYPE_INT64,
    GGUF_VALUE_TYPE_STRING,
    GGUF_VALUE_TYPE_UINT8,
    GGUF_VALUE_TYPE_UINT16,
    GGUF_VALUE_TYPE_UINT32,
    GGUF_VALUE_TYPE_UINT64,
    GgufLayout,
    GgufMetadata,
    Tensor,
    f16_bits_to_f32,
    ggml_type_label,
    load_gguf,
    parse_bytes,
};
pub use moe::{MoeExpertWeights, RawTensor, extract_expert, list_experts};

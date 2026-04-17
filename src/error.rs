//! Error and result types for the engram parser.

use std::io;

use thiserror::Error;

/// All errors surfaced by `engram-parser`.
#[derive(Debug, Error)]
pub enum ParserError {
    /// Failure opening or memory-mapping the checkpoint file.
    #[error("I/O error for '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: io::Error,
    },

    /// GGUF file could not be parsed (bad magic, bad version, overflow, …).
    #[error("unsupported GGUF format in '{path}': {reason}")]
    UnsupportedFormat { path: String, reason: String },

    /// A named tensor was not present in the checkpoint.
    #[error("missing tensor '{name}' in '{path}'")]
    MissingTensor { name: String, path: String },

    /// Tensor uses a GGML dtype this parser does not decode
    /// (typically a block-quantized format such as `Q4_0` or `Q6_K`).
    #[error("tensor '{name}' uses unsupported ggml type {ggml_type}")]
    UnsupportedDtype { name: String, ggml_type: u32 },

    /// Tensor shape is incompatible with the requested operation.
    #[error("tensor '{name}' shape {dims:?} incompatible with {op}")]
    ShapeMismatch {
        name: String,
        dims: Vec<usize>,
        op: &'static str,
    },

    /// MoE expert index was out of range for the isolated block.
    #[error("expert index {index} out of range (num_experts = {num_experts})")]
    ExpertIndexOutOfRange { index: usize, num_experts: usize },

    /// MoE surgery referenced a tensor the checkpoint doesn't contain.
    #[error("MoE block {block} missing required tensor '{name}'")]
    MissingMoeTensor { block: usize, name: String },
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, ParserError>;

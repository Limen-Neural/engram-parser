//! # engram-parser
//!
//! CPU-only GGUF parser. Converts static weight matrices into **engram
//! structures** — raw `f32` voltage vectors an SNN can consume directly —
//! and performs brain-surgeon isolation of Mixture-of-Experts (MoE) layers.
//!
//! ## Responsibilities
//!
//! 1. Memory-map a GGUF v3 checkpoint and inspect its tensor table.
//! 2. Cast static `F16` / `F32` weight tensors into `f32` voltage vectors
//!    ([`EngramVector`], [`EngramMatrix`]).
//! 3. Isolate a specific MoE block ([`MoeLayer::isolate`]) and extract raw
//!    per-expert weight matrices ([`MoeLayer::expert`]).
//!
//! There is intentionally no CUDA, no kernel loader, and no inference
//! runtime. This crate is a parser / dissector.
//!
//! ## Minimal example
//!
//! ```no_run
//! use engram_parser::{GgufCheckpoint, MoeLayer};
//!
//! let checkpoint = GgufCheckpoint::open("models/olmoe.gguf")?;
//!
//! // Cast any static weight matrix into raw voltages.
//! let token_embd: Vec<f32> = checkpoint.extract_as_f32("token_embd.weight")?;
//!
//! // Brain surgery on the first MoE block.
//! let moe = MoeLayer::isolate(&checkpoint, 0)?;
//! let router = moe.router_matrix()?;        // EngramMatrix
//! let expert5 = moe.expert(5)?;             // gate / up / down slices
//! # Ok::<(), engram_parser::ParserError>(())
//! ```

pub mod dtype;
pub mod engram;
pub mod error;
pub mod gguf;
pub mod moe;

pub use dtype::{cast_bytes_to_f32, cast_f16_bytes_to_f32, cast_f32_bytes_to_f32, f16_to_f32, GgmlType};
pub use engram::{EngramMatrix, EngramVector};
pub use error::{ParserError, Result};
pub use gguf::{CheckpointMetadata, GgufCheckpoint, GgufTensor, TensorLayout};
pub use moe::{MoeExpertWeights, MoeLayer};

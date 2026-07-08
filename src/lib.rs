// SPDX-License-Identifier: MIT OR Apache-2.0

//! # engram-parser
//!
//! Pure-Rust, **zero-dependency** parser for `.gguf` and `.safetensors`
//! checkpoint formats with Mixture-of-Experts support.
//!
//! ## GGUF
//!
//! This crate performs **no** neural-network math: it parses the GGUF
//! file format, exposes a tensor directory, and can rip out the raw
//! byte buffers for any single expert's `gate` / `up` / `down`
//! projection. Downstream crates (e.g. SNN or dense inference engines)
//! are responsible for anything involving arithmetic on those bytes.
//!
//! ```no_run
//! use engram_parser::{extract_expert, load_gguf};
//!
//! let layout = load_gguf("./model.gguf")?;
//! println!("architecture = {}", layout.metadata.architecture());
//!
//! let expert = extract_expert(&layout, 0, 3)?;
//! if let Some(gate) = &expert.gate {
//!     println!("expert gate: dims={:?} dtype={:?} bytes={}", gate.dims, gate.dtype, gate.bytes.len());
//! }
//! # Ok::<(), engram_parser::ParserError>(())
//! ```
//!
//! ## Safetensors
//!
//! The [`safetensors`] module provides zero-dependency inspection of
//! Safetensors checkpoints. It parses headers, generates manifests,
//! and discovers MoE router/expert candidates.
//!
//! ```no_run
//! use engram_parser::safetensors::inspect_safetensors_checkpoint;
//!
//! let manifest = inspect_safetensors_checkpoint("./model.safetensors")?;
//! println!("Found {} tensors", manifest.tensors.len());
//!
//! for router in &manifest.candidates.router_candidates {
//!     println!("Router: {} (score={})", router.name, router.score);
//! }
//! # Ok::<(), engram_parser::ParserError>(())
//! ```

pub mod error;
pub mod gguf;
pub mod moe;
pub mod safetensors;

pub use error::{ParserError, Result};
pub use gguf::{DType, GgufLayout, GgufMetadata, Tensor, f16_bits_to_f32, load_gguf, parse_bytes};
pub use moe::{MoeExpertWeights, RawTensor, extract_expert, list_experts};
pub use safetensors::inspect_safetensors_checkpoint;

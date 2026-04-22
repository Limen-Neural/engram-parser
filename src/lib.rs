//! # engram-parser
//!
//! Pure-Rust, **zero-dependency** `.gguf` deserializer and
//! Mixture-of-Experts per-expert weight extractor.
//!
//! This crate performs **no** neural-network math: it parses the GGUF
//! file format, exposes a tensor directory, and can rip out the raw
//! byte buffers for any single expert's `gate` / `up` / `down`
//! projection. Downstream crates (e.g. SNN or dense inference engines)
//! are responsible for anything involving arithmetic on those bytes.
//!
//! ## Quick start
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

pub mod error;
pub mod gguf;
pub mod moe;

pub use error::{ParserError, Result};
pub use gguf::{
    DType, GgufLayout, GgufMetadata, Tensor, f16_bits_to_f32, load_gguf, parse_bytes,
};
pub use moe::{MoeExpertWeights, RawTensor, extract_expert, list_experts};

//! Mixture-of-Experts weight extraction.
//!
//! Locates MoE expert tensors inside a parsed [`GgufLayout`](crate::gguf::GgufLayout)
//! and returns raw per-expert byte buffers. Performs no neural-network
//! math — callers receive `Vec<u8>` + shape/dtype metadata and are
//! responsible for any downstream compute.

mod expert;
mod extract;

pub use expert::{MoeExpertWeights, RawTensor};
pub use extract::{extract_expert, list_experts};

//! GGUF v3 reader: CPU-only, memory-mapped, parser-focused.

mod checkpoint;
mod cursor;
mod layout;
mod tensor;

pub use checkpoint::GgufCheckpoint;
pub use layout::CheckpointMetadata;
pub use tensor::{GgufTensor, TensorLayout};

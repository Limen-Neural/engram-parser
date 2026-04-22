//! GGUF file format: header + metadata + tensor directory.
//!
//! Entry point: [`load_gguf`] reads a `.gguf` file into a [`GgufLayout`]
//! containing parsed metadata, a tensor directory, and the underlying
//! byte buffer. No neural-network math — only deserialization.

mod cursor;
mod layout;
mod tensor;

use std::fs;
use std::path::Path;

pub use layout::{GgufLayout, GgufMetadata};
pub use tensor::{
    DType, GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ3_S, GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, Tensor, f16_bits_to_f32,
};

use crate::error::{ParserError, Result};

/// Load a `.gguf` checkpoint from disk and parse its header, KV
/// metadata, and tensor directory.
///
/// The full file contents are read into memory (no mmap; zero-dep by
/// design). Tensor payloads remain available as raw byte slices via
/// [`GgufLayout::tensor_bytes`].
pub fn load_gguf<P: AsRef<Path>>(path: P) -> Result<GgufLayout> {
    let path_ref = path.as_ref();
    let path_str = path_ref.display().to_string();
    let bytes = fs::read(path_ref).map_err(|e| ParserError::Io {
        path: path_str.clone(),
        source: e,
    })?;
    parse_bytes(bytes, path_str)
}

/// Parse an already-loaded byte buffer as a GGUF checkpoint. Useful for
/// unit tests and in-memory round-trips.
pub fn parse_bytes(bytes: Vec<u8>, path: String) -> Result<GgufLayout> {
    let (metadata, tensors, alignment, tensor_data_offset) = layout::parse_layout(&bytes, &path)?;
    Ok(GgufLayout {
        path,
        metadata,
        tensors,
        alignment,
        tensor_data_offset,
        bytes,
    })
}

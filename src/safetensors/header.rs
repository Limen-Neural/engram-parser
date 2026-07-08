// SPDX-License-Identifier: MIT OR Apache-2.0

//! Safetensors header parsing.
//!
//! Safetensors binary format:
//! - 8 bytes: little-endian u64 header length
//! - N bytes: JSON header
//! - remaining: tensor data (aligned to 8 bytes)
//!
//! The JSON header contains tensor metadata with dtype, shape, and data_offsets.

use crate::error::{ParserError, Result};
use std::collections::BTreeMap;
use super::json::parse_json;

/// Maximum header size (64 MiB) to prevent OOM from malformed files.
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// Parsed Safetensors header.
#[derive(Debug, Clone)]
pub struct SafetensorsHeader {
    /// Global metadata (e.g., format, quantization).
    pub metadata: BTreeMap<String, String>,
    /// Tensor descriptors keyed by tensor name.
    pub tensors: BTreeMap<String, TensorDescriptor>,
}

/// Descriptor for a single tensor in the header.
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    /// Data type (e.g., "F32", "F16", "BF16").
    pub dtype: String,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Byte offsets [start, end) in the data section.
    pub data_offsets: [usize; 2],
}

impl TensorDescriptor {
    /// Calculate the number of elements in this tensor.
#[allow(dead_code)]
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Calculate the byte size of this tensor's data.
    pub fn byte_size(&self) -> usize {
        self.data_offsets[1].saturating_sub(self.data_offsets[0])
    }
}

/// Parse a Safetensors header from a byte slice.
///
/// The slice should contain the full file contents. Returns the parsed header
/// and the offset where tensor data begins.
pub fn parse_header(bytes: &[u8], path: &str) -> Result<(SafetensorsHeader, usize)> {
    if bytes.len() < 8 {
        return Err(ParserError::InvalidLayout {
            path: path.to_owned(),
            reason: "file too short for Safetensors header length".into(),
        });
    }

    // Read header length (little-endian u64)
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    
    if header_len > MAX_HEADER_BYTES {
        return Err(ParserError::InvalidLayout {
            path: path.to_owned(),
            reason: format!("header length {} exceeds maximum {}", header_len, MAX_HEADER_BYTES),
        });
    }

    let header_end = 8 + header_len as usize;
    if header_end > bytes.len() {
        return Err(ParserError::InvalidLayout {
            path: path.to_owned(),
            reason: format!("header extends beyond file (need {} bytes, have {})", header_end, bytes.len()),
        });
    }

    // Parse JSON header
    let header_bytes = &bytes[8..header_end];
    let header_str = std::str::from_utf8(header_bytes).map_err(|e| ParserError::InvalidLayout {
        path: path.to_owned(),
        reason: format!("header is not valid UTF-8: {}", e),
    })?;

    let json = parse_json(header_str, path)?;
    let obj = json.as_object().ok_or_else(|| ParserError::InvalidLayout {
        path: path.to_owned(),
        reason: "Safetensors header must be a JSON object".into(),
    })?;

    // Extract metadata
    let mut metadata = BTreeMap::new();
    if let Some(meta_value) = obj.get("__metadata__") {
        if let Some(meta_obj) = meta_value.as_object() {
            for (key, value) in meta_obj {
                if let Some(s) = value.as_str() {
                    metadata.insert(key.clone(), s.to_owned());
                }
            }
        }
    }

    // Extract tensor descriptors
    let mut tensors = BTreeMap::new();
    for (name, value) in obj {
        if name == "__metadata__" {
            continue;
        }

        let tensor_obj = value.as_object().ok_or_else(|| ParserError::InvalidLayout {
            path: path.to_owned(),
            reason: format!("tensor '{}' must be an object", name),
        })?;

        let dtype = tensor_obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ParserError::InvalidLayout {
                path: path.to_owned(),
                reason: format!("tensor '{}' missing or invalid 'dtype'", name),
            })?
            .to_owned();

        let shape = tensor_obj
            .get("shape")
            .and_then(|v| v.as_usize_array())
            .ok_or_else(|| ParserError::InvalidLayout {
                path: path.to_owned(),
                reason: format!("tensor '{}' missing or invalid 'shape'", name),
            })?;

        let offsets_array = tensor_obj
            .get("data_offsets")
            .and_then(|v| v.as_usize_array())
            .ok_or_else(|| ParserError::InvalidLayout {
                path: path.to_owned(),
                reason: format!("tensor '{}' missing or invalid 'data_offsets'", name),
            })?;

        if offsets_array.len() != 2 {
            return Err(ParserError::InvalidLayout {
                path: path.to_owned(),
                reason: format!("tensor '{}' data_offsets must have exactly 2 elements", name),
            });
        }

        let data_offsets = [offsets_array[0], offsets_array[1]];

        // Validate offsets
        if data_offsets[0] > data_offsets[1] {
            return Err(ParserError::InvalidLayout {
                path: path.to_owned(),
                reason: format!("tensor '{}' has inverted data_offsets: {:?}", name, data_offsets),
            });
        }

        tensors.insert(
            name.clone(),
            TensorDescriptor {
                dtype,
                shape,
                data_offsets,
            },
        );
    }

    Ok((SafetensorsHeader { metadata, tensors }, header_end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_safetensors_bytes(header_json: &str, data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header_json.as_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn parses_valid_header() {
        let json = r#"{
            "__metadata__": {"format": "pt"},
            "tensor1": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}
        }"#;
        let data = vec![0u8; 24];
        let bytes = build_safetensors_bytes(json, &data);
        
        let (header, data_offset) = parse_header(&bytes, "test").unwrap();
        
        assert_eq!(header.metadata.get("format").unwrap(), "pt");
        assert_eq!(header.tensors.len(), 1);
        
        let tensor = header.tensors.get("tensor1").unwrap();
        assert_eq!(tensor.dtype, "F32");
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.data_offsets, [0, 24]);
        assert_eq!(tensor.num_elements(), 6);
        assert_eq!(tensor.byte_size(), 24);
        
        assert_eq!(data_offset, 8 + json.len());
    }

    #[test]
    fn rejects_file_too_short() {
        let bytes = vec![0u8; 7];
        assert!(parse_header(&bytes, "test").is_err());
    }

    #[test]
    fn rejects_header_beyond_file() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&100u64.to_le_bytes()); // claims 100 bytes of header
        bytes.extend_from_slice(b"short"); // but only has 5 bytes
        
        assert!(parse_header(&bytes, "test").is_err());
    }

    #[test]
    fn rejects_excessive_header_size() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(MAX_HEADER_BYTES + 1).to_le_bytes());
        
        assert!(parse_header(&bytes, "test").is_err());
    }

    #[test]
    fn validates_data_offsets() {
        let json = r#"{"tensor": {"dtype": "F32", "shape": [2], "data_offsets": [10, 5]}}"#;
        let bytes = build_safetensors_bytes(json, &[]);
        
        assert!(parse_header(&bytes, "test").is_err());
    }
}

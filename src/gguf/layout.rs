//! GGUF header + tensor-table parser.

use std::collections::HashMap;

use crate::dtype::GgmlType;
use crate::error::{ParserError, Result};
use crate::gguf::cursor::{GgufCursor, GGUF_MAGIC, GGUF_VERSION, VT_ARRAY, VT_STRING};
use crate::gguf::tensor::TensorLayout;

/// Lightweight metadata surfaced from the GGUF key-value table.
///
/// Scalar values are stringified so the parser stays schema-agnostic.
/// Arrays are recorded as `array<type=X,len=N>` so callers can still
/// detect them without the parser caring about their contents.
#[derive(Debug, Clone, Default)]
pub struct CheckpointMetadata {
    /// Tensor-data alignment (`general.alignment`), defaulting to 32.
    pub alignment: usize,
    /// `general.file_type` field when present.
    pub file_type: Option<u32>,
    /// All scalar KV entries as `(key, stringified_value)`.
    pub kv: HashMap<String, String>,
}

pub(crate) struct ParsedLayout {
    pub(crate) metadata: CheckpointMetadata,
    pub(crate) tensors: HashMap<String, TensorLayout>,
}

pub(crate) fn parse_gguf_layout(bytes: &[u8], path: &str) -> Result<ParsedLayout> {
    let mut cursor = GgufCursor::new(bytes, path);

    let magic = cursor.read_exact(4)?;
    if magic != GGUF_MAGIC {
        return Err(ParserError::UnsupportedFormat {
            path: path.to_owned(),
            reason: format!("unrecognised GGUF magic {magic:?}"),
        });
    }

    let version = cursor.read_u32()?;
    if version != GGUF_VERSION {
        return Err(ParserError::UnsupportedFormat {
            path: path.to_owned(),
            reason: format!("unsupported GGUF version {version}; expected {GGUF_VERSION}"),
        });
    }

    let tensor_count = cursor.read_u64()? as usize;
    let kv_count = cursor.read_u64()? as usize;

    let mut alignment = 32usize;
    let mut file_type: Option<u32> = None;
    let mut kv: HashMap<String, String> = HashMap::with_capacity(kv_count);

    for _ in 0..kv_count {
        let key = cursor.read_string()?;
        let value_type = cursor.read_u32()?;

        match (key.as_str(), value_type) {
            ("general.alignment", _) => {
                alignment = cursor.read_numeric_as_usize(value_type)?;
            }
            ("general.file_type", _) => {
                file_type = Some(cursor.read_numeric_as_u64(value_type)? as u32);
            }
            (_, VT_ARRAY) => {
                let nested = cursor.read_u32()?;
                let len = cursor.read_u64()? as usize;
                for _ in 0..len {
                    cursor.skip_value(nested)?;
                }
                kv.insert(key, format!("array<type={nested},len={len}>"));
            }
            (_, VT_STRING) => {
                let value = cursor.read_string()?;
                kv.insert(key, value);
            }
            _ => {
                let rendered = cursor.read_scalar_as_string(value_type)?;
                kv.insert(key, rendered);
            }
        }
    }

    let mut tensors: HashMap<String, TensorLayout> = HashMap::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = cursor.read_string()?;
        let n_dims = cursor.read_u32()? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(cursor.read_u64()? as usize);
        }
        let ggml_type_raw = cursor.read_u32()?;
        let relative_offset = cursor.read_u64()? as usize;

        let n_elements = dims
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| ParserError::UnsupportedFormat {
                path: path.to_owned(),
                reason: format!("tensor '{name}' element count overflow"),
            })?;

        tensors.insert(
            name.clone(),
            TensorLayout {
                name,
                dims,
                ggml_type: GgmlType::from_raw(ggml_type_raw),
                absolute_offset: relative_offset, // patched below once we know the data base
                n_elements,
            },
        );
    }

    let tensor_data_offset = align_up(cursor.offset(), alignment);
    for tensor in tensors.values_mut() {
        tensor.absolute_offset = tensor_data_offset + tensor.absolute_offset;
    }

    Ok(ParsedLayout {
        metadata: CheckpointMetadata {
            alignment,
            file_type,
            kv,
        },
        tensors,
    })
}

fn align_up(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

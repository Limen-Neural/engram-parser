//! Parsed GGUF file layout: metadata KV store + tensor directory.
//!
//! This module is file-agnostic: it operates on an already-loaded
//! `&[u8]` slice (typically the full contents of a `.gguf` file). It
//! does **not** perform any neural-network math — only header parsing,
//! bounds-checked tensor-directory extraction, and metadata coercion.

use std::collections::HashMap;

use super::cursor::{
    GGUF_MAGIC, GGUF_VERSION, GgufCursor, VT_ARRAY, VT_STRING, invalid_layout, unsupported,
};
use super::tensor::{DType, Tensor};
use crate::error::{ParserError, Result};

/// Upper bounds to prevent OOM from malformed inputs.
const MAX_TENSOR_COUNT: u64 = 1_000_000;
const MAX_KV_COUNT: u64 = 1_000_000;
const MAX_TENSOR_DIMS: usize = 8;

/// Parsed GGUF metadata key-value store.
///
/// GGUF stores arbitrary scalar KV pairs. We keep strings and numeric
/// values in two typed maps; array values are skipped (their byte
/// contents remain available via the original file buffer if needed).
#[derive(Debug, Clone, Default)]
pub struct GgufMetadata {
    /// String-typed KV pairs (e.g. `general.architecture = "olmoe"`).
    pub strings: HashMap<String, String>,
    /// Numeric-typed KV pairs coerced to `u64`
    /// (e.g. `olmoe.expert_count = 64`).
    pub numerics: HashMap<String, u64>,
    /// `f32`-typed KV pairs.
    pub floats_32: HashMap<String, f32>,
    /// `f64`-typed KV pairs.
    pub floats_64: HashMap<String, f64>,
}

impl GgufMetadata {
    /// Convenience: architecture string (`general.architecture`) or `"unknown"`.
    pub fn architecture(&self) -> &str {
        self.strings
            .get("general.architecture")
            .map(String::as_str)
            .unwrap_or("unknown")
    }

    /// Convenience: numeric KV coerced to `usize`.
    pub fn numeric(&self, key: &str) -> Option<usize> {
        self.numerics.get(key).map(|&v| v as usize)
    }
}

/// Fully-parsed GGUF checkpoint layout.
///
/// Owns the raw file bytes (loaded into a `Vec<u8>` by [`load_gguf`])
/// and all derived tensor directory + metadata information. Callers use
/// the [`Tensor`] handles + [`GgufLayout::tensor_bytes`] to obtain raw
/// byte slices for each weight.
///
/// [`load_gguf`]: super::load_gguf
#[derive(Debug)]
pub struct GgufLayout {
    /// Path the checkpoint was loaded from (kept for error messages).
    pub path: String,
    /// Parsed KV metadata.
    pub metadata: GgufMetadata,
    /// Name -> tensor directory entry.
    pub tensors: HashMap<String, Tensor>,
    /// Byte alignment specified by the file (default 32).
    pub alignment: usize,
    /// Absolute byte offset within `bytes` where tensor payloads begin.
    pub tensor_data_offset: usize,
    /// Full file contents. Tensor payloads live at `bytes[tensor.absolute_offset ..]`.
    pub bytes: Vec<u8>,
}

impl GgufLayout {
    /// Return a borrowed slice of the raw tensor payload bytes.
    pub fn tensor_bytes<'a>(&'a self, tensor: &Tensor) -> Result<&'a [u8]> {
        let start = tensor.absolute_offset;
        let end = start.checked_add(tensor.byte_len).ok_or_else(|| {
            invalid_layout(
                &self.path,
                format!("tensor '{}' byte-length overflow", tensor.name),
            )
        })?;
        if end > self.bytes.len() {
            return Err(invalid_layout(
                &self.path,
                format!(
                    "tensor '{}' extends beyond mapped file ({end} > {})",
                    tensor.name,
                    self.bytes.len()
                ),
            ));
        }
        Ok(&self.bytes[start..end])
    }

    /// Lookup a tensor by exact name.
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors.get(name).ok_or_else(|| ParserError::MissingTensor {
            name: name.to_owned(),
            path: self.path.clone(),
        })
    }

    /// Find all tensors whose name ends with the given suffix. Useful
    /// for locating per-block MoE tensors like `ffn_gate_exps.weight`.
    pub fn find_tensors_with_suffix<'a>(&'a self, suffix: &str) -> Vec<&'a Tensor> {
        let mut matches: Vec<&Tensor> = self
            .tensors
            .values()
            .filter(|t| t.name.ends_with(suffix))
            .collect();
        matches.sort_unstable_by_key(|t| tensor_block_sort_key(&t.name));
        matches
    }
}

/// Parse the GGUF header + KV metadata + tensor directory out of a
/// byte slice. Does not validate payload bytes, only directory offsets.
pub(crate) fn parse_layout(bytes: &[u8], path: &str) -> Result<(GgufMetadata, HashMap<String, Tensor>, usize, usize)> {
    let mut cursor = GgufCursor::new(bytes, path);

    let magic = cursor.read_exact(4)?;
    if magic != GGUF_MAGIC {
        return Err(unsupported(
            path,
            format!("unrecognised GGUF magic bytes: {magic:?}"),
        ));
    }

    let version = cursor.read_u32()?;
    if version != GGUF_VERSION {
        return Err(unsupported(
            path,
            format!("unsupported GGUF version {version}; expected {GGUF_VERSION}"),
        ));
    }

    let tensor_count_raw = cursor.read_u64()?;
    if tensor_count_raw > MAX_TENSOR_COUNT {
        return Err(unsupported(
            path,
            format!("tensor_count {tensor_count_raw} exceeds sanity limit {MAX_TENSOR_COUNT}"),
        ));
    }
    let tensor_count = tensor_count_raw as usize;

    let kv_count_raw = cursor.read_u64()?;
    if kv_count_raw > MAX_KV_COUNT {
        return Err(unsupported(
            path,
            format!("kv_count {kv_count_raw} exceeds sanity limit {MAX_KV_COUNT}"),
        ));
    }
    let kv_count = kv_count_raw as usize;

    let mut alignment: usize = 32;
    let mut metadata = GgufMetadata::default();

    for _ in 0..kv_count {
        let key = cursor.read_string()?;
        let value_type = cursor.read_u32()?;
        match key.as_str() {
            "general.alignment" => {
                alignment = cursor.read_numeric_as_usize(value_type)?.max(1);
            }
            _ => {
                capture_kv(&mut cursor, &mut metadata, key, value_type)?;
            }
        }
    }

    let mut tensors = HashMap::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = cursor.read_string()?;
        let n_dims_raw = cursor.read_u32()? as usize;
        if n_dims_raw > MAX_TENSOR_DIMS {
            return Err(unsupported(
                path,
                format!("tensor '{name}' has {n_dims_raw} dims; max {MAX_TENSOR_DIMS}"),
            ));
        }
        let mut dims = Vec::with_capacity(n_dims_raw);
        for _ in 0..n_dims_raw {
            dims.push(cursor.read_u64()? as usize);
        }
        let ggml_type = cursor.read_u32()?;
        let relative_offset = cursor.read_u64()? as usize;
        let dtype = DType::from_ggml_type(ggml_type);

        let n_elements = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)).ok_or_else(
            || invalid_layout(path, format!("tensor '{name}' element count overflow")),
        )?;
        let byte_len = dtype.byte_len_for_elements(n_elements).ok_or_else(|| {
            invalid_layout(
                path,
                format!(
                    "tensor '{name}' (ggml_type={ggml_type}) has unknown byte-length for {n_elements} elements",
                ),
            )
        })?;

        tensors.insert(
            name.clone(),
            Tensor {
                name,
                dims,
                dtype,
                ggml_type,
                n_elements,
                byte_len,
                relative_offset,
                absolute_offset: 0,
            },
        );
    }

    let tensor_data_offset = align_up(cursor.offset(), alignment);
    for tensor in tensors.values_mut() {
        tensor.absolute_offset = tensor_data_offset + tensor.relative_offset;
    }

    Ok((metadata, tensors, alignment, tensor_data_offset))
}

fn capture_kv(
    cursor: &mut GgufCursor<'_>,
    metadata: &mut GgufMetadata,
    key: String,
    value_type: u32,
) -> Result<()> {
    use super::cursor::{
        VT_BOOL, VT_F32, VT_F64, VT_I8, VT_I16, VT_I32, VT_I64, VT_U8, VT_U16, VT_U32, VT_U64,
    };
    match value_type {
        VT_U8 | VT_I8 | VT_U16 | VT_I16 | VT_U32 | VT_I32 | VT_U64 | VT_I64 | VT_BOOL => {
            let v = cursor.read_numeric_as_u64(value_type)?;
            metadata.numerics.insert(key, v);
        }
        VT_F32 => {
            let v = cursor.read_f32()?;
            metadata.floats_32.insert(key, v);
        }
        VT_F64 => {
            let v = cursor.read_f64()?;
            metadata.floats_64.insert(key, v);
        }
        VT_STRING => {
            let v = cursor.read_string()?;
            metadata.strings.insert(key, v);
        }
        VT_ARRAY => {
            cursor.skip_value(value_type)?;
        }
        _ => {
            cursor.skip_value(value_type)?;
        }
    }
    Ok(())
}

fn align_up(value: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

fn tensor_block_sort_key(name: &str) -> (usize, String) {
    let block = name
        .strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(idx, _)| idx.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    (block, name.to_owned())
}

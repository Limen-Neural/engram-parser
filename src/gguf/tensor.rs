//! Tensor directory entry + dtype enumeration.
//!
//! A [`Tensor`] is a pure metadata descriptor: name, shape, dtype, and
//! byte offset within the file. It owns no weight data itself — callers
//! pass it back to [`GgufLayout::tensor_bytes`](super::layout::GgufLayout::tensor_bytes)
//! to obtain the raw `&[u8]` payload.

use crate::error::{ParserError, Result};

/// GGML tensor dtype codes encountered in GGUF checkpoints.
///
/// Values mirror the `GGML_TYPE_*` constants in `ggml.h`. The parser
/// understands their byte layout (for bounds checking) but performs no
/// arithmetic — raw bytes are returned as-is. `BF16` layout parsing is
/// supported even though no BF16→F32 conversion is provided.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit little-endian float.
    F32,
    /// 16-bit IEEE-754 half float.
    F16,
    /// Google Brain bfloat16 (`GGML_TYPE_BF16 = 30`).
    BF16,
    /// `GGML_TYPE_Q8_0` blocked 8-bit quantization.
    Q8_0,
    /// `GGML_TYPE_Q5_K` k-quant.
    Q5_K,
    /// `GGML_TYPE_Q4_K` k-quant.
    Q4_K,
    /// `GGML_TYPE_Q6_K` k-quant.
    Q6_K,
    /// `GGML_TYPE_IQ3_S` i-quant (3.44 bpw).
    IQ3_S,
    /// Any other GGML dtype not explicitly enumerated above. The raw
    /// `u32` code is preserved so callers can dispatch on it.
    Other(u32),
}

/// `GGML_TYPE_*` constants used to map raw u32 codes to [`DType`].
pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_K: u32 = 12;
pub const GGML_TYPE_Q5_K: u32 = 13;
pub const GGML_TYPE_Q6_K: u32 = 14;
pub const GGML_TYPE_Q8_0: u32 = 8;
pub const GGML_TYPE_IQ3_S: u32 = 21;
pub const GGML_TYPE_BF16: u32 = 30;

impl DType {
    /// Map a raw `ggml_type` code to a [`DType`] enum.
    pub fn from_ggml_type(code: u32) -> Self {
        match code {
            GGML_TYPE_F32 => Self::F32,
            GGML_TYPE_F16 => Self::F16,
            GGML_TYPE_BF16 => Self::BF16,
            GGML_TYPE_Q8_0 => Self::Q8_0,
            GGML_TYPE_Q5_K => Self::Q5_K,
            GGML_TYPE_Q4_K => Self::Q4_K,
            GGML_TYPE_Q6_K => Self::Q6_K,
            GGML_TYPE_IQ3_S => Self::IQ3_S,
            other => Self::Other(other),
        }
    }

    /// The raw `ggml_type` code for this dtype.
    pub fn ggml_type(self) -> u32 {
        match self {
            Self::F32 => GGML_TYPE_F32,
            Self::F16 => GGML_TYPE_F16,
            Self::BF16 => GGML_TYPE_BF16,
            Self::Q8_0 => GGML_TYPE_Q8_0,
            Self::Q5_K => GGML_TYPE_Q5_K,
            Self::Q4_K => GGML_TYPE_Q4_K,
            Self::Q6_K => GGML_TYPE_Q6_K,
            Self::IQ3_S => GGML_TYPE_IQ3_S,
            Self::Other(code) => code,
        }
    }

    /// Size in bytes of `n_elements` values of this dtype, or `None`
    /// for quantized/unknown layouts whose byte-length depends on the
    /// tensor's inner dimension (not a simple `n * sizeof(T)`).
    ///
    /// For quantized dtypes we return the correct blocked byte count
    /// when the total element count is divisible by the block size;
    /// otherwise `None`.
    pub fn byte_len_for_elements(self, n_elements: usize) -> Option<usize> {
        match self {
            Self::F32 => Some(n_elements.checked_mul(4)?),
            Self::F16 | Self::BF16 => Some(n_elements.checked_mul(2)?),
            Self::Q8_0 => block_bytes(n_elements, 32, 2 + 32),
            Self::Q5_K => block_bytes(n_elements, 256, 2 + 2 + 12 + 32 + 128),
            Self::Q4_K => block_bytes(n_elements, 256, 2 + 2 + 12 + 128),
            Self::Q6_K => block_bytes(n_elements, 256, 128 + 64 + 16 + 2),
            // Unknown / unsupported quantizations: byte length cannot
            // be derived without the ggml block descriptor.
            Self::IQ3_S | Self::Other(_) => None,
        }
    }

    /// Whether the dtype is a plain (non-quantized) float layout.
    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }

    /// Byte width of a single element, or `None` for block-quantized dtypes.
    pub fn element_size(self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 | Self::BF16 => Some(2),
            _ => None,
        }
    }
}

fn block_bytes(n_elements: usize, block_size: usize, block_bytes: usize) -> Option<usize> {
    if n_elements % block_size != 0 {
        return None;
    }
    (n_elements / block_size).checked_mul(block_bytes)
}

/// Tensor directory entry.
///
/// A `Tensor` is a lightweight descriptor — it does **not** own weight
/// data. Use [`GgufLayout::tensor_bytes`](super::layout::GgufLayout::tensor_bytes)
/// to fetch the raw payload slice for this tensor.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Name of the tensor as stored in the GGUF directory
    /// (e.g. `"blk.0.ffn_gate_exps.weight"`).
    pub name: String,
    /// Dimensions, in GGML order (innermost first).
    pub dims: Vec<usize>,
    /// Parsed dtype.
    pub dtype: DType,
    /// Raw `ggml_type` code as stored in the file.
    pub ggml_type: u32,
    /// Total number of elements (product of `dims`).
    pub n_elements: usize,
    /// Byte length of the tensor payload.
    pub byte_len: usize,
    /// Byte offset of the payload relative to the tensor-data section.
    pub relative_offset: usize,
    /// Absolute byte offset of the payload within the file (filled in
    /// after the data section start is resolved).
    pub absolute_offset: usize,
}

impl Tensor {
    /// Reinterpret the tensor's payload as a `&[f32]` slice.
    ///
    /// Returns an error if the dtype is not `F32` or if alignment /
    /// length invariants are violated.
    pub fn as_f32_slice<'a>(&self, bytes: &'a [u8]) -> Result<&'a [f32]> {
        if self.dtype != DType::F32 {
            return Err(ParserError::UnsupportedFormat {
                path: self.name.clone(),
                reason: format!("as_f32_slice called on dtype {:?}", self.dtype),
            });
        }
        if bytes.len() != self.n_elements * 4 {
            return Err(ParserError::InvalidLayout {
                path: self.name.clone(),
                reason: format!(
                    "f32 byte-length mismatch: bytes={}, expected={}",
                    bytes.len(),
                    self.n_elements * 4
                ),
            });
        }
        if bytes.as_ptr() as usize % std::mem::align_of::<f32>() != 0 {
            return Err(ParserError::InvalidLayout {
                path: self.name.clone(),
                reason: "f32 tensor payload is not 4-byte aligned".into(),
            });
        }
        // SAFETY: dtype, length, and alignment all checked above; lifetime
        // is tied to the input slice which borrows the owning layout.
        let slice = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.n_elements)
        };
        Ok(slice)
    }

    /// Reinterpret the tensor's payload as a `&[u16]` slice of raw F16
    /// or BF16 bits (no conversion performed).
    pub fn as_u16_bits<'a>(&self, bytes: &'a [u8]) -> Result<&'a [u16]> {
        if !matches!(self.dtype, DType::F16 | DType::BF16) {
            return Err(ParserError::UnsupportedFormat {
                path: self.name.clone(),
                reason: format!("as_u16_bits called on dtype {:?}", self.dtype),
            });
        }
        if bytes.len() != self.n_elements * 2 {
            return Err(ParserError::InvalidLayout {
                path: self.name.clone(),
                reason: format!(
                    "16-bit byte-length mismatch: bytes={}, expected={}",
                    bytes.len(),
                    self.n_elements * 2
                ),
            });
        }
        if bytes.as_ptr() as usize % std::mem::align_of::<u16>() != 0 {
            return Err(ParserError::InvalidLayout {
                path: self.name.clone(),
                reason: "16-bit tensor payload is not 2-byte aligned".into(),
            });
        }
        // SAFETY: dtype, length, and alignment checked above.
        let slice = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const u16, self.n_elements)
        };
        Ok(slice)
    }

    /// Decode an F16 tensor into a newly-allocated `Vec<f32>`. The only
    /// numeric conversion exposed by this crate — purely a bit-level
    /// reinterpretation of each 16-bit half into a 32-bit float.
    pub fn dequantize_f16(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        if self.dtype != DType::F16 {
            return Err(ParserError::UnsupportedFormat {
                path: self.name.clone(),
                reason: format!("dequantize_f16 called on dtype {:?}", self.dtype),
            });
        }
        if bytes.len() != self.n_elements * 2 {
            return Err(ParserError::InvalidLayout {
                path: self.name.clone(),
                reason: format!(
                    "f16 byte-length mismatch: bytes={}, expected={}",
                    bytes.len(),
                    self.n_elements * 2
                ),
            });
        }
        let out: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        Ok(out)
    }
}

/// Convert a 16-bit IEEE-754 half-precision float (as raw bits) into a
/// 32-bit float. Pure bit manipulation — no external math library.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits as u32) & 0x7C00) >> 10;
    let mant = ((bits as u32) & 0x03FF) << 13;
    let val = if exp == 0 {
        mant
    } else if exp == 31 {
        0x7F800000 | mant
    } else {
        ((exp + 127 - 15) << 23) | mant
    };
    f32::from_bits(sign | val)
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tensor directory entry + dtype enumeration + GGML type helpers.
//!
//! A [`Tensor`] is a pure-metadata descriptor: name, shape, dtype, and
//! byte offset within the file. It owns no weight data itself — callers
//! pass it back to [`GgufLayout::tensor_bytes`](super::layout::GgufLayout::tensor_bytes)
//! to obtain the raw `&[u8]` payload.
//!
//! ## GGML type constants
//!
//! The `GGML_TYPE_*` constants mirror the `ggml.h` enum and cover every
//! dtype that has appeared in a GGUF v3 checkpoint to date. The
//! [`ggml_type_label`] helper maps any `u32` code to a short human
//! string for diagnostics.

use crate::error::{ParserError, Result};

// ---------------------------------------------------------------------------
// GGML type constants (mirror `ggml.h` as of 2025-06).
// ---------------------------------------------------------------------------

/// `GGML_TYPE_F32` — 32-bit IEEE-754 float.
pub const GGML_TYPE_F32: u32 = 0;
/// `GGML_TYPE_F16` — 16-bit IEEE-754 half float.
pub const GGML_TYPE_F16: u32 = 1;
/// `GGML_TYPE_Q4_0` — 4-bit quantization (symmetric, block size 32).
pub const GGML_TYPE_Q4_0: u32 = 2;
/// `GGML_TYPE_Q4_1` — 4-bit quantization (with min, block size 32).
pub const GGML_TYPE_Q4_1: u32 = 3;
/// `GGML_TYPE_Q5_0` — 5-bit quantization (symmetric, block size 32).
pub const GGML_TYPE_Q5_0: u32 = 6;
/// `GGML_TYPE_Q5_1` — 5-bit quantization (with min, block size 32).
pub const GGML_TYPE_Q5_1: u32 = 7;
/// `GGML_TYPE_Q8_0` — 8-bit quantization (symmetric, block size 32).
pub const GGML_TYPE_Q8_0: u32 = 8;
/// `GGML_TYPE_Q8_1` — 8-bit quantization (with min, block size 32).
pub const GGML_TYPE_Q8_1: u32 = 9;
/// `GGML_TYPE_Q2_K` — k-quant 2-bit.
pub const GGML_TYPE_Q2_K: u32 = 10;
/// `GGML_TYPE_Q3_K` — k-quant 3-bit.
pub const GGML_TYPE_Q3_K: u32 = 11;
/// `GGML_TYPE_Q4_K` — k-quant 4-bit.
pub const GGML_TYPE_Q4_K: u32 = 12;
/// `GGML_TYPE_Q5_K` — k-quant 5-bit.
pub const GGML_TYPE_Q5_K: u32 = 13;
/// `GGML_TYPE_Q6_K` — k-quant 6-bit.
pub const GGML_TYPE_Q6_K: u32 = 14;
/// `GGML_TYPE_Q8_K` — k-quant 8-bit.
pub const GGML_TYPE_Q8_K: u32 = 15;
/// `GGML_TYPE_IQ2_XXS` — i-quant 2-bit extra-extra-small.
pub const GGML_TYPE_IQ2_XXS: u32 = 16;
/// `GGML_TYPE_IQ2_XS` — i-quant 2-bit extra-small.
pub const GGML_TYPE_IQ2_XS: u32 = 17;
/// `GGML_TYPE_IQ3_XXS` — i-quant 3-bit extra-extra-small.
pub const GGML_TYPE_IQ3_XXS: u32 = 18;
/// `GGML_TYPE_IQ1_S` — i-quant 1-bit small.
pub const GGML_TYPE_IQ1_S: u32 = 19;
/// `GGML_TYPE_IQ4_NL` — i-quant 4-bit non-linear.
pub const GGML_TYPE_IQ4_NL: u32 = 20;
/// `GGML_TYPE_IQ3_S` — i-quant 3-bit small (3.44 bpw).
pub const GGML_TYPE_IQ3_S: u32 = 21;
/// `GGML_TYPE_IQ2_S` — i-quant 2-bit small.
pub const GGML_TYPE_IQ2_S: u32 = 22;
/// `GGML_TYPE_IQ4_XS` — i-quant 4-bit extra-small.
pub const GGML_TYPE_IQ4_XS: u32 = 23;
/// `GGML_TYPE_I8` — 8-bit signed integer.
pub const GGML_TYPE_I8: u32 = 24;
/// `GGML_TYPE_I16` — 16-bit signed integer.
pub const GGML_TYPE_I16: u32 = 25;
/// `GGML_TYPE_I32` — 32-bit signed integer.
pub const GGML_TYPE_I32: u32 = 26;
/// `GGML_TYPE_I64` — 64-bit signed integer.
pub const GGML_TYPE_I64: u32 = 27;
/// `GGML_TYPE_F64` — 64-bit IEEE-754 double float.
pub const GGML_TYPE_F64: u32 = 28;
/// `GGML_TYPE_IQ1_M` — i-quant 1-bit medium.
pub const GGML_TYPE_IQ1_M: u32 = 29;
/// `GGML_TYPE_BF16` — Google Brain bfloat16.
pub const GGML_TYPE_BF16: u32 = 30;
/// `GGML_TYPE_IQ3_M` — i-quant 3-bit medium.
pub const GGML_TYPE_IQ3_M: u32 = 31;

// ---------------------------------------------------------------------------
// Human-readable label helper.
// ---------------------------------------------------------------------------

/// Map a raw GGML `ggml_type` `u32` code to a short human-readable label.
///
/// Returns `"unknown"` for codes not in the known set. This is a pure
/// function with no side effects — safe to call from diagnostics, `Debug`
/// impls, or logging.
///
/// # Examples
///
/// ```
/// use engram_parser::ggml_type_label;
/// assert_eq!(ggml_type_label(0), "F32");
/// assert_eq!(ggml_type_label(31), "IQ3_M");
/// assert_eq!(ggml_type_label(9999), "unknown");
/// ```
pub fn ggml_type_label(ggml_type: u32) -> &'static str {
    match ggml_type {
        GGML_TYPE_F32 => "F32",
        GGML_TYPE_F16 => "F16",
        GGML_TYPE_Q4_0 => "Q4_0",
        GGML_TYPE_Q4_1 => "Q4_1",
        GGML_TYPE_Q5_0 => "Q5_0",
        GGML_TYPE_Q5_1 => "Q5_1",
        GGML_TYPE_Q8_0 => "Q8_0",
        GGML_TYPE_Q8_1 => "Q8_1",
        GGML_TYPE_Q2_K => "Q2_K",
        GGML_TYPE_Q3_K => "Q3_K",
        GGML_TYPE_Q4_K => "Q4_K",
        GGML_TYPE_Q5_K => "Q5_K",
        GGML_TYPE_Q6_K => "Q6_K",
        GGML_TYPE_Q8_K => "Q8_K",
        GGML_TYPE_IQ2_XXS => "IQ2_XXS",
        GGML_TYPE_IQ2_XS => "IQ2_XS",
        GGML_TYPE_IQ3_XXS => "IQ3_XXS",
        GGML_TYPE_IQ1_S => "IQ1_S",
        GGML_TYPE_IQ4_NL => "IQ4_NL",
        GGML_TYPE_IQ3_S => "IQ3_S",
        GGML_TYPE_IQ2_S => "IQ2_S",
        GGML_TYPE_IQ4_XS => "IQ4_XS",
        GGML_TYPE_I8 => "I8",
        GGML_TYPE_I16 => "I16",
        GGML_TYPE_I32 => "I32",
        GGML_TYPE_I64 => "I64",
        GGML_TYPE_F64 => "F64",
        GGML_TYPE_IQ1_M => "IQ1_M",
        GGML_TYPE_BF16 => "BF16",
        GGML_TYPE_IQ3_M => "IQ3_M",
        _ => "unknown",
    }
}

// ---------------------------------------------------------------------------
// DType enum.
// ---------------------------------------------------------------------------

/// GGML tensor dtype codes encountered in GGUF checkpoints.
///
/// Values mirror the `GGML_TYPE_*` constants in `ggml.h`. The parser
/// understands their byte layout (for bounds checking) but performs no
/// arithmetic — raw bytes are returned as-is. `BF16` layout parsing is
/// supported even though no BF16→F32 conversion is provided.
///
/// Types not explicitly enumerated are captured by [`DType::Other(u32)`]
/// which preserves the raw code for callers to dispatch on.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit little-endian float (`GGML_TYPE_F32 = 0`).
    F32,
    /// 16-bit IEEE-754 half float (`GGML_TYPE_F16 = 1`).
    F16,
    /// `GGML_TYPE_Q4_0` — 4-bit symmetric quantization (block size 32).
    Q4_0,
    /// `GGML_TYPE_Q4_1` — 4-bit quantization with min (block size 32).
    Q4_1,
    /// `GGML_TYPE_Q5_0` — 5-bit symmetric quantization (block size 32).
    Q5_0,
    /// `GGML_TYPE_Q5_1` — 5-bit quantization with min (block size 32).
    Q5_1,
    /// `GGML_TYPE_Q8_0` — 8-bit symmetric quantization (block size 32).
    Q8_0,
    /// `GGML_TYPE_Q8_1` — 8-bit quantization with min (block size 32).
    Q8_1,
    /// `GGML_TYPE_Q2_K` — k-quant 2-bit.
    Q2_K,
    /// `GGML_TYPE_Q3_K` — k-quant 3-bit.
    Q3_K,
    /// `GGML_TYPE_Q4_K` — k-quant 4-bit.
    Q4_K,
    /// `GGML_TYPE_Q5_K` — k-quant 5-bit.
    Q5_K,
    /// `GGML_TYPE_Q6_K` — k-quant 6-bit.
    Q6_K,
    /// `GGML_TYPE_Q8_K` — k-quant 8-bit.
    Q8_K,
    /// `GGML_TYPE_IQ3_S` — i-quant 3-bit small (3.44 bpw).
    IQ3_S,
    /// `GGML_TYPE_IQ3_M` — i-quant 3-bit medium.
    IQ3_M,
    /// Google Brain bfloat16 (`GGML_TYPE_BF16 = 30`).
    BF16,
    /// 64-bit IEEE-754 double float (`GGML_TYPE_F64 = 28`).
    F64,
    /// 8-bit signed integer (`GGML_TYPE_I8 = 24`).
    I8,
    /// 16-bit signed integer (`GGML_TYPE_I16 = 25`).
    I16,
    /// 32-bit signed integer (`GGML_TYPE_I32 = 26`).
    I32,
    /// 64-bit signed integer (`GGML_TYPE_I64 = 27`).
    I64,
    /// Any other GGML dtype not explicitly enumerated above. The raw
    /// `u32` code is preserved so callers can dispatch on it.
    Other(u32),
}

impl DType {
    /// Map a raw `ggml_type` code to a [`DType`] enum.
    pub fn from_ggml_type(code: u32) -> Self {
        match code {
            GGML_TYPE_F32 => Self::F32,
            GGML_TYPE_F16 => Self::F16,
            GGML_TYPE_Q4_0 => Self::Q4_0,
            GGML_TYPE_Q4_1 => Self::Q4_1,
            GGML_TYPE_Q5_0 => Self::Q5_0,
            GGML_TYPE_Q5_1 => Self::Q5_1,
            GGML_TYPE_Q8_0 => Self::Q8_0,
            GGML_TYPE_Q8_1 => Self::Q8_1,
            GGML_TYPE_Q2_K => Self::Q2_K,
            GGML_TYPE_Q3_K => Self::Q3_K,
            GGML_TYPE_Q4_K => Self::Q4_K,
            GGML_TYPE_Q5_K => Self::Q5_K,
            GGML_TYPE_Q6_K => Self::Q6_K,
            GGML_TYPE_Q8_K => Self::Q8_K,
            GGML_TYPE_IQ3_S => Self::IQ3_S,
            GGML_TYPE_IQ3_M => Self::IQ3_M,
            GGML_TYPE_BF16 => Self::BF16,
            GGML_TYPE_F64 => Self::F64,
            GGML_TYPE_I8 => Self::I8,
            GGML_TYPE_I16 => Self::I16,
            GGML_TYPE_I32 => Self::I32,
            GGML_TYPE_I64 => Self::I64,
            other => Self::Other(other),
        }
    }

    /// The raw `ggml_type` code for this dtype.
    pub fn ggml_type(self) -> u32 {
        match self {
            Self::F32 => GGML_TYPE_F32,
            Self::F16 => GGML_TYPE_F16,
            Self::Q4_0 => GGML_TYPE_Q4_0,
            Self::Q4_1 => GGML_TYPE_Q4_1,
            Self::Q5_0 => GGML_TYPE_Q5_0,
            Self::Q5_1 => GGML_TYPE_Q5_1,
            Self::Q8_0 => GGML_TYPE_Q8_0,
            Self::Q8_1 => GGML_TYPE_Q8_1,
            Self::Q2_K => GGML_TYPE_Q2_K,
            Self::Q3_K => GGML_TYPE_Q3_K,
            Self::Q4_K => GGML_TYPE_Q4_K,
            Self::Q5_K => GGML_TYPE_Q5_K,
            Self::Q6_K => GGML_TYPE_Q6_K,
            Self::Q8_K => GGML_TYPE_Q8_K,
            Self::IQ3_S => GGML_TYPE_IQ3_S,
            Self::IQ3_M => GGML_TYPE_IQ3_M,
            Self::BF16 => GGML_TYPE_BF16,
            Self::F64 => GGML_TYPE_F64,
            Self::I8 => GGML_TYPE_I8,
            Self::I16 => GGML_TYPE_I16,
            Self::I32 => GGML_TYPE_I32,
            Self::I64 => GGML_TYPE_I64,
            Self::Other(code) => code,
        }
    }

    /// Short human-readable label for this dtype (e.g. `"F32"`, `"IQ3_M"`).
    ///
    /// Delegates to [`ggml_type_label`] for `Other(code)` variants.
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::IQ3_S => "IQ3_S",
            Self::IQ3_M => "IQ3_M",
            Self::BF16 => "BF16",
            Self::F64 => "F64",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Other(code) => ggml_type_label(code),
        }
    }

    /// Size in bytes of `n_elements` values of this dtype, or `None`
    /// for quantized/unknown layouts whose byte-length depends on the
    /// tensor's inner dimension (not a simple `n * sizeof(T)`).
    ///
    /// For quantized dtypes we return the correct blocked byte count
    /// when the total element count is divisible by the block size;
    /// otherwise `None`.
    ///
    /// Block sizes follow the GGML specification:
    /// - Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1: block size 32
    /// - Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K: block size 256
    /// - IQ3_S: block size 256 (opaque, byte count not computed)
    /// - IQ3_M: block size 256 (opaque, byte count not computed)
    pub fn byte_len_for_elements(self, n_elements: usize) -> Option<usize> {
        match self {
            Self::F32 => Some(n_elements.checked_mul(4)?),
            Self::F16 | Self::BF16 => Some(n_elements.checked_mul(2)?),
            Self::F64 | Self::I64 => Some(n_elements.checked_mul(8)?),
            Self::I32 => Some(n_elements.checked_mul(4)?),
            Self::I16 => Some(n_elements.checked_mul(2)?),
            Self::I8 => Some(n_elements),
            // Q*_0/Q*_1 blocked quants: block size 32.
            // Per-block byte counts (header + packed weights):
            //   Q4_0: 18 B/block, Q4_1: 20 B/block
            //   Q5_0: 22 B/block, Q5_1: 24 B/block
            //   Q8_0: 34 B/block, Q8_1: 36 B/block
            Self::Q4_0 => blocked_byte_len(n_elements, 32, 18),
            Self::Q4_1 => blocked_byte_len(n_elements, 32, 20),
            Self::Q5_0 => blocked_byte_len(n_elements, 32, 22),
            Self::Q5_1 => blocked_byte_len(n_elements, 32, 24),
            Self::Q8_0 => blocked_byte_len(n_elements, 32, 34),
            Self::Q8_1 => blocked_byte_len(n_elements, 32, 36),
            // K-quants: block size 256.
            //   Q2_K: 84 B/256 elements (2.625 bpw)
            //   Q3_K: 110 B/256 elements (3.4375 bpw)
            //   Q4_K: 144 B/256 elements (4.5 bpw)
            //   Q5_K: 176 B/256 elements (5.5 bpw)
            //   Q6_K: 210 B/256 elements (6.5625 bpw)
            //   Q8_K: 292 B/256 elements (9.125 bpw)
            Self::Q2_K => blocked_byte_len(n_elements, 256, 84),
            Self::Q3_K => blocked_byte_len(n_elements, 256, 110),
            Self::Q4_K => blocked_byte_len(n_elements, 256, 144),
            Self::Q5_K => blocked_byte_len(n_elements, 256, 176),
            Self::Q6_K => blocked_byte_len(n_elements, 256, 210),
            Self::Q8_K => blocked_byte_len(n_elements, 256, 292),
            // IQ3_S: block size 256, 50 bytes per block
            //   d(2) + qs(32) + qh(4) + signs(12) = 50 bytes
            Self::IQ3_S => blocked_byte_len(n_elements, 256, 50),
            // IQ3_M: block size 256, 111 bytes per block
            //   d(2) + hmask(32) + qs(64) + scales(12) + scales_h(1) = 111 bytes
            Self::IQ3_M => blocked_byte_len(n_elements, 256, 111),
            // Other opaque / unknown quant types.
            Self::Other(_) => None,
        }
    }

    /// `true` if this dtype's byte layout is a fixed multiple of the
    /// element count (F32, F16, BF16, F64, I8, I16, I32, I64, and
    /// the simple blocked quants when aligned).
    pub fn has_known_byte_layout(self) -> bool {
        !matches!(self, Self::Other(_))
    }
}

/// Compute the total byte length for a blocked quantization format.
///
/// Returns `None` if `n_elements` is not divisible by `block_size`
/// or if the multiplication overflows.
fn blocked_byte_len(n_elements: usize, block_size: usize, bytes_per_block: usize) -> Option<usize> {
    if !n_elements.is_multiple_of(block_size) {
        return None;
    }
    let n_blocks = n_elements / block_size;
    n_blocks.checked_mul(bytes_per_block)
}

// ---------------------------------------------------------------------------
// Tensor directory entry.
// ---------------------------------------------------------------------------

/// A single tensor's directory entry: name, shape, dtype, and offsets.
///
/// This is a metadata-only descriptor — it owns no weight data. Use
/// [`GgufLayout::tensor_bytes`](super::layout::GgufLayout::tensor_bytes)
/// to obtain the raw payload.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Full tensor name as stored in the GGUF directory.
    pub name: String,
    /// Shape dimensions (GGML innermost-first order).
    pub dims: Vec<usize>,
    /// Parsed dtype enum.
    pub dtype: DType,
    /// Raw `ggml_type` code (preserved for round-tripping).
    pub ggml_type: u32,
    /// Total number of elements (product of `dims`).
    pub n_elements: usize,
    /// Total byte length of the tensor payload.
    pub byte_len: usize,
    /// Offset relative to the tensor data region start.
    pub relative_offset: usize,
    /// Absolute byte offset within the file buffer.
    pub absolute_offset: usize,
}

impl Tensor {
    /// Read the raw tensor bytes as little-endian `f32` values.
    ///
    /// Only valid for [`DType::F32`] tensors; returns an error otherwise.
    pub fn read_f32_values(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(ParserError::UnsupportedFormat {
                path: self.name.clone(),
                reason: format!("read_f32_values called on dtype {:?}", self.dtype),
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
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    /// Decode F16/BF16 tensor bytes into raw `u16` lane values using
    /// little-endian chunk parsing (no numeric conversion).
    pub fn read_u16_values(&self, bytes: &[u8]) -> Result<Vec<u16>> {
        if !matches!(self.dtype, DType::F16 | DType::BF16) {
            return Err(ParserError::UnsupportedFormat {
                path: self.name.clone(),
                reason: format!("read_u16_values called on dtype {:?}", self.dtype),
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
        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
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
    f32::from_bits(sign | f16_payload_bits(exp, mant))
}

fn f16_payload_bits(exp: u32, mant: u32) -> u32 {
    match exp {
        0 => mant,
        31 => 0x7F800000 | mant,
        biased => ((biased + 127 - 15) << 23) | mant,
    }
}

// ---------------------------------------------------------------------------
// Unit tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_round_trips_through_ggml_type() {
        let variants = [
            DType::F32,
            DType::F16,
            DType::Q4_0,
            DType::Q4_1,
            DType::Q5_0,
            DType::Q5_1,
            DType::Q8_0,
            DType::Q8_1,
            DType::Q2_K,
            DType::Q3_K,
            DType::Q4_K,
            DType::Q5_K,
            DType::Q6_K,
            DType::Q8_K,
            DType::IQ3_S,
            DType::IQ3_M,
            DType::BF16,
            DType::F64,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
        ];
        for dt in variants {
            let code = dt.ggml_type();
            let back = DType::from_ggml_type(code);
            assert_eq!(dt, back, "round-trip failed for {dt:?} (code={code})");
        }
    }

    #[test]
    fn unknown_code_becomes_other() {
        let dt = DType::from_ggml_type(9999);
        assert_eq!(dt, DType::Other(9999));
        assert_eq!(dt.ggml_type(), 9999);
    }

    #[test]
    fn ggml_type_label_known_codes() {
        assert_eq!(ggml_type_label(GGML_TYPE_F32), "F32");
        assert_eq!(ggml_type_label(GGML_TYPE_F16), "F16");
        assert_eq!(ggml_type_label(GGML_TYPE_Q4_0), "Q4_0");
        assert_eq!(ggml_type_label(GGML_TYPE_Q8_0), "Q8_0");
        assert_eq!(ggml_type_label(GGML_TYPE_Q2_K), "Q2_K");
        assert_eq!(ggml_type_label(GGML_TYPE_Q6_K), "Q6_K");
        assert_eq!(ggml_type_label(GGML_TYPE_IQ3_S), "IQ3_S");
        assert_eq!(ggml_type_label(GGML_TYPE_IQ3_M), "IQ3_M");
        assert_eq!(ggml_type_label(GGML_TYPE_BF16), "BF16");
        assert_eq!(ggml_type_label(GGML_TYPE_F64), "F64");
        assert_eq!(ggml_type_label(GGML_TYPE_I8), "I8");
        assert_eq!(ggml_type_label(GGML_TYPE_I16), "I16");
        assert_eq!(ggml_type_label(GGML_TYPE_I32), "I32");
        assert_eq!(ggml_type_label(GGML_TYPE_I64), "I64");
        assert_eq!(ggml_type_label(GGML_TYPE_IQ1_M), "IQ1_M");
        assert_eq!(ggml_type_label(GGML_TYPE_IQ1_S), "IQ1_S");
        assert_eq!(ggml_type_label(GGML_TYPE_IQ2_XXS), "IQ2_XXS");
        assert_eq!(ggml_type_label(GGML_TYPE_IQ2_XS), "IQ2_XS");
        assert_eq!(ggml_type_label(GGML_TYPE_IQ2_S), "IQ2_S");
        assert_eq!(ggml_type_label(GGML_TYPE_IQ3_XXS), "IQ3_XXS");
        assert_eq!(ggml_type_label(GGML_TYPE_IQ4_NL), "IQ4_NL");
        assert_eq!(ggml_type_label(GGML_TYPE_IQ4_XS), "IQ4_XS");
        assert_eq!(ggml_type_label(GGML_TYPE_Q4_1), "Q4_1");
        assert_eq!(ggml_type_label(GGML_TYPE_Q5_0), "Q5_0");
        assert_eq!(ggml_type_label(GGML_TYPE_Q5_1), "Q5_1");
        assert_eq!(ggml_type_label(GGML_TYPE_Q8_1), "Q8_1");
        assert_eq!(ggml_type_label(GGML_TYPE_Q3_K), "Q3_K");
        assert_eq!(ggml_type_label(GGML_TYPE_Q4_K), "Q4_K");
        assert_eq!(ggml_type_label(GGML_TYPE_Q5_K), "Q5_K");
        assert_eq!(ggml_type_label(GGML_TYPE_Q8_K), "Q8_K");
    }

    #[test]
    fn ggml_type_label_unknown() {
        assert_eq!(ggml_type_label(9999), "unknown");
        assert_eq!(ggml_type_label(u32::MAX), "unknown");
    }

    #[test]
    fn dtype_label_matches_ggml_type_label() {
        let variants = [
            DType::F32,
            DType::F16,
            DType::Q4_0,
            DType::Q8_0,
            DType::IQ3_S,
            DType::IQ3_M,
            DType::BF16,
            DType::F64,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
        ];
        for dt in variants {
            assert_eq!(
                dt.label(),
                ggml_type_label(dt.ggml_type()),
                "label mismatch for {dt:?}"
            );
        }
    }

    #[test]
    fn dtype_label_other_delegates() {
        let dt = DType::Other(9999);
        assert_eq!(dt.label(), "unknown");

        // An Other wrapping a known code should return the known label.
        let dt2 = DType::Other(GGML_TYPE_IQ4_NL);
        assert_eq!(dt2.label(), "IQ4_NL");
    }

    #[test]
    fn byte_len_for_simple_types() {
        assert_eq!(DType::F32.byte_len_for_elements(100), Some(400));
        assert_eq!(DType::F16.byte_len_for_elements(100), Some(200));
        assert_eq!(DType::BF16.byte_len_for_elements(100), Some(200));
        assert_eq!(DType::F64.byte_len_for_elements(10), Some(80));
        assert_eq!(DType::I8.byte_len_for_elements(10), Some(10));
        assert_eq!(DType::I16.byte_len_for_elements(10), Some(20));
        assert_eq!(DType::I32.byte_len_for_elements(10), Some(40));
        assert_eq!(DType::I64.byte_len_for_elements(10), Some(80));
    }

    #[test]
    fn byte_len_for_blocked_quants() {
        // Q4_0: block_size=32, 18 bytes per block
        assert_eq!(DType::Q4_0.byte_len_for_elements(32), Some(18));
        assert_eq!(DType::Q4_0.byte_len_for_elements(64), Some(36));
        assert_eq!(DType::Q4_0.byte_len_for_elements(33), None);

        // Q8_0: block_size=32, 34 bytes per block
        assert_eq!(DType::Q8_0.byte_len_for_elements(32), Some(34));

        // Q4_K: block_size=256, 144 bytes per block
        assert_eq!(DType::Q4_K.byte_len_for_elements(256), Some(144));
        assert_eq!(DType::Q4_K.byte_len_for_elements(128), None);

        // Q6_K: block_size=256, 210 bytes per block
        assert_eq!(DType::Q6_K.byte_len_for_elements(256), Some(210));

        // Q8_K: block_size=256, 292 bytes per block
        assert_eq!(DType::Q8_K.byte_len_for_elements(256), Some(292));
    }

    #[test]
    fn byte_len_for_iq_quants() {
        // IQ3_S: block_size=256, 50 bytes per block
        assert_eq!(DType::IQ3_S.byte_len_for_elements(256), Some(50));
        assert_eq!(DType::IQ3_S.byte_len_for_elements(512), Some(100));
        assert_eq!(DType::IQ3_S.byte_len_for_elements(100), None);
        // IQ3_M: block_size=256, 111 bytes per block
        assert_eq!(DType::IQ3_M.byte_len_for_elements(256), Some(111));
        assert_eq!(DType::IQ3_M.byte_len_for_elements(512), Some(222));
        assert_eq!(DType::IQ3_M.byte_len_for_elements(100), None);
        assert_eq!(DType::Other(99).byte_len_for_elements(100), None);
    }

    #[test]
    fn has_known_byte_layout_check() {
        assert!(DType::F32.has_known_byte_layout());
        assert!(DType::Q8_0.has_known_byte_layout());
        assert!(DType::Q4_K.has_known_byte_layout());
        assert!(DType::IQ3_S.has_known_byte_layout());
        assert!(DType::IQ3_M.has_known_byte_layout());
        assert!(!DType::Other(99).has_known_byte_layout());
    }

    #[test]
    fn ggml_type_constants_match_values() {
        assert_eq!(GGML_TYPE_F32, 0);
        assert_eq!(GGML_TYPE_F16, 1);
        assert_eq!(GGML_TYPE_Q4_0, 2);
        assert_eq!(GGML_TYPE_Q4_1, 3);
        assert_eq!(GGML_TYPE_Q5_0, 6);
        assert_eq!(GGML_TYPE_Q5_1, 7);
        assert_eq!(GGML_TYPE_Q8_0, 8);
        assert_eq!(GGML_TYPE_Q8_1, 9);
        assert_eq!(GGML_TYPE_Q2_K, 10);
        assert_eq!(GGML_TYPE_Q3_K, 11);
        assert_eq!(GGML_TYPE_Q4_K, 12);
        assert_eq!(GGML_TYPE_Q5_K, 13);
        assert_eq!(GGML_TYPE_Q6_K, 14);
        assert_eq!(GGML_TYPE_Q8_K, 15);
        assert_eq!(GGML_TYPE_IQ2_XXS, 16);
        assert_eq!(GGML_TYPE_IQ2_XS, 17);
        assert_eq!(GGML_TYPE_IQ3_XXS, 18);
        assert_eq!(GGML_TYPE_IQ1_S, 19);
        assert_eq!(GGML_TYPE_IQ4_NL, 20);
        assert_eq!(GGML_TYPE_IQ3_S, 21);
        assert_eq!(GGML_TYPE_IQ2_S, 22);
        assert_eq!(GGML_TYPE_IQ4_XS, 23);
        assert_eq!(GGML_TYPE_I8, 24);
        assert_eq!(GGML_TYPE_I16, 25);
        assert_eq!(GGML_TYPE_I32, 26);
        assert_eq!(GGML_TYPE_I64, 27);
        assert_eq!(GGML_TYPE_F64, 28);
        assert_eq!(GGML_TYPE_IQ1_M, 29);
        assert_eq!(GGML_TYPE_BF16, 30);
        assert_eq!(GGML_TYPE_IQ3_M, 31);
    }

    #[test]
    fn f16_to_f32_known_values() {
        // 0.0
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        // 1.0
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        // -1.0
        assert_eq!(f16_bits_to_f32(0xBC00), -1.0);
        // +inf
        assert!(f16_bits_to_f32(0x7C00).is_infinite());
        // -inf
        assert!(f16_bits_to_f32(0xFC00).is_infinite());
    }
}

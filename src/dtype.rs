//! Dtype definitions and `f16 -> f32` casting logic.
//!
//! Only dense float formats are decoded here. Quantized GGML formats
//! (`Q4_0`, `Q6_K`, …) are surfaced as [`GgmlType::Other`] so callers
//! can detect and skip them without this crate pretending to dequantize.

use crate::error::{ParserError, Result};

/// Subset of GGML tensor dtypes that this parser understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    /// IEEE 754 binary32, stored little-endian.
    F32,
    /// IEEE 754 binary16, stored little-endian.
    F16,
    /// A GGML dtype we do not decode (e.g. block-quantized formats).
    Other(u32),
}

impl GgmlType {
    /// GGML raw tag for F32.
    pub const TAG_F32: u32 = 0;
    /// GGML raw tag for F16.
    pub const TAG_F16: u32 = 1;

    /// Decode the on-disk GGML tag.
    pub fn from_raw(tag: u32) -> Self {
        match tag {
            Self::TAG_F32 => Self::F32,
            Self::TAG_F16 => Self::F16,
            other => Self::Other(other),
        }
    }

    /// Raw on-disk GGML tag.
    pub fn raw(self) -> u32 {
        match self {
            Self::F32 => Self::TAG_F32,
            Self::F16 => Self::TAG_F16,
            Self::Other(v) => v,
        }
    }

    /// Size of a single element in bytes, or `None` for unsupported dtypes.
    pub fn element_size(self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 => Some(2),
            Self::Other(_) => None,
        }
    }

    /// `true` for dense float dtypes this crate can cast to `f32`.
    pub fn is_dense_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16)
    }
}

/// Convert a single IEEE 754 binary16 bit pattern into binary32.
///
/// Handles subnormals and the infinity/NaN exponent. Branch-light,
/// no allocation, suitable for inner loops.
#[inline(always)]
pub fn f16_to_f32(bits: u16) -> f32 {
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

/// Cast a little-endian `f16` byte slice into an owned `Vec<f32>`.
///
/// Trailing odd byte (if any) is silently dropped; GGUF tensors are
/// always aligned so this tail is never hit for valid inputs.
pub fn cast_f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|pair| f16_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
        .collect()
}

/// Copy a little-endian `f32` byte slice into an owned `Vec<f32>`.
pub fn cast_f32_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

/// Cast any dense-dtype byte slice into an owned `Vec<f32>` voltage vector.
///
/// Returns [`ParserError::UnsupportedDtype`] for non-dense dtypes.
pub fn cast_bytes_to_f32(bytes: &[u8], dtype: GgmlType, tensor_name: &str) -> Result<Vec<f32>> {
    match dtype {
        GgmlType::F32 => Ok(cast_f32_bytes_to_f32(bytes)),
        GgmlType::F16 => Ok(cast_f16_bytes_to_f32(bytes)),
        GgmlType::Other(tag) => Err(ParserError::UnsupportedDtype {
            name: tensor_name.to_owned(),
            ggml_type: tag,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_known_values() {
        // 0.0 -> 0.0
        assert_eq!(f16_to_f32(0x0000), 0.0_f32);
        // -0.0 -> -0.0
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0_f32).to_bits());
        // 1.0 -> 1.0
        assert_eq!(f16_to_f32(0x3C00), 1.0_f32);
        // -2.0 -> -2.0
        assert_eq!(f16_to_f32(0xC000), -2.0_f32);
        // Smallest positive normal (2^-14).
        assert_eq!(f16_to_f32(0x0400), 2.0_f32.powi(-14));
    }

    #[test]
    fn f16_handles_infinity_and_nan() {
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0xFC00).is_infinite() && f16_to_f32(0xFC00) < 0.0);
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn cast_bytes_f16_matches_scalar() {
        let bytes = [0x00, 0x3C, 0x00, 0xC0]; // 1.0, -2.0
        let cast = cast_f16_bytes_to_f32(&bytes);
        assert_eq!(cast, vec![1.0_f32, -2.0_f32]);
    }

    #[test]
    fn cast_bytes_f32_preserves_value() {
        let bytes = 3.5_f32.to_le_bytes();
        let cast = cast_f32_bytes_to_f32(&bytes);
        assert_eq!(cast, vec![3.5_f32]);
    }

    #[test]
    fn unsupported_dtype_errors() {
        let err = cast_bytes_to_f32(&[0u8; 4], GgmlType::Other(14), "q4_k").unwrap_err();
        match err {
            ParserError::UnsupportedDtype { name, ggml_type } => {
                assert_eq!(name, "q4_k");
                assert_eq!(ggml_type, 14);
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}

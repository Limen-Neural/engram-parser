// SPDX-License-Identifier: MIT OR Apache-2.0

//! Streaming cursor over the raw GGUF byte stream.
//!
//! Only what the parser needs: little-endian scalar reads, length-prefixed
//! strings, and the ability to skip (or stringify) unknown KV values.

use crate::error::{ParserError, Result};

pub(crate) const GGUF_MAGIC: [u8; 4] = *b"GGUF";
pub(crate) const GGUF_VERSION: u32 = 3;

pub(crate) fn unsupported(path: &str, reason: impl Into<String>) -> ParserError {
    ParserError::UnsupportedFormat {
        path: path.to_owned(),
        reason: reason.into(),
    }
}

pub(crate) fn invalid_layout(path: &str, reason: impl Into<String>) -> ParserError {
    ParserError::InvalidLayout {
        path: path.to_owned(),
        reason: reason.into(),
    }
}

pub const GGUF_VALUE_TYPE_UINT8: u32 = 0;
pub const GGUF_VALUE_TYPE_INT8: u32 = 1;
pub const GGUF_VALUE_TYPE_UINT16: u32 = 2;
pub const GGUF_VALUE_TYPE_INT16: u32 = 3;
pub const GGUF_VALUE_TYPE_UINT32: u32 = 4;
pub const GGUF_VALUE_TYPE_INT32: u32 = 5;
pub const GGUF_VALUE_TYPE_FLOAT32: u32 = 6;
pub const GGUF_VALUE_TYPE_BOOL: u32 = 7;
pub const GGUF_VALUE_TYPE_STRING: u32 = 8;
pub const GGUF_VALUE_TYPE_ARRAY: u32 = 9;
pub const GGUF_VALUE_TYPE_UINT64: u32 = 10;
pub const GGUF_VALUE_TYPE_INT64: u32 = 11;
pub const GGUF_VALUE_TYPE_FLOAT64: u32 = 12;

pub(crate) struct GgufCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
    path: &'a str,
}

impl<'a> GgufCursor<'a> {
    pub(crate) fn new(bytes: &'a [u8], path: &'a str) -> Self {
        Self {
            bytes,
            offset: 0,
            path,
        }
    }

    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    fn unsupported(&self, reason: String) -> ParserError {
        ParserError::UnsupportedFormat {
            path: self.path.to_owned(),
            reason,
        }
    }

    pub(crate) fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| self.unsupported("cursor overflow".into()))?;
        if end > self.bytes.len() {
            return Err(self.unsupported("unexpected EOF while parsing GGUF".into()));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    pub(crate) fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_exact(1)?[0])
    }

    pub(crate) fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    pub(crate) fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_exact(4)?;
        let arr: [u8; 4] = bytes
            .try_into()
            .map_err(|_| self.unsupported("expected 4-byte u32 payload".into()))?;
        Ok(u32::from_le_bytes(arr))
    }

    pub(crate) fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_exact(8)?;
        let arr: [u8; 8] = bytes
            .try_into()
            .map_err(|_| self.unsupported("expected 8-byte u64 payload".into()))?;
        Ok(u64::from_le_bytes(arr))
    }

    pub(crate) fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    pub(crate) fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    pub(crate) fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    pub(crate) fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    pub(crate) fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    pub(crate) fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_exact(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| self.unsupported(format!("invalid UTF-8 in GGUF string: {e}")))
    }

    /// Read a numeric-typed GGUF value and coerce it to `u64`.
    pub(crate) fn read_numeric_as_u64(&mut self, value_type: u32) -> Result<u64> {
        match value_type {
            GGUF_VALUE_TYPE_UINT8 => self.read_u8_as_u64(),
            GGUF_VALUE_TYPE_INT8 => self.read_i8_as_u64(),
            GGUF_VALUE_TYPE_UINT16 => self.read_u16_as_u64(),
            GGUF_VALUE_TYPE_INT16 => self.read_i16_as_u64(),
            GGUF_VALUE_TYPE_UINT32 => self.read_u32_as_u64(),
            GGUF_VALUE_TYPE_INT32 => self.read_i32_as_u64(),
            GGUF_VALUE_TYPE_UINT64 => self.read_u64(),
            GGUF_VALUE_TYPE_INT64 => self.read_i64_as_u64(),
            GGUF_VALUE_TYPE_BOOL => self.read_u8_as_u64(),
            other => {
                Err(self.unsupported(format!("expected numeric GGUF value, got type {other}")))
            }
        }
    }

    fn read_u8_as_u64(&mut self) -> Result<u64> {
        Ok(self.read_u8()? as u64)
    }

    fn read_i8_as_u64(&mut self) -> Result<u64> {
        Ok(self.read_u8()? as i8 as i64 as u64)
    }

    fn read_u16_as_u64(&mut self) -> Result<u64> {
        Ok(self.read_u16()? as u64)
    }

    fn read_i16_as_u64(&mut self) -> Result<u64> {
        Ok(self.read_i16()? as i64 as u64)
    }

    fn read_u32_as_u64(&mut self) -> Result<u64> {
        Ok(self.read_u32()? as u64)
    }

    fn read_i32_as_u64(&mut self) -> Result<u64> {
        Ok(self.read_i32()? as i64 as u64)
    }

    fn read_i64_as_u64(&mut self) -> Result<u64> {
        Ok(self.read_i64()? as u64)
    }

    /// Read a numeric-typed GGUF value and coerce it to `usize`.
    pub(crate) fn read_numeric_as_usize(&mut self, value_type: u32) -> Result<usize> {
        Ok(self.read_numeric_as_u64(value_type)? as usize)
    }

    /// Render a scalar GGUF value as a string (used for metadata KV).
    #[allow(dead_code)]
    pub(crate) fn read_scalar_as_string(&mut self, value_type: u32) -> Result<String> {
        match value_type {
            GGUF_VALUE_TYPE_UINT8 | GGUF_VALUE_TYPE_INT8 | GGUF_VALUE_TYPE_UINT16 | GGUF_VALUE_TYPE_INT16 | GGUF_VALUE_TYPE_UINT32 | GGUF_VALUE_TYPE_INT32 | GGUF_VALUE_TYPE_UINT64 | GGUF_VALUE_TYPE_INT64 | GGUF_VALUE_TYPE_BOOL => {
                Ok(self.read_numeric_as_u64(value_type)?.to_string())
            }
            GGUF_VALUE_TYPE_FLOAT32 => Ok(self.read_f32()?.to_string()),
            GGUF_VALUE_TYPE_FLOAT64 => Ok(self.read_f64()?.to_string()),
            GGUF_VALUE_TYPE_STRING => self.read_string(),
            other => Err(self.unsupported(format!("unexpected scalar GGUF value type {other}"))),
        }
    }

    /// Skip an arbitrary GGUF value without materialising it.
    pub(crate) fn skip_value(&mut self, value_type: u32) -> Result<()> {
        match value_type {
            GGUF_VALUE_TYPE_UINT8 | GGUF_VALUE_TYPE_INT8 | GGUF_VALUE_TYPE_BOOL => {
                self.read_exact(1)?;
            }
            GGUF_VALUE_TYPE_UINT16 | GGUF_VALUE_TYPE_INT16 => {
                self.read_exact(2)?;
            }
            GGUF_VALUE_TYPE_UINT32 | GGUF_VALUE_TYPE_INT32 | GGUF_VALUE_TYPE_FLOAT32 => {
                self.read_exact(4)?;
            }
            GGUF_VALUE_TYPE_UINT64 | GGUF_VALUE_TYPE_INT64 | GGUF_VALUE_TYPE_FLOAT64 => {
                self.read_exact(8)?;
            }
            GGUF_VALUE_TYPE_STRING => {
                let _ = self.read_string()?;
            }
            GGUF_VALUE_TYPE_ARRAY => self.skip_array_value()?,
            other => {
                return Err(self.unsupported(format!("unsupported GGUF value type {other}")));
            }
        }
        Ok(())
    }

    fn skip_array_value(&mut self) -> Result<()> {
        let nested = self.read_u32()?;
        let len = self.read_u64()? as usize;
        for _ in 0..len {
            self.skip_value(nested)?;
        }
        Ok(())
    }
}

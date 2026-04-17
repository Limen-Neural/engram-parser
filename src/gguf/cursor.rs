//! Streaming cursor over the raw GGUF byte stream.
//!
//! Only what the parser needs: little-endian scalar reads, length-prefixed
//! strings, and the ability to skip (or stringify) unknown KV values.

use crate::error::{ParserError, Result};

pub(crate) const GGUF_MAGIC: [u8; 4] = [b'G', b'G', b'U', b'F'];
pub(crate) const GGUF_VERSION: u32 = 3;

pub(crate) const VT_U8: u32 = 0;
pub(crate) const VT_I8: u32 = 1;
pub(crate) const VT_U16: u32 = 2;
pub(crate) const VT_I16: u32 = 3;
pub(crate) const VT_U32: u32 = 4;
pub(crate) const VT_I32: u32 = 5;
pub(crate) const VT_F32: u32 = 6;
pub(crate) const VT_BOOL: u32 = 7;
pub(crate) const VT_STRING: u32 = 8;
pub(crate) const VT_ARRAY: u32 = 9;
pub(crate) const VT_U64: u32 = 10;
pub(crate) const VT_I64: u32 = 11;
pub(crate) const VT_F64: u32 = 12;

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
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("slice length is 4"),
        ))
    }

    pub(crate) fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("slice length is 8"),
        ))
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
            VT_U8 => Ok(self.read_u8()? as u64),
            VT_I8 => Ok(self.read_u8()? as i8 as i64 as u64),
            VT_U16 => Ok(self.read_u16()? as u64),
            VT_I16 => Ok(self.read_i16()? as i64 as u64),
            VT_U32 => Ok(self.read_u32()? as u64),
            VT_I32 => Ok(self.read_i32()? as i64 as u64),
            VT_U64 => self.read_u64(),
            VT_I64 => Ok(self.read_i64()? as u64),
            VT_BOOL => Ok(self.read_u8()? as u64),
            other => Err(self.unsupported(format!(
                "expected numeric GGUF value, got type {other}"
            ))),
        }
    }

    /// Read a numeric-typed GGUF value and coerce it to `usize`.
    pub(crate) fn read_numeric_as_usize(&mut self, value_type: u32) -> Result<usize> {
        Ok(self.read_numeric_as_u64(value_type)? as usize)
    }

    /// Render a scalar GGUF value as a string (used for metadata KV).
    pub(crate) fn read_scalar_as_string(&mut self, value_type: u32) -> Result<String> {
        match value_type {
            VT_U8 | VT_I8 | VT_U16 | VT_I16 | VT_U32 | VT_I32 | VT_U64 | VT_I64 | VT_BOOL => {
                Ok(self.read_numeric_as_u64(value_type)?.to_string())
            }
            VT_F32 => Ok(self.read_f32()?.to_string()),
            VT_F64 => Ok(self.read_f64()?.to_string()),
            VT_STRING => self.read_string(),
            other => Err(self.unsupported(format!("unexpected scalar GGUF value type {other}"))),
        }
    }

    /// Skip an arbitrary GGUF value without materialising it.
    pub(crate) fn skip_value(&mut self, value_type: u32) -> Result<()> {
        match value_type {
            VT_U8 | VT_I8 | VT_BOOL => {
                self.read_exact(1)?;
            }
            VT_U16 | VT_I16 => {
                self.read_exact(2)?;
            }
            VT_U32 | VT_I32 | VT_F32 => {
                self.read_exact(4)?;
            }
            VT_U64 | VT_I64 | VT_F64 => {
                self.read_exact(8)?;
            }
            VT_STRING => {
                let _ = self.read_string()?;
            }
            VT_ARRAY => {
                let nested = self.read_u32()?;
                let len = self.read_u64()? as usize;
                for _ in 0..len {
                    self.skip_value(nested)?;
                }
            }
            other => {
                return Err(self.unsupported(format!("unsupported GGUF value type {other}")));
            }
        }
        Ok(())
    }
}

//! Flat engram voltage vector.

use crate::dtype::cast_bytes_to_f32;
use crate::error::Result;
use crate::gguf::GgufTensor;

/// A raw voltage vector (`Vec<f32>`) plus the source tensor shape.
///
/// Engram vectors are the downstream SNN contract: every element is a
/// cleanly cast `f32` voltage, regardless of whether the static weight
/// tensor on disk was stored as `F16` or `F32`.
#[derive(Debug, Clone)]
pub struct EngramVector {
    /// Canonical tensor name this vector was derived from.
    pub name: String,
    /// Original tensor shape (GGUF order — `dims[0]` is innermost).
    pub dims: Vec<usize>,
    /// Raw voltages, row-major following `dims`.
    pub voltages: Vec<f32>,
}

impl EngramVector {
    /// Cast a parser tensor view into an engram voltage vector.
    pub fn from_tensor(tensor: &GgufTensor<'_>) -> Result<Self> {
        let voltages = cast_bytes_to_f32(tensor.bytes(), tensor.dtype(), tensor.name())?;
        Ok(Self {
            name: tensor.name().to_owned(),
            dims: tensor.dims().to_vec(),
            voltages,
        })
    }

    pub fn len(&self) -> usize {
        self.voltages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.voltages.is_empty()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.voltages
    }

    pub fn into_inner(self) -> Vec<f32> {
        self.voltages
    }
}

//! Two-dimensional engram voltage matrix.

use crate::dtype::cast_bytes_to_f32;
use crate::error::{ParserError, Result};
use crate::gguf::GgufTensor;

/// A 2-D voltage matrix derived from a GGUF weight tensor.
///
/// GGUF stores `dims[0]` as the fastest-varying (innermost) axis. We
/// interpret that as the matrix's **column** stride, i.e. a row of the
/// matrix is a contiguous slab of `cols` voltages.
#[derive(Debug, Clone)]
pub struct EngramMatrix {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub voltages: Vec<f32>,
}

impl EngramMatrix {
    /// Cast a rank-2 parser tensor view into an engram matrix.
    pub fn from_tensor(tensor: &GgufTensor<'_>) -> Result<Self> {
        if tensor.dims().len() != 2 {
            return Err(ParserError::ShapeMismatch {
                name: tensor.name().to_owned(),
                dims: tensor.dims().to_vec(),
                op: "EngramMatrix::from_tensor (expected rank-2)",
            });
        }
        let cols = tensor.dims()[0];
        let rows = tensor.dims()[1];
        let voltages = cast_bytes_to_f32(tensor.bytes(), tensor.dtype(), tensor.name())?;
        Ok(Self {
            name: tensor.name().to_owned(),
            rows,
            cols,
            voltages,
        })
    }

    /// Build an engram matrix from an explicit voltage slab plus `rows`/`cols`.
    ///
    /// Used when slicing a stacked expert tensor into a per-expert matrix.
    pub fn from_parts(
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        voltages: Vec<f32>,
    ) -> Result<Self> {
        let name = name.into();
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| ParserError::ShapeMismatch {
                name: name.clone(),
                dims: vec![cols, rows],
                op: "EngramMatrix::from_parts (overflow)",
            })?;
        if expected != voltages.len() {
            return Err(ParserError::ShapeMismatch {
                name,
                dims: vec![cols, rows],
                op: "EngramMatrix::from_parts (slab length mismatch)",
            });
        }
        Ok(Self {
            name,
            rows,
            cols,
            voltages,
        })
    }

    /// Borrow a single row of the matrix.
    pub fn row(&self, idx: usize) -> Option<&[f32]> {
        if idx >= self.rows {
            return None;
        }
        let start = idx * self.cols;
        Some(&self.voltages[start..start + self.cols])
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.voltages
    }

    pub fn len(&self) -> usize {
        self.voltages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.voltages.is_empty()
    }
}

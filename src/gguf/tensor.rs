//! Tensor metadata and mapped-byte views.

use crate::dtype::GgmlType;

/// On-disk layout of a single GGUF tensor.
///
/// `dims[0]` is the fastest-varying (innermost) axis per GGUF convention.
/// `absolute_offset` is the byte offset into the mapped file where the
/// tensor data begins.
#[derive(Debug, Clone)]
pub struct TensorLayout {
    pub name: String,
    pub dims: Vec<usize>,
    pub ggml_type: GgmlType,
    pub absolute_offset: usize,
    pub n_elements: usize,
}

impl TensorLayout {
    /// Total size of the tensor in bytes, or `None` for unsupported dtypes.
    pub fn byte_len(&self) -> Option<usize> {
        self.ggml_type.element_size().map(|sz| sz * self.n_elements)
    }

    /// Rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

/// Borrowed view over the mmapped bytes of a single tensor.
#[derive(Debug)]
pub struct GgufTensor<'a> {
    layout: &'a TensorLayout,
    bytes: &'a [u8],
}

impl<'a> GgufTensor<'a> {
    pub(crate) fn new(layout: &'a TensorLayout, bytes: &'a [u8]) -> Self {
        Self { layout, bytes }
    }

    pub fn layout(&self) -> &TensorLayout {
        self.layout
    }

    pub fn name(&self) -> &str {
        &self.layout.name
    }

    pub fn dims(&self) -> &[usize] {
        &self.layout.dims
    }

    pub fn dtype(&self) -> GgmlType {
        self.layout.ggml_type
    }

    pub fn bytes(&self) -> &[u8] {
        self.bytes
    }

    pub fn n_elements(&self) -> usize {
        self.layout.n_elements
    }
}

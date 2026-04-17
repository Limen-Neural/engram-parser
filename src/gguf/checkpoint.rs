//! High-level mmapped GGUF checkpoint.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapOptions};

use crate::dtype::cast_bytes_to_f32;
use crate::error::{ParserError, Result};
use crate::gguf::layout::{parse_gguf_layout, CheckpointMetadata, ParsedLayout};
use crate::gguf::tensor::{GgufTensor, TensorLayout};

/// Memory-mapped GGUF checkpoint ready for parser queries.
///
/// The mmap stays alive for the lifetime of the checkpoint; all
/// [`GgufTensor`] views borrow directly from it without copying.
#[derive(Debug)]
pub struct GgufCheckpoint {
    path: PathBuf,
    mmap: Mmap,
    tensors: HashMap<String, TensorLayout>,
    metadata: CheckpointMetadata,
}

impl GgufCheckpoint {
    /// Open a GGUF file, parse its header, and keep it memory-mapped read-only.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let path_str = path.to_string_lossy().to_string();

        let file = File::open(&path).map_err(|source| ParserError::Io {
            path: path_str.clone(),
            source,
        })?;

        let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|source| ParserError::Io {
            path: path_str.clone(),
            source,
        })?;

        let ParsedLayout { tensors, metadata } = parse_gguf_layout(&mmap, &path_str)?;

        Ok(Self {
            path,
            mmap,
            tensors,
            metadata,
        })
    }

    /// Filesystem path this checkpoint was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Parsed KV metadata.
    pub fn metadata(&self) -> &CheckpointMetadata {
        &self.metadata
    }

    /// Iterate over all tensor names present in the checkpoint.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Check whether a tensor with the given canonical name exists.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Look up a tensor's on-disk layout metadata.
    pub fn tensor_layout(&self, name: &str) -> Result<&TensorLayout> {
        self.tensors
            .get(name)
            .ok_or_else(|| ParserError::MissingTensor {
                name: name.to_owned(),
                path: self.path_str(),
            })
    }

    /// Borrow a single tensor's mmapped bytes along with its layout.
    pub fn tensor(&self, name: &str) -> Result<GgufTensor<'_>> {
        let layout = self.tensor_layout(name)?;
        let elem_size = layout
            .ggml_type
            .element_size()
            .ok_or_else(|| ParserError::UnsupportedDtype {
                name: layout.name.clone(),
                ggml_type: layout.ggml_type.raw(),
            })?;

        let end = layout.absolute_offset + layout.n_elements * elem_size;
        if end > self.mmap.len() {
            return Err(ParserError::UnsupportedFormat {
                path: self.path_str(),
                reason: format!("tensor '{name}' extends beyond mapped file"),
            });
        }

        Ok(GgufTensor::new(
            layout,
            &self.mmap[layout.absolute_offset..end],
        ))
    }

    /// Cast any dense (`F32`/`F16`) tensor to a flat `Vec<f32>` voltage vector.
    pub fn extract_as_f32(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self.tensor(name)?;
        cast_bytes_to_f32(tensor.bytes(), tensor.dtype(), tensor.name())
    }

    /// Enumerate tensors whose canonical name starts with `prefix`, useful
    /// for listing every tensor that belongs to a single transformer block.
    pub fn tensors_with_prefix<'a>(
        &'a self,
        prefix: &'a str,
    ) -> impl Iterator<Item = &'a TensorLayout> + 'a {
        self.tensors
            .values()
            .filter(move |t| t.name.starts_with(prefix))
    }

    fn path_str(&self) -> String {
        self.path.to_string_lossy().to_string()
    }
}

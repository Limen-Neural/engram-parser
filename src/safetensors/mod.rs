// SPDX-License-Identifier: MIT OR Apache-2.0

//! Safetensors format parser and MoE candidate discovery.
//!
//! Safetensors is a simple format for storing tensors:
//! - 8 bytes: little-endian u64 header length
//! - N bytes: JSON header with tensor metadata
//! - remaining: tensor data (aligned)
//!
//! This module provides:
//! - Header parsing with zero external dependencies
//! - Manifest generation for single-file and multi-shard checkpoints
//! - MoE router/expert candidate discovery
//! - Layout family detection

mod discovery;
mod header;
mod json;
mod manifest;

use crate::error::{ParserError, Result};
use header::{SafetensorsHeader, parse_header};
use manifest::{
    SafetensorsCheckpointSource,
    SafetensorsManifest, SafetensorsTensorRecord,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub use discovery::{classify_tensor, discover_candidates};
pub use manifest::{
    SafetensorsCandidateSummary as CandidateSummary,
    SafetensorsCheckpointSource as CheckpointSource,
    SafetensorsExpertGroup as ExpertGroup,
    SafetensorsManifest as Manifest,
    SafetensorsRouterCandidate as RouterCandidate,
    SafetensorsTensorRecord as TensorRecord,
};

/// Inspect a Safetensors checkpoint and generate a manifest.
///
/// Supports:
/// - Single-file checkpoints: 
/// - Multi-shard indexed checkpoints:  + shards
/// - Directory scanning: finds all  files
pub fn inspect_safetensors_checkpoint(path: impl AsRef<Path>) -> Result<SafetensorsManifest> {
    let path = path.as_ref();
    
    // Check if path exists
    if !path.exists() {
        return Err(ParserError::Io {
            path: path.display().to_string(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "path not found"),
        });
    }

    // Determine input type
    if path.is_dir() {
        inspect_directory(path)
    } else if is_index_file(path) {
        inspect_indexed_checkpoint(path)
    } else if is_safetensors_file(path) {
        inspect_single_file(path)
    } else {
        Err(ParserError::UnsupportedFormat {
            path: path.display().to_string(),
            reason: "expected .safetensors file, .safetensors.index.json, or directory".into(),
        })
    }
}

/// Check if path is a Safetensors index file.
fn is_index_file(path: &Path) -> bool {
    path.to_str()
        .map(|s| s.ends_with(".safetensors.index.json"))
        .unwrap_or(false)
}

/// Check if path is a Safetensors file.
fn is_safetensors_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext == "safetensors")
        .unwrap_or(false)
}

/// Inspect a single-file checkpoint.
fn inspect_single_file(path: &Path) -> Result<SafetensorsManifest> {
    let bytes = fs::read(path).map_err(|e| ParserError::Io {
        path: path.display().to_string(),
        source: e,
    })?;

    let (header, _data_offset) = parse_header(&bytes, &path.display().to_string())?;
    let shard_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.safetensors");

    let tensors = build_tensor_records(&header, shard_name);
    let candidates = discover_candidates(&tensors);

    Ok(SafetensorsManifest {
        manifest_version: 1,
        format: "safetensors",
        checkpoint: SafetensorsCheckpointSource {
            input_kind: "single_file".into(),
            index_file: None,
            shard_count: 1,
            tensor_count: tensors.len(),
            metadata: header.metadata,
        },
        tensors,
        candidates,
    })
}

/// Inspect a directory of Safetensors files.
fn inspect_directory(dir: &Path) -> Result<SafetensorsManifest> {
    // Check for index file
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        return inspect_indexed_checkpoint(&index_path);
    }

    // Scan for .safetensors files
    let mut shard_paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| ParserError::Io {
            path: dir.display().to_string(),
            source: e,
        })?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| is_safetensors_file(path))
        .collect();

    shard_paths.sort();

    if shard_paths.is_empty() {
        return Err(ParserError::UnsupportedFormat {
            path: dir.display().to_string(),
            reason: "no .safetensors files found in directory".into(),
        });
    }

    inspect_multi_shard(&shard_paths, "directory", None)
}

/// Inspect an indexed checkpoint.
fn inspect_indexed_checkpoint(index_path: &Path) -> Result<SafetensorsManifest> {
    // Parse index file
    let index_content = fs::read_to_string(index_path).map_err(|e| ParserError::Io {
        path: index_path.display().to_string(),
        source: e,
    })?;

    let index_json = json::parse_json(&index_content, &index_path.display().to_string())?;
    let index_obj = index_json.as_object().ok_or_else(|| ParserError::InvalidLayout {
        path: index_path.display().to_string(),
        reason: "index file must be a JSON object".into(),
    })?;

    // Extract weight map
    let weight_map = index_obj
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| ParserError::InvalidLayout {
            path: index_path.display().to_string(),
            reason: "index file missing 'weight_map' object".into(),
        })?;

    // Group tensors by shard
    let mut shard_to_tensors: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (tensor_name, shard_value) in weight_map {
        let shard_name = shard_value.as_str().ok_or_else(|| ParserError::InvalidLayout {
            path: index_path.display().to_string(),
            reason: format!("weight_map value for '{}' must be a string", tensor_name),
        })?;
        shard_to_tensors
            .entry(shard_name.to_string())
            .or_default()
            .push(tensor_name.clone());
    }

    // Load shards
    let mut shard_paths: Vec<PathBuf> = shard_to_tensors
        .keys()
        .map(|shard_name| index_path.parent().unwrap_or(Path::new(".")).join(shard_name))
        .collect();
    shard_paths.sort();

    let index_filename = index_path
        .file_name()
        .and_then(|n| n.to_str())
        .map(String::from);

    inspect_multi_shard(&shard_paths, "hf_index", index_filename)
}

/// Inspect multiple shards.
fn inspect_multi_shard(
    shard_paths: &[PathBuf],
    input_kind: &str,
    index_file: Option<String>,
) -> Result<SafetensorsManifest> {
    let mut all_tensors = Vec::new();
    let mut merged_metadata = BTreeMap::new();

    for shard_path in shard_paths {
        let bytes = fs::read(shard_path).map_err(|e| ParserError::Io {
            path: shard_path.display().to_string(),
            source: e,
        })?;

        let (header, _data_offset) = parse_header(&bytes, &shard_path.display().to_string())?;
        let shard_name = shard_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.safetensors");

        // Merge metadata (first shard wins for conflicts)
        for (key, value) in &header.metadata {
            merged_metadata.entry(key.clone()).or_insert_with(|| value.clone());
        }

        let tensors = build_tensor_records(&header, shard_name);
        all_tensors.extend(tensors);
    }

    let candidates = discover_candidates(&all_tensors);

    Ok(SafetensorsManifest {
        manifest_version: 1,
        format: "safetensors",
        checkpoint: SafetensorsCheckpointSource {
            input_kind: input_kind.into(),
            index_file,
            shard_count: shard_paths.len(),
            tensor_count: all_tensors.len(),
            metadata: merged_metadata,
        },
        tensors: all_tensors,
        candidates,
    })
}

/// Build tensor records from a header.
fn build_tensor_records(header: &SafetensorsHeader, shard_name: &str) -> Vec<SafetensorsTensorRecord> {
    let mut records: Vec<SafetensorsTensorRecord> = header
        .tensors
        .iter()
        .map(|(name, desc)| {
            let labels = classify_tensor(name, &desc.shape);
            SafetensorsTensorRecord {
                name: name.clone(),
                dtype: desc.dtype.clone(),
                shape: desc.shape.clone(),
                byte_size: desc.byte_size(),
                source_shard: shard_name.to_string(),
                data_offsets: desc.data_offsets,
                labels,
            }
        })
        .collect();

    // Sort by name for deterministic output
    records.sort_by(|a, b| a.name.cmp(&b.name));
    records
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_safetensors_bytes(header_json: &str, data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header_json.as_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn parses_single_file() {
        let json = r#"{
            "__metadata__": {"format": "pt"},
            "tensor1": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}
        }"#;
        let data = vec![0u8; 24];
        let bytes = build_safetensors_bytes(json, &data);
        
        let dir = std::env::temp_dir().join("engram_test_single");
        fs::create_dir_all(&dir).ok();
        let file_path = dir.join("model.safetensors");
        fs::write(&file_path, &bytes).unwrap();
        
        let manifest = inspect_safetensors_checkpoint(&file_path).unwrap();
        
        assert_eq!(manifest.checkpoint.input_kind, "single_file");
        assert_eq!(manifest.checkpoint.shard_count, 1);
        assert_eq!(manifest.tensors.len(), 1);
        assert_eq!(manifest.tensors[0].name, "tensor1");
        assert_eq!(manifest.tensors[0].dtype, "F32");
        assert_eq!(manifest.tensors[0].shape, vec![2, 3]);
        
        fs::remove_dir_all(&dir).ok();
    }
}

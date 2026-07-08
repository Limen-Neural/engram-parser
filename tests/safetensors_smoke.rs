// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end smoke tests for Safetensors format support.

use engram_parser::safetensors::inspect_safetensors_checkpoint;
use std::fs;
use std::path::PathBuf;

fn build_safetensors_bytes(header_json: &str, data: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header_json.as_bytes());
    bytes.extend_from_slice(data);
    bytes
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("engram_test_{}", name));
        fs::create_dir_all(&path).ok();
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).ok();
    }
}

impl AsRef<std::path::Path> for TempDir {
    fn as_ref(&self) -> &std::path::Path {
        &self.0
    }
}

#[test]
fn parses_single_file_checkpoint() {
    let dir = TempDir::new("single");
    let json = r#"{
        "__metadata__": {"format": "pt", "total_size": "24"},
        "layer1.weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
        "layer2.bias": {"dtype": "F32", "shape": [3], "data_offsets": [24, 36]}
    }"#;
    let data = vec![0u8; 36];
    let bytes = build_safetensors_bytes(json, &data);
    
    let file_path = dir.as_ref().join("model.safetensors");
    fs::write(&file_path, &bytes).unwrap();
    
    let manifest = inspect_safetensors_checkpoint(&file_path).unwrap();
    
    assert_eq!(manifest.format, "safetensors");
    assert_eq!(manifest.checkpoint.input_kind, "single_file");
    assert_eq!(manifest.checkpoint.shard_count, 1);
    assert_eq!(manifest.checkpoint.tensor_count, 2);
    assert_eq!(manifest.checkpoint.metadata.get("format").unwrap(), "pt");
    
    assert_eq!(manifest.tensors.len(), 2);
    assert_eq!(manifest.tensors[0].name, "layer1.weight");
    assert_eq!(manifest.tensors[0].dtype, "F32");
    assert_eq!(manifest.tensors[0].shape, vec![2, 3]);
    assert_eq!(manifest.tensors[0].byte_size, 24);
    
    assert_eq!(manifest.tensors[1].name, "layer2.bias");
    assert_eq!(manifest.tensors[1].shape, vec![3]);
    assert_eq!(manifest.tensors[1].byte_size, 12);
}

#[test]
fn detects_moe_router_candidates() {
    let dir = TempDir::new("moe_router");
    let json = r#"{
        "model.layers.0.mlp.gate.weight": {"dtype": "F32", "shape": [8, 2048], "data_offsets": [0, 65536]},
        "model.layers.0.mlp.experts.0.w1.weight": {"dtype": "F16", "shape": [2048, 4096], "data_offsets": [65536, 16842752]}
    }"#;
    let data = vec![0u8; 16842752];
    let bytes = build_safetensors_bytes(json, &data);
    
    let file_path = dir.as_ref().join("model.safetensors");
    fs::write(&file_path, &bytes).unwrap();
    
    let manifest = inspect_safetensors_checkpoint(&file_path).unwrap();
    
    assert_eq!(manifest.tensors.len(), 2);
    
    // Check router detection
    assert_eq!(manifest.candidates.router_tensors.len(), 1);
    assert_eq!(manifest.candidates.router_tensors[0], "model.layers.0.mlp.gate.weight");
    
    // Check expert detection
    assert_eq!(manifest.candidates.expert_tensors.len(), 1);
    assert_eq!(manifest.candidates.expert_tensors[0], "model.layers.0.mlp.experts.0.w1.weight");
    
    // Check router candidate scoring
    assert_eq!(manifest.candidates.router_candidates.len(), 1);
    let router = &manifest.candidates.router_candidates[0];
    assert_eq!(router.name, "model.layers.0.mlp.gate.weight");
    assert_eq!(router.layer_hint, Some(0));
    assert!(router.score >= 60);
    assert!(!router.reasons.is_empty());
    
    // Check layout family detection
    assert_eq!(manifest.candidates.detected_layout_family, Some("generic_moe"));
}

#[test]
fn groups_expert_tensors() {
    let dir = TempDir::new("expert_groups");
    let json = r#"{
        "model.layers.0.mlp.experts.0.w1.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]},
        "model.layers.0.mlp.experts.0.w2.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [8, 16]},
        "model.layers.0.mlp.experts.1.w1.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [16, 24]},
        "model.layers.0.mlp.experts.1.w2.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [24, 32]}
    }"#;
    let data = vec![0u8; 32];
    let bytes = build_safetensors_bytes(json, &data);
    
    let file_path = dir.as_ref().join("model.safetensors");
    fs::write(&file_path, &bytes).unwrap();
    
    let manifest = inspect_safetensors_checkpoint(&file_path).unwrap();
    
    assert_eq!(manifest.candidates.expert_groups.len(), 1);
    let group = &manifest.candidates.expert_groups[0];
    
    assert_eq!(group.group_key, "model.layers.0.mlp");
    assert_eq!(group.layer_hint, Some(0));
    assert_eq!(group.expert_indices, vec![0, 1]);
    assert_eq!(group.tensor_names.len(), 4);
    assert!(group.weight_kinds.contains(&"w1"));
    assert!(group.weight_kinds.contains(&"w2"));
}

#[test]
fn parses_multi_shard_indexed_checkpoint() {
    let dir = TempDir::new("indexed");
    
    // Create shard 1
    let json1 = r#"{
        "layer1.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}
    }"#;
    let data1 = vec![0u8; 16];
    let bytes1 = build_safetensors_bytes(json1, &data1);
    fs::write(dir.as_ref().join("model-00001-of-00002.safetensors"), &bytes1).unwrap();
    
    // Create shard 2
    let json2 = r#"{
        "layer2.weight": {"dtype": "F32", "shape": [3, 3], "data_offsets": [0, 36]}
    }"#;
    let data2 = vec![0u8; 36];
    let bytes2 = build_safetensors_bytes(json2, &data2);
    fs::write(dir.as_ref().join("model-00002-of-00002.safetensors"), &bytes2).unwrap();
    
    // Create index file
    let index_json = r#"{
        "metadata": {"total_size": "52"},
        "weight_map": {
            "layer1.weight": "model-00001-of-00002.safetensors",
            "layer2.weight": "model-00002-of-00002.safetensors"
        }
    }"#;
    fs::write(dir.as_ref().join("model.safetensors.index.json"), index_json).unwrap();
    
    let manifest = inspect_safetensors_checkpoint(dir.as_ref().join("model.safetensors.index.json")).unwrap();
    
    assert_eq!(manifest.checkpoint.input_kind, "hf_index");
    assert_eq!(manifest.checkpoint.shard_count, 2);
    assert_eq!(manifest.checkpoint.tensor_count, 2);
    assert!(manifest.checkpoint.index_file.is_some());
    
    assert_eq!(manifest.tensors.len(), 2);
    
    // Verify tensors from different shards
    let layer1 = manifest.tensors.iter().find(|t| t.name == "layer1.weight").unwrap();
    let layer2 = manifest.tensors.iter().find(|t| t.name == "layer2.weight").unwrap();
    
    assert_eq!(layer1.source_shard, "model-00001-of-00002.safetensors");
    assert_eq!(layer2.source_shard, "model-00002-of-00002.safetensors");
}

#[test]
fn scans_directory_for_shards() {
    let dir = TempDir::new("directory");
    
    // Create multiple shards
    let json1 = r#"{"tensor1": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}"#;
    let bytes1 = build_safetensors_bytes(json1, &[0u8; 8]);
    fs::write(dir.as_ref().join("shard1.safetensors"), &bytes1).unwrap();
    
    let json2 = r#"{"tensor2": {"dtype": "F32", "shape": [3], "data_offsets": [0, 12]}}"#;
    let bytes2 = build_safetensors_bytes(json2, &[0u8; 12]);
    fs::write(dir.as_ref().join("shard2.safetensors"), &bytes2).unwrap();
    
    let manifest = inspect_safetensors_checkpoint(&dir.0).unwrap();
    
    assert_eq!(manifest.checkpoint.input_kind, "directory");
    assert_eq!(manifest.checkpoint.shard_count, 2);
    assert_eq!(manifest.checkpoint.tensor_count, 2);
}

#[test]
fn rejects_invalid_format() {
    let dir = TempDir::new("invalid");
    let file_path = dir.as_ref().join("invalid.txt");
    fs::write(&file_path, "not a safetensors file").unwrap();
    
    let result = inspect_safetensors_checkpoint(&file_path);
    assert!(result.is_err());
}

#[test]
fn rejects_missing_file() {
    let result = inspect_safetensors_checkpoint("/nonexistent/path/model.safetensors");
    assert!(result.is_err());
}

#[test]
fn handles_empty_metadata() {
    let dir = TempDir::new("no_metadata");
    let json = r#"{"tensor": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}"#;
    let bytes = build_safetensors_bytes(json, &[0u8; 8]);
    
    let file_path = dir.as_ref().join("model.safetensors");
    fs::write(&file_path, &bytes).unwrap();
    
    let manifest = inspect_safetensors_checkpoint(&file_path).unwrap();
    assert!(manifest.checkpoint.metadata.is_empty());
}

#[test]
fn classifies_tensors_correctly() {
    let dir = TempDir::new("classify");
    let json = r#"{
        "model.layers.0.mlp.gate.weight": {"dtype": "F32", "shape": [8, 2048], "data_offsets": [0, 65536]},
        "model.layers.0.mlp.experts.0.w1.weight": {"dtype": "F16", "shape": [2048, 4096], "data_offsets": [65536, 16842752]},
        "embedding.weight": {"dtype": "F32", "shape": [1000, 512], "data_offsets": [16842752, 20480000]}
    }"#;
    let data = vec![0u8; 20480000];
    let bytes = build_safetensors_bytes(json, &data);
    
    let file_path = dir.as_ref().join("model.safetensors");
    fs::write(&file_path, &bytes).unwrap();
    
    let manifest = inspect_safetensors_checkpoint(&file_path).unwrap();
    
    // Find tensors and check their labels
    let gate_tensor = manifest.tensors.iter().find(|t| t.name.contains("gate.weight")).unwrap();
    let expert_tensor = manifest.tensors.iter().find(|t| t.name.contains("experts.0")).unwrap();
    let embedding_tensor = manifest.tensors.iter().find(|t| t.name == "embedding.weight").unwrap();
    
    assert!(gate_tensor.labels.contains(&"moe_router_candidate"));
    assert!(expert_tensor.labels.contains(&"moe_expert_candidate"));
    assert!(embedding_tensor.labels.contains(&"possible_moe_router_shape")); // Shape-based hint only
}

#[test]
fn detects_specialized_layout_families() {
    let dir = TempDir::new("phimoe");
    let json = r#"{
        "phimoe.layers.0.mlp.gate.weight": {"dtype": "F32", "shape": [8, 2048], "data_offsets": [0, 65536]},
        "phimoe.layers.0.mlp.experts.0.w1.weight": {"dtype": "F16", "shape": [2048, 4096], "data_offsets": [65536, 16842752]}
    }"#;
    let data = vec![0u8; 16842752];
    let bytes = build_safetensors_bytes(json, &data);
    
    let file_path = dir.as_ref().join("model.safetensors");
    fs::write(&file_path, &bytes).unwrap();
    
    let manifest = inspect_safetensors_checkpoint(&file_path).unwrap();
    
    assert_eq!(manifest.candidates.detected_layout_family, Some("phimoe"));
}

#[test]
fn handles_deepseek_v3_family() {
    let dir = TempDir::new("deepseek");
    let json = r#"{
        "model.layers.0.block_sparse_moe.gate.weight": {"dtype": "F32", "shape": [8, 2048], "data_offsets": [0, 65536]},
        "model.layers.0.block_sparse_moe.experts.0.w1.weight": {"dtype": "F16", "shape": [2048, 4096], "data_offsets": [65536, 16842752]}
    }"#;
    let data = vec![0u8; 16842752];
    let bytes = build_safetensors_bytes(json, &data);
    
    let file_path = dir.as_ref().join("model.safetensors");
    fs::write(&file_path, &bytes).unwrap();
    
    let manifest = inspect_safetensors_checkpoint(&file_path).unwrap();
    
    assert_eq!(manifest.candidates.detected_layout_family, Some("deepseek_v3_family"));
    
    // Verify the router was detected with high confidence
    let router = &manifest.candidates.router_candidates[0];
    assert!(router.reasons.contains(&"deepseek_v3_family_block_sparse_moe_gate"));
    assert!(router.score >= 90);
}

#[test]
fn sorts_tensors_deterministically() {
    let dir = TempDir::new("sort");
    let json = r#"{
        "z_tensor": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "a_tensor": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]},
        "m_tensor": {"dtype": "F32", "shape": [2], "data_offsets": [16, 24]}
    }"#;
    let bytes = build_safetensors_bytes(json, &[0u8; 24]);
    
    let file_path = dir.as_ref().join("model.safetensors");
    fs::write(&file_path, &bytes).unwrap();
    
    let manifest = inspect_safetensors_checkpoint(&file_path).unwrap();
    
    // Verify tensors are sorted by name
    assert_eq!(manifest.tensors[0].name, "a_tensor");
    assert_eq!(manifest.tensors[1].name, "m_tensor");
    assert_eq!(manifest.tensors[2].name, "z_tensor");
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Safetensors manifest types and generation.
//!
//! A manifest describes the structure of a Safetensors checkpoint:
//! - Source information (single file, indexed, directory)
//! - Tensor records with metadata
//! - MoE candidate discovery results

use std::collections::BTreeMap;

/// Complete manifest for a Safetensors checkpoint.
#[derive(Debug, Clone)]
pub struct SafetensorsManifest {
    /// Manifest format version.
    pub manifest_version: u32,
    /// Format identifier.
    pub format: &'static str,
    /// Checkpoint source information.
    pub checkpoint: SafetensorsCheckpointSource,
    /// Tensor records.
    pub tensors: Vec<SafetensorsTensorRecord>,
    /// MoE candidate discovery results.
    pub candidates: SafetensorsCandidateSummary,
}

/// Describes where the checkpoint was loaded from.
#[derive(Debug, Clone)]
pub struct SafetensorsCheckpointSource {
    /// Type of input: "single_file", "hf_index", or "directory".
    pub input_kind: String,
    /// Index file path (for indexed checkpoints).
    pub index_file: Option<String>,
    /// Number of shards.
    pub shard_count: usize,
    /// Total number of tensors.
    pub tensor_count: usize,
    /// Global metadata.
    pub metadata: BTreeMap<String, String>,
}

/// Record for a single tensor in the checkpoint.
#[derive(Debug, Clone)]
pub struct SafetensorsTensorRecord {
    /// Tensor name.
    pub name: String,
    /// Data type (e.g., "F32", "F16").
    pub dtype: String,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Byte size of tensor data.
    pub byte_size: usize,
    /// Source shard filename.
    pub source_shard: String,
    /// Byte offsets [start, end) in the shard.
    pub data_offsets: [usize; 2],
    /// Classification labels (e.g., "moe_router_candidate").
    pub labels: Vec<&'static str>,
}

/// Summary of MoE candidate discovery.
#[derive(Debug, Clone, Default)]
pub struct SafetensorsCandidateSummary {
    /// Detected layout family (e.g., "generic_moe", "deepseek_v3_family").
    pub detected_layout_family: Option<&'static str>,
    /// Names of router tensors.
    pub router_tensors: Vec<String>,
    /// Names of expert tensors.
    pub expert_tensors: Vec<String>,
    /// Router candidates with scoring.
    pub router_candidates: Vec<SafetensorsRouterCandidate>,
    /// Expert groups.
    pub expert_groups: Vec<SafetensorsExpertGroup>,
}

/// A candidate router tensor with scoring information.
#[derive(Debug, Clone)]
pub struct SafetensorsRouterCandidate {
    /// Tensor name.
    pub name: String,
    /// Layer index hint (if extractable from name).
    pub layer_hint: Option<usize>,
    /// Source shard filename.
    pub source_shard: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Confidence score.
    pub score: u32,
    /// Reasons for classification.
    pub reasons: Vec<&'static str>,
}

/// A group of expert tensors belonging to the same MoE layer.
#[derive(Debug, Clone)]
pub struct SafetensorsExpertGroup {
    /// Group identifier (e.g., "model.layers.0.mlp.experts").
    pub group_key: String,
    /// Layer index hint (if extractable from name).
    pub layer_hint: Option<usize>,
    /// Expert indices found in this group.
    pub expert_indices: Vec<usize>,
    /// Names of tensors in this group.
    pub tensor_names: Vec<String>,
    /// Source shards containing this group.
    pub source_shards: Vec<String>,
    /// Weight kinds (e.g., "gate_proj", "up_proj", "down_proj").
    pub weight_kinds: Vec<&'static str>,
}

//! Per-expert raw weight bundle.

use crate::engram::EngramMatrix;

/// Raw weights for a single MoE expert, already cast into engram voltage matrices.
///
/// Each field is `Option<_>` because GGUF checkpoints differ on which
/// expert projections they store. OLMoE exposes `gate / up / down`;
/// other MoE families may omit or rename a subset.
#[derive(Debug)]
pub struct MoeExpertWeights {
    /// Transformer block this expert belongs to.
    pub block: usize,
    /// Expert index within the block.
    pub expert: usize,
    /// `ffn_gate` projection slice, when present.
    pub gate: Option<EngramMatrix>,
    /// `ffn_up` projection slice, when present.
    pub up: Option<EngramMatrix>,
    /// `ffn_down` projection slice, when present.
    pub down: Option<EngramMatrix>,
}

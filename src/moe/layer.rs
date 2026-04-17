//! MoE block isolation and per-expert slicing.
//!
//! Supports the "stacked expert" layout used by OLMoE / llama.cpp
//! conversions where every expert's projection lives inside one rank-3
//! tensor of shape `[inner, outer, num_experts]` (GGUF dim order, so
//! `inner` is the fastest-varying axis).

use crate::dtype::cast_bytes_to_f32;
use crate::engram::{EngramMatrix, EngramVector};
use crate::error::{ParserError, Result};
use crate::gguf::{GgufCheckpoint, TensorLayout};
use crate::moe::expert::MoeExpertWeights;

/// A single MoE transformer block wired up to its router and expert tensors.
#[derive(Debug)]
pub struct MoeLayer<'a> {
    checkpoint: &'a GgufCheckpoint,
    block: usize,
    router_layout: TensorLayout,
    gate_exps: Option<TensorLayout>,
    up_exps: Option<TensorLayout>,
    down_exps: Option<TensorLayout>,
    num_experts: usize,
}

impl<'a> MoeLayer<'a> {
    /// Locate and isolate MoE tensors for transformer block `block`.
    ///
    /// Requires at minimum the router tensor `blk.{block}.ffn_gate_inp.weight`.
    /// Returns [`ParserError::MissingMoeTensor`] if the router is absent.
    pub fn isolate(checkpoint: &'a GgufCheckpoint, block: usize) -> Result<Self> {
        let router_name = format!("blk.{block}.ffn_gate_inp.weight");
        let router_layout = checkpoint
            .tensor_layout(&router_name)
            .map_err(|_| ParserError::MissingMoeTensor {
                block,
                name: router_name,
            })?
            .clone();

        let gate_exps = checkpoint
            .tensor_layout(&format!("blk.{block}.ffn_gate_exps.weight"))
            .ok()
            .cloned();
        let up_exps = checkpoint
            .tensor_layout(&format!("blk.{block}.ffn_up_exps.weight"))
            .ok()
            .cloned();
        let down_exps = checkpoint
            .tensor_layout(&format!("blk.{block}.ffn_down_exps.weight"))
            .ok()
            .cloned();

        // Deduce num_experts. Stacked-expert tensors carry it as the
        // slowest-varying (last) axis; the router tensor carries it as
        // one of its two axes.
        let num_experts = gate_exps
            .as_ref()
            .or(up_exps.as_ref())
            .or(down_exps.as_ref())
            .and_then(|layout| layout.dims.last().copied())
            .or_else(|| router_layout.dims.last().copied())
            .unwrap_or(0);

        Ok(Self {
            checkpoint,
            block,
            router_layout,
            gate_exps,
            up_exps,
            down_exps,
            num_experts,
        })
    }

    pub fn block(&self) -> usize {
        self.block
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn has_gate_exps(&self) -> bool {
        self.gate_exps.is_some()
    }

    pub fn has_up_exps(&self) -> bool {
        self.up_exps.is_some()
    }

    pub fn has_down_exps(&self) -> bool {
        self.down_exps.is_some()
    }

    /// Name of the router tensor driving this block.
    pub fn router_name(&self) -> &str {
        &self.router_layout.name
    }

    /// Extract the full router matrix (`blk.N.ffn_gate_inp.weight`).
    pub fn router_matrix(&self) -> Result<EngramMatrix> {
        let tensor = self.checkpoint.tensor(&self.router_layout.name)?;
        EngramMatrix::from_tensor(&tensor)
    }

    /// Extract the router as a flat voltage vector, regardless of rank.
    pub fn router_vector(&self) -> Result<EngramVector> {
        let tensor = self.checkpoint.tensor(&self.router_layout.name)?;
        EngramVector::from_tensor(&tensor)
    }

    /// Extract raw weights for a single expert from the stacked expert tensors.
    pub fn expert(&self, expert_idx: usize) -> Result<MoeExpertWeights> {
        if self.num_experts == 0 || expert_idx >= self.num_experts {
            return Err(ParserError::ExpertIndexOutOfRange {
                index: expert_idx,
                num_experts: self.num_experts,
            });
        }

        let gate = self.slice_expert_matrix(self.gate_exps.as_ref(), expert_idx, "gate")?;
        let up = self.slice_expert_matrix(self.up_exps.as_ref(), expert_idx, "up")?;
        let down = self.slice_expert_matrix(self.down_exps.as_ref(), expert_idx, "down")?;

        Ok(MoeExpertWeights {
            block: self.block,
            expert: expert_idx,
            gate,
            up,
            down,
        })
    }

    fn slice_expert_matrix(
        &self,
        layout: Option<&TensorLayout>,
        expert_idx: usize,
        role: &'static str,
    ) -> Result<Option<EngramMatrix>> {
        let Some(layout) = layout else {
            return Ok(None);
        };

        if layout.dims.len() != 3 {
            return Err(ParserError::ShapeMismatch {
                name: layout.name.clone(),
                dims: layout.dims.clone(),
                op: "MoeLayer::expert (expected rank-3 stacked-expert tensor)",
            });
        }

        let cols = layout.dims[0];
        let rows = layout.dims[1];
        let num_experts_t = layout.dims[2];
        if expert_idx >= num_experts_t {
            return Err(ParserError::ExpertIndexOutOfRange {
                index: expert_idx,
                num_experts: num_experts_t,
            });
        }

        let elem_size =
            layout
                .ggml_type
                .element_size()
                .ok_or_else(|| ParserError::UnsupportedDtype {
                    name: layout.name.clone(),
                    ggml_type: layout.ggml_type.raw(),
                })?;
        let per_expert_elems =
            rows.checked_mul(cols)
                .ok_or_else(|| ParserError::ShapeMismatch {
                    name: layout.name.clone(),
                    dims: layout.dims.clone(),
                    op: "MoeLayer::expert (per-expert element overflow)",
                })?;
        let per_expert_bytes = per_expert_elems * elem_size;

        let full_tensor = self.checkpoint.tensor(&layout.name)?;
        let start = expert_idx * per_expert_bytes;
        let end = start + per_expert_bytes;
        let bytes = full_tensor
            .bytes()
            .get(start..end)
            .ok_or_else(|| ParserError::ShapeMismatch {
                name: layout.name.clone(),
                dims: layout.dims.clone(),
                op: "MoeLayer::expert (byte range out of bounds)",
            })?;

        let voltages = cast_bytes_to_f32(bytes, layout.ggml_type, &layout.name)?;
        let slice_name = format!("{}[expert={expert_idx}|{role}]", layout.name);
        Ok(Some(EngramMatrix::from_parts(
            slice_name, rows, cols, voltages,
        )?))
    }
}

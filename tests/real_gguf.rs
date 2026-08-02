// SPDX-License-Identifier: MIT OR Apache-2.0

//! Path-gated real GGUF pilots (xai-dissect style).
//!
//! CI never runs these: they are `#[ignore]` and need multi-GB weights on disk.
//! Locally, point at a file or a tree under `~/.models/gguf`:
//!
//! ```bash
//! ENGRAM_GGUF=~/.models/gguf/.../model.gguf \
//!   cargo test --test real_gguf -- --ignored --nocapture
//!
//! ENGRAM_MODEL_DIR=~/.models/gguf \
//!   ENGRAM_GGUF_MAX=3 \
//!   cargo test --test real_gguf -- --ignored --nocapture
//!
//! # Hard-fail if no MoE experts are discovered; sample multiple expert pairs:
//! ENGRAM_GGUF=~/.models/gguf/.../moe.gguf \
//!   ENGRAM_EXPECT_MOE=1 ENGRAM_MOE_SAMPLES=3 \
//!   cargo test --test real_gguf real_gguf_moe -- --ignored --nocapture
//! ```
//!
//! GPU / kernel experiments on the same weights belong in
//! `~/rmems/blackwell-kernel-lab` (or myelin-accelerator), not this crate.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use engram_parser::{DType, extract_expert, list_experts, load_gguf};

const ENV_GGUF: &str = "ENGRAM_GGUF";
const ENV_MODEL_DIR: &str = "ENGRAM_MODEL_DIR";
const ENV_MAX: &str = "ENGRAM_GGUF_MAX";

/// When set to `1`/`true`/`yes`, MoE extract must find at least one expert
/// pair and a successful `extract_expert` (hard fail on dense / unknown names).
const ENV_EXPECT_MOE: &str = "ENGRAM_EXPECT_MOE";

/// How many (block, expert) pairs to extract when MoE is present (default 1).
const ENV_MOE_SAMPLES: &str = "ENGRAM_MOE_SAMPLES";

fn expect_moe() -> bool {
    match env::var(ENV_EXPECT_MOE) {
        Ok(v) => matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => false,
    }
}

fn moe_sample_count() -> usize {
    env::var(ENV_MOE_SAMPLES)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1)
}

/// Resolve pilot paths the way xai-dissect resolves checkpoint pilots:
/// explicit file, else scan a directory for `*.gguf` (non-recursive by default
/// depth-limited walk so huge trees stay controllable).
fn pilot_gguf_paths() -> Vec<PathBuf> {
    if let Ok(single) = env::var(ENV_GGUF) {
        let p = PathBuf::from(single);
        return if p.is_file() { vec![p] } else { Vec::new() };
    }

    let Ok(root) = env::var(ENV_MODEL_DIR) else {
        return Vec::new();
    };
    let root = PathBuf::from(root);
    if !root.is_dir() {
        return Vec::new();
    }

    let max = env::var(ENV_MAX)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8);

    let mut out = Vec::new();
    collect_gguf(&root, 0, 6, max, &mut out);
    out.sort();
    out
}

fn collect_gguf(
    dir: &Path,
    depth: usize,
    max_depth: usize,
    max_files: usize,
    out: &mut Vec<PathBuf>,
) {
    if out.len() >= max_files || depth > max_depth {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut dirs = Vec::new();
    for entry in entries.flatten() {
        if out.len() >= max_files {
            break;
        }
        let path = entry.path();
        if path.is_file() {
            if path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
            {
                out.push(path);
            }
        } else if path.is_dir() {
            dirs.push(path);
        }
    }
    dirs.sort();
    for d in dirs {
        collect_gguf(&d, depth + 1, max_depth, max_files, out);
        if out.len() >= max_files {
            break;
        }
    }
}

fn require_pilots() -> Vec<PathBuf> {
    let paths = pilot_gguf_paths();
    assert!(
        !paths.is_empty(),
        "no pilot GGUFs found — set {ENV_GGUF}=/path/to/model.gguf \
         or {ENV_MODEL_DIR}=~/.models/gguf (optional {ENV_MAX}=N)"
    );
    for p in &paths {
        assert!(p.is_file(), "not a file: {}", p.display());
    }
    paths
}

#[test]
#[ignore = "pilot: set ENGRAM_GGUF or ENGRAM_MODEL_DIR; not run in CI"]
fn real_gguf_parse_inventory() {
    let paths = require_pilots();
    for path in paths {
        let t0 = Instant::now();
        let layout = load_gguf(&path).unwrap_or_else(|e| {
            panic!("load_gguf({}) failed: {e}", path.display());
        });
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        assert!(
            !layout.tensors.is_empty(),
            "{}: expected tensors",
            path.display()
        );
        assert!(layout.alignment >= 1, "alignment");

        // Every directory entry must have a consistent byte_len for known dtypes.
        for (name, tensor) in &layout.tensors {
            assert!(
                tensor.byte_len > 0 || tensor.n_elements == 0,
                "{name}: zero byte_len with n_elements={}",
                tensor.n_elements
            );
            if let Some(expected) = tensor.dtype.byte_len_for_elements(tensor.n_elements) {
                assert_eq!(
                    tensor.byte_len, expected,
                    "{name}: byte_len mismatch for {:?}",
                    tensor.dtype
                );
            }
            // Payload must be in-range.
            let bytes = layout.tensor_bytes(tensor).unwrap_or_else(|e| {
                panic!("{name}: tensor_bytes: {e}");
            });
            assert_eq!(bytes.len(), tensor.byte_len, "{name}: payload len");
        }

        eprintln!(
            "OK inventory {}  tensors={} arch={} quant={} parse_ms={ms:.1}",
            path.display(),
            layout.tensors.len(),
            layout.metadata.architecture(),
            layout.metadata.quantization(),
        );
    }
}

#[test]
#[ignore = "pilot: set ENGRAM_GGUF or ENGRAM_MODEL_DIR; not run in CI"]
fn real_gguf_moe_extract_when_present() {
    let paths = require_pilots();
    let hard = expect_moe();
    let samples = moe_sample_count();
    let mut any_moe = false;

    for path in paths {
        let t0 = Instant::now();
        let layout = load_gguf(&path).expect("load");
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let experts = list_experts(&layout);
        eprintln!(
            "moe_scan {}  arch={} quant={} expert_meta={:?} pairs={} load_ms={load_ms:.1}",
            path.display(),
            layout.metadata.architecture(),
            layout.metadata.quantization(),
            layout.metadata.expert_count(),
            experts.len(),
        );

        if experts.is_empty() {
            eprintln!("skip MoE (none discovered): {}", path.display());
            continue;
        }
        any_moe = true;

        let take = samples.min(experts.len());
        for &(b, e) in experts.iter().take(take) {
            let w = extract_expert(&layout, b, e).unwrap_or_else(|err| {
                panic!("extract_expert({}, {b}, {e}): {err}", path.display());
            });

            assert!(
                w.gate.is_some() || w.up.is_some() || w.down.is_some(),
                "{}: empty extract for ({b},{e})",
                path.display()
            );

            for (role, opt) in [
                ("gate", w.gate.as_ref()),
                ("up", w.up.as_ref()),
                ("down", w.down.as_ref()),
            ] {
                if let Some(t) = opt {
                    assert!(!t.bytes.is_empty(), "{role} empty bytes");
                    assert!(!t.dims.is_empty(), "{role} empty dims");
                    if matches!(t.dtype, DType::F16 | DType::BF16) {
                        assert_eq!(t.bytes.len(), t.dims.iter().product::<usize>() * 2);
                    }
                    if t.dtype == DType::F32 {
                        assert_eq!(t.bytes.len(), t.dims.iter().product::<usize>() * 4);
                    }
                }
            }

            eprintln!(
                "OK moe {}  pair=({b},{e}) complete={} stacked_gate={}",
                path.display(),
                w.is_complete(),
                w.gate.as_ref().map(|g| g.stacked_slice).unwrap_or(false),
            );
        }

        if let Some(n) = layout.metadata.expert_count() {
            assert!(n > 0, "{}: expert_count metadata is 0", path.display());
        }
    }

    if hard {
        assert!(
            any_moe,
            "ENGRAM_EXPECT_MOE set but no MoE expert tensors found in pilot set"
        );
    } else if !any_moe {
        eprintln!(
            "note: no MoE expert tensors in pilot set; inventory-only is fine for dense GGUFs"
        );
    }
}

#[test]
fn real_gguf_helpers_document_env() {
    // Always runs in CI: documents the pilot contract without touching disk weights.
    assert_eq!(ENV_GGUF, "ENGRAM_GGUF");
    assert_eq!(ENV_MODEL_DIR, "ENGRAM_MODEL_DIR");
    assert_eq!(ENV_MAX, "ENGRAM_GGUF_MAX");
    assert_eq!(ENV_EXPECT_MOE, "ENGRAM_EXPECT_MOE");
    assert_eq!(ENV_MOE_SAMPLES, "ENGRAM_MOE_SAMPLES");
    // Defaults only when those vars are unset (local pilot env must not break CI-safe test).
    if env::var_os(ENV_EXPECT_MOE).is_none() {
        assert!(!expect_moe());
    }
    if env::var_os(ENV_MOE_SAMPLES).is_none() {
        assert_eq!(moe_sample_count(), 1);
    }
    // With no path env, pilot list is empty (CI safe).
    if env::var_os(ENV_GGUF).is_none() && env::var_os(ENV_MODEL_DIR).is_none() {
        assert!(pilot_gguf_paths().is_empty());
    }
}

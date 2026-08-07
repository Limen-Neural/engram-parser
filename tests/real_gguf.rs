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
use std::process;
use std::time::Instant;

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

use engram_parser::{GgufLayout, MoeExpertWeights, extract_expert, list_experts, load_gguf};

/// Assert that an extracted expert has at least one role tensor and that each
/// role tensor has a consistent payload size for its declared dtype.
fn assert_expert_weights_valid(path: &Path, b: usize, e: usize, w: &MoeExpertWeights) {
    assert!(
        w.gate.is_some() || w.up.is_some() || w.down.is_some(),
        "{}: empty extract for ({b},{e})",
        path.display()
    );

    for (role, opt) in [("gate", &w.gate), ("up", &w.up), ("down", &w.down)] {
        let Some(t) = opt else { continue };
        assert!(!t.bytes.is_empty(), "{role} empty bytes");
        assert!(!t.dims.is_empty(), "{role} empty dims");

        let n_elements = t.dims.iter().product::<usize>();
        if let Some(expected) = t.dtype.byte_len_for_elements(n_elements) {
            assert_eq!(
                t.bytes.len(),
                expected,
                "{role} byte length for {} ({b},{e})",
                path.display()
            );
        }
    }
}

/// Resolve pilot paths the way xai-dissect resolves checkpoint pilots:
/// explicit file, else scan a directory for `*.gguf` recursively, limited by
/// the configured depth so huge trees stay controllable.
fn load_and_scan(path: &Path) -> (GgufLayout, Vec<(usize, usize)>) {
    let t0 = Instant::now();
    let layout = load_gguf(path).expect("load");
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

    (layout, experts)
}

fn extract_and_report(layout: &GgufLayout, path: &Path, b: usize, e: usize) {
    let w = extract_expert(layout, b, e).unwrap_or_else(|err| {
        panic!("extract_expert({}, {b}, {e}): {err}", path.display());
    });

    assert_expert_weights_valid(path, b, e, &w);

    eprintln!(
        "OK moe {}  pair=({b},{e}) complete={} stacked_gate={}",
        path.display(),
        w.is_complete(),
        w.gate.as_ref().map(|g| g.stacked_slice).unwrap_or(false),
    );
}

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

    // Collect the full candidate set first, then sort and cap — `read_dir`
    // order is unspecified, so capping during walk is non-reproducible.
    let mut out = Vec::new();
    collect_gguf(&root, 0, 6, &mut out);
    out.sort();
    out.truncate(max);
    out
}

fn collect_gguf(dir: &Path, depth: usize, max_depth: usize, out: &mut Vec<PathBuf>) {
    if depth > max_depth {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut dirs = Vec::new();
    for entry in entries.flatten() {
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
        collect_gguf(&d, depth + 1, max_depth, out);
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
        let (layout, experts) = load_and_scan(&path);

        if experts.is_empty() {
            eprintln!("skip MoE (none discovered): {}", path.display());
            continue;
        }
        any_moe = true;

        let take = samples.min(experts.len());
        for &(b, e) in experts.iter().take(take) {
            extract_and_report(&layout, &path, b, e);
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
    // Always runs in CI: documents the pilot contract and exercises the
    // env-dependent helpers in a controlled, single-threaded test context.
    assert_eq!(ENV_GGUF, "ENGRAM_GGUF");
    assert_eq!(ENV_MODEL_DIR, "ENGRAM_MODEL_DIR");
    assert_eq!(ENV_MAX, "ENGRAM_GGUF_MAX");
    assert_eq!(ENV_EXPECT_MOE, "ENGRAM_EXPECT_MOE");
    assert_eq!(ENV_MOE_SAMPLES, "ENGRAM_MOE_SAMPLES");

    let prev_moe = env::var_os(ENV_MOE_SAMPLES);
    let prev_gguf = env::var_os(ENV_GGUF);

    // SAFETY: env mutation is isolated to this single-threaded test process.
    unsafe {
        env::set_var(ENV_MOE_SAMPLES, "7");
    }
    assert_eq!(moe_sample_count(), 7);
    unsafe {
        env::set_var(ENV_MOE_SAMPLES, "0");
    }
    assert_eq!(moe_sample_count(), 1); // clamped to 1
    match prev_moe {
        Some(v) => unsafe { env::set_var(ENV_MOE_SAMPLES, v) },
        None => unsafe { env::remove_var(ENV_MOE_SAMPLES) },
    }

    // pilot_gguf_paths resolves a single ENGRAM_GGUF file.
    let tmp = env::temp_dir().join(format!("engram_helpers_test_{}.gguf", process::id()));
    fs::File::create(&tmp).expect("create temp file");
    unsafe {
        env::set_var(ENV_GGUF, &tmp);
    }
    assert_eq!(pilot_gguf_paths(), vec![tmp.clone()]);
    match prev_gguf {
        Some(v) => unsafe { env::set_var(ENV_GGUF, v) },
        None => unsafe { env::remove_var(ENV_GGUF) },
    }
    let _ = fs::remove_file(&tmp);
}

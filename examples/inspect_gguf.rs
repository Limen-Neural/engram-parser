// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inventory a real on-disk GGUF (xai-dissect-style pilot path).
//!
//! # Usage
//!
//! ```bash
//! cargo run --example inspect_gguf -- /path/to/model.gguf
//! ENGRAM_GGUF=~/.models/gguf/foo.gguf cargo run --example inspect_gguf
//! ```
//!
//! CPU-only. No CUDA, no dequant, no generation. For GPU experiments on the
//! same weights, use `~/rmems/blackwell-kernel-lab` (or myelin-accelerator
//! kernels), not this crate.

use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use engram_parser::{GgufLayout, extract_expert, ggml_type_label, list_experts, load_gguf};

fn main() -> ExitCode {
    let path = match resolve_path() {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("{msg}");
            eprintln!("usage: cargo run --example inspect_gguf -- <model.gguf>");
            eprintln!("   or: ENGRAM_GGUF=<model.gguf> cargo run --example inspect_gguf");
            return ExitCode::from(2);
        }
    };

    if !path.is_file() {
        eprintln!("not a file: {}", path.display());
        return ExitCode::from(1);
    }

    let t0 = Instant::now();
    let layout = match load_gguf(&path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("load_gguf failed: {e}");
            return ExitCode::from(1);
        }
    };
    // Includes full-file read + parse (not parse-only).
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

    print_inventory(&layout, &path, load_ms);
    print_dtype_histogram(&layout);
    print_moe_summary(&layout);
    print_tensor_sample(&layout);

    ExitCode::SUCCESS
}

fn print_inventory(layout: &GgufLayout, path: &Path, load_ms: f64) {
    println!("path:          {}", path.display());
    println!("load_ms:       {load_ms:.2}");
    println!("architecture:  {}", layout.metadata.architecture());
    println!("quantization:  {}", layout.metadata.quantization());
    println!("alignment:     {}", layout.alignment);
    println!("tensor_count:  {}", layout.tensors.len());
    println!("block_count:   {:?}", layout.metadata.block_count());
    println!("expert_count:  {:?}", layout.metadata.expert_count());
    println!("expert_used:   {:?}", layout.metadata.expert_used_count());
    println!("embed_len:     {:?}", layout.metadata.embedding_length());
}

fn print_dtype_histogram(layout: &GgufLayout) {
    let mut counts: Vec<(String, usize)> = {
        let mut m: HashMap<String, usize> = HashMap::new();
        for t in layout.tensors.values() {
            *m.entry(ggml_type_label(t.ggml_type).to_owned())
                .or_default() += 1;
        }
        let mut v: Vec<_> = m.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v
    };
    if counts.len() > 16 {
        counts.truncate(16);
    }
    println!("dtype_hist:    {counts:?}");
}

fn print_moe_summary(layout: &GgufLayout) {
    let experts = list_experts(layout);
    println!("moe_pairs:     {} (block,expert)", experts.len());
    if experts.is_empty() {
        return;
    }

    let show = experts.len().min(8);
    println!("moe_pairs_hd:  {:?}", &experts[..show]);

    let (b, e) = experts[0];
    let t1 = Instant::now();
    match extract_expert(layout, b, e) {
        Ok(w) => {
            let extract_ms = t1.elapsed().as_secs_f64() * 1000.0;
            println!(
                "extract ({b},{e}): complete={} extract_ms={extract_ms:.2}",
                w.is_complete()
            );
            if let Some(g) = w.gate.as_ref() {
                println!(
                    "  gate: dims={:?} bytes={} dtype={:?} stacked={}",
                    g.dims,
                    g.bytes.len(),
                    g.dtype,
                    g.stacked_slice
                );
            }
            if let Some(u) = w.up.as_ref() {
                println!(
                    "  up:   dims={:?} bytes={} dtype={:?}",
                    u.dims,
                    u.bytes.len(),
                    u.dtype
                );
            }
            if let Some(d) = w.down.as_ref() {
                println!(
                    "  down: dims={:?} bytes={} dtype={:?}",
                    d.dims,
                    d.bytes.len(),
                    d.dtype
                );
            }
        }
        Err(err) => println!("extract ({b},{e}) failed: {err}"),
    }
}

fn print_tensor_sample(layout: &GgufLayout) {
    let mut names: Vec<_> = layout.tensors.keys().cloned().collect();
    names.sort();
    let n = names.len().min(12);
    println!("tensor_names_hd ({n}/{}):", names.len());
    for name in &names[..n] {
        let t = &layout.tensors[name];
        println!(
            "  {name}: dims={:?} type={} byte_len={}",
            t.dims,
            ggml_type_label(t.ggml_type),
            t.byte_len
        );
    }
}

fn resolve_path() -> Result<PathBuf, String> {
    // nosemgrep: argv is used only for CLI dispatch, never as a security trust anchor.
    let mut args = env::args_os().skip(1);
    if let Some(p) = args.next() {
        if let Some(s) = p.to_str() {
            if s == "--help" || s == "-h" || s == "--version" || s == "-V" {
                return Err("help requested".into());
            }
            if s.starts_with('-') {
                return Err(format!("unknown option {s}"));
            }
        }
        return Ok(PathBuf::from(p));
    }
    env::var("ENGRAM_GGUF")
        .map(PathBuf::from)
        .map_err(|_| "missing model path (arg or ENGRAM_GGUF)".into())
}

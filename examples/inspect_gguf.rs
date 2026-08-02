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

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use engram_parser::{extract_expert, ggml_type_label, list_experts, load_gguf};

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
    let parse_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("path:          {}", path.display());
    println!("parse_ms:      {parse_ms:.2}");
    println!("architecture:  {}", layout.metadata.architecture());
    println!("quantization:  {}", layout.metadata.quantization());
    println!("alignment:     {}", layout.alignment);
    println!("tensor_count:  {}", layout.tensors.len());
    println!("block_count:   {:?}", layout.metadata.block_count());
    println!("expert_count:  {:?}", layout.metadata.expert_count());
    println!("expert_used:   {:?}", layout.metadata.expert_used_count());
    println!("embed_len:     {:?}", layout.metadata.embedding_length());

    // Dtype histogram (first 16 labels by frequency).
    let mut counts: Vec<(String, usize)> = {
        use std::collections::HashMap;
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

    let experts = list_experts(&layout);
    println!("moe_pairs:     {} (block,expert)", experts.len());
    if !experts.is_empty() {
        let show = experts.len().min(8);
        println!("moe_pairs_hd:  {:?}", &experts[..show]);

        let (b, e) = experts[0];
        let t1 = Instant::now();
        match extract_expert(&layout, b, e) {
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

    // Sample a few tensor names (sorted) for inventory smoke.
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

    ExitCode::SUCCESS
}

fn resolve_path() -> Result<PathBuf, String> {
    let mut args = env::args().skip(1);
    if let Some(p) = args.next() {
        return Ok(PathBuf::from(p));
    }
    env::var("ENGRAM_GGUF")
        .map(PathBuf::from)
        .map_err(|_| "missing model path (arg or ENGRAM_GGUF)".into())
}

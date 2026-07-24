// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end smoke test: build a synthetic GGUF in memory, parse it,
//! and verify that expert extraction round-trips both the stacked and
//! per-expert storage conventions.

use engram_parser::{DType, extract_expert, list_experts, parse_bytes};

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: u32 = 32;

// Value types.
const VT_UINT32: u32 = 4;
const VT_STRING: u32 = 8;

// Dtypes (GGUF wire type ids).
const GGML_F32: u32 = 0;
const GGML_Q8_0: u32 = 8;
const GGML_Q4_K: u32 = 12;
const GGML_IQ3_S: u32 = 21;

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn push_string(out: &mut Vec<u8>, s: &str) {
    push_u64(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}
fn push_kv_u32(out: &mut Vec<u8>, key: &str, v: u32) {
    push_string(out, key);
    push_u32(out, VT_UINT32);
    push_u32(out, v);
}
fn push_kv_string(out: &mut Vec<u8>, key: &str, v: &str) {
    push_string(out, key);
    push_u32(out, VT_STRING);
    push_string(out, v);
}

struct TensorSpec {
    name: &'static str,
    dims: Vec<usize>,
    ggml_type: u32,
    payload: Vec<u8>,
}

fn build_gguf(kv: &[(&str, KvValue)], tensors: &[TensorSpec]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC);
    push_u32(&mut out, GGUF_VERSION);
    push_u64(&mut out, tensors.len() as u64);
    push_u64(&mut out, kv.len() as u64);

    for (key, value) in kv {
        match value {
            KvValue::U32(v) => push_kv_u32(&mut out, key, *v),
            KvValue::Str(v) => push_kv_string(&mut out, key, v),
            KvValue::F32(v) => {
                push_string(&mut out, key);
                push_u32(&mut out, 6); // VT_F32
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    // First pass: tensor directory with relative offsets.
    let mut cum: usize = 0;
    for spec in tensors {
        push_string(&mut out, spec.name);
        push_u32(&mut out, spec.dims.len() as u32);
        for &d in &spec.dims {
            push_u64(&mut out, d as u64);
        }
        push_u32(&mut out, spec.ggml_type);
        push_u64(&mut out, cum as u64);
        cum += spec.payload.len();
    }

    // Align then write payloads.
    while out.len() % ALIGNMENT as usize != 0 {
        out.push(0);
    }
    for spec in tensors {
        out.extend_from_slice(&spec.payload);
    }
    out
}

#[allow(dead_code)]
enum KvValue {
    U32(u32),
    Str(&'static str),
    F32(f32),
}

fn f32_vec_to_le_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[test]
fn parses_magic_and_metadata() {
    let kv = [
        ("general.alignment", KvValue::U32(ALIGNMENT)),
        ("general.architecture", KvValue::Str("olmoe")),
        ("olmoe.expert_count", KvValue::U32(4)),
        ("olmoe.block_count", KvValue::U32(1)),
    ];
    let tensors = [TensorSpec {
        name: "token_embd.weight",
        dims: vec![4, 2],
        ggml_type: GGML_F32,
        payload: f32_vec_to_le_bytes(&[0.0; 4 * 2]),
    }];
    let bytes = build_gguf(&kv, &tensors);
    let layout = parse_bytes(bytes, "mem://test".into()).expect("parse");
    assert_eq!(layout.metadata.architecture(), "olmoe");
    assert_eq!(layout.metadata.numeric("olmoe.expert_count"), Some(4));
    assert_eq!(layout.alignment, ALIGNMENT as usize);
    assert!(layout.tensors.contains_key("token_embd.weight"));
    let t = &layout.tensors["token_embd.weight"];
    assert_eq!(t.dtype, DType::F32);
    assert_eq!(t.dims, vec![4, 2]);
}

#[test]
fn extracts_stacked_expert_slices() {
    // 3 experts, each a 4x2 matrix of f32.
    // Expert e fills its slice with the value (e + 1).
    let inner = 4usize;
    let outer = 2usize;
    let n_experts = 3usize;
    let per_expert = inner * outer;

    let mut gate = vec![0.0f32; n_experts * per_expert];
    let mut up = vec![0.0f32; n_experts * per_expert];
    let mut down = vec![0.0f32; n_experts * per_expert];
    for e in 0..n_experts {
        let base = e * per_expert;
        for slot in base..base + per_expert {
            gate[slot] = (e as f32) + 0.1;
            up[slot] = (e as f32) + 0.2;
            down[slot] = (e as f32) + 0.3;
        }
    }

    // GGML dims are innermost-first, so the expert axis is the last dim.
    let stacked_dims = vec![inner, outer, n_experts];

    let tensors = [
        TensorSpec {
            name: "blk.0.ffn_gate_exps.weight",
            dims: stacked_dims.clone(),
            ggml_type: GGML_F32,
            payload: f32_vec_to_le_bytes(&gate),
        },
        TensorSpec {
            name: "blk.0.ffn_up_exps.weight",
            dims: stacked_dims.clone(),
            ggml_type: GGML_F32,
            payload: f32_vec_to_le_bytes(&up),
        },
        TensorSpec {
            name: "blk.0.ffn_down_exps.weight",
            dims: stacked_dims.clone(),
            ggml_type: GGML_F32,
            payload: f32_vec_to_le_bytes(&down),
        },
    ];
    let kv = [
        ("general.alignment", KvValue::U32(ALIGNMENT)),
        ("general.architecture", KvValue::Str("olmoe")),
    ];
    let bytes = build_gguf(&kv, &tensors);
    let layout = parse_bytes(bytes, "mem://stacked".into()).expect("parse");

    let pairs = list_experts(&layout);
    assert_eq!(pairs, vec![(0, 0), (0, 1), (0, 2)]);

    for e in 0..n_experts {
        let out = extract_expert(&layout, 0, e).expect("extract");
        assert_eq!(out.block, 0);
        assert_eq!(out.expert, e);
        let gate = out.gate.as_ref().expect("gate present");
        assert!(gate.stacked_slice);
        assert_eq!(gate.dims, vec![inner, outer]);
        assert_eq!(gate.bytes.len(), per_expert * 4);
        let first = f32::from_le_bytes(gate.bytes[0..4].try_into().unwrap());
        assert!((first - ((e as f32) + 0.1)).abs() < 1e-6);

        let up = out.up.as_ref().expect("up present");
        let first_up = f32::from_le_bytes(up.bytes[0..4].try_into().unwrap());
        assert!((first_up - ((e as f32) + 0.2)).abs() < 1e-6);

        let down = out.down.as_ref().expect("down present");
        let first_down = f32::from_le_bytes(down.bytes[0..4].try_into().unwrap());
        assert!((first_down - ((e as f32) + 0.3)).abs() < 1e-6);

        assert!(out.is_complete());
    }
}

#[test]
fn extracts_per_expert_tensors() {
    let inner = 2usize;
    let outer = 2usize;

    let gate0 = f32_vec_to_le_bytes(&[10.0; 4]);
    let gate1 = f32_vec_to_le_bytes(&[11.0; 4]);
    let up0 = f32_vec_to_le_bytes(&[20.0; 4]);
    let up1 = f32_vec_to_le_bytes(&[21.0; 4]);
    let down0 = f32_vec_to_le_bytes(&[30.0; 4]);
    let down1 = f32_vec_to_le_bytes(&[31.0; 4]);

    let tensors = [
        TensorSpec {
            name: "blk.0.ffn_gate.0.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: gate0,
        },
        TensorSpec {
            name: "blk.0.ffn_gate.1.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: gate1,
        },
        TensorSpec {
            name: "blk.0.ffn_up.0.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: up0,
        },
        TensorSpec {
            name: "blk.0.ffn_up.1.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: up1,
        },
        TensorSpec {
            name: "blk.0.ffn_down.0.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: down0,
        },
        TensorSpec {
            name: "blk.0.ffn_down.1.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: down1,
        },
    ];
    let kv = [("general.architecture", KvValue::Str("qwen3moe"))];
    let bytes = build_gguf(&kv, &tensors);
    let layout = parse_bytes(bytes, "mem://peri".into()).expect("parse");

    let pairs = list_experts(&layout);
    assert_eq!(pairs, vec![(0, 0), (0, 1)]);

    let e0 = extract_expert(&layout, 0, 0).unwrap();
    let gate0 = e0.gate.as_ref().unwrap();
    assert!(!gate0.stacked_slice);
    assert_eq!(gate0.source_name, "blk.0.ffn_gate.0.weight");
    let first = f32::from_le_bytes(gate0.bytes[0..4].try_into().unwrap());
    assert!((first - 10.0).abs() < 1e-6);

    let e1 = extract_expert(&layout, 0, 1).unwrap();
    let up1 = e1.up.as_ref().unwrap();
    let first = f32::from_le_bytes(up1.bytes[0..4].try_into().unwrap());
    assert!((first - 21.0).abs() < 1e-6);
}

#[test]
fn rejects_bad_magic() {
    let mut bytes = vec![b'X', b'Y', b'Z', b'!'];
    bytes.extend_from_slice(&0u64.to_le_bytes());
    let err = parse_bytes(bytes, "mem://bad-magic".into()).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("unsupported GGUF format"), "got: {msg}");
}

#[test]
fn expert_out_of_range() {
    let inner = 2usize;
    let outer = 2usize;
    let n_experts = 2usize;
    let data = f32_vec_to_le_bytes(&vec![0.0f32; inner * outer * n_experts]);
    let tensors = [TensorSpec {
        name: "blk.0.ffn_gate_exps.weight",
        dims: vec![inner, outer, n_experts],
        ggml_type: GGML_F32,
        payload: data,
    }];
    let kv = [("general.architecture", KvValue::Str("olmoe"))];
    let bytes = build_gguf(&kv, &tensors);
    let layout = parse_bytes(bytes, "mem://oor".into()).unwrap();

    let err = extract_expert(&layout, 0, 5).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("expert index out of range"), "got: {msg}");
}

#[test]
fn metadata_helper_methods() {
    // Value type for f32
    const VT_F32: u32 = 6;

    fn push_kv_f32(out: &mut Vec<u8>, key: &str, v: f32) {
        push_string(out, key);
        push_u32(out, VT_F32);
        out.extend_from_slice(&v.to_le_bytes());
    }

    let kv = [
        ("general.architecture", KvValue::Str("qwen2moe")),
        ("general.quantization_type", KvValue::Str("Q4_K_M")),
        ("qwen2moe.block_count", KvValue::U32(28)),
        ("qwen2moe.expert_count", KvValue::U32(64)),
        ("qwen2moe.expert_used_count", KvValue::U32(8)),
        ("qwen2moe.embedding_length", KvValue::U32(2048)),
        ("qwen2moe.attention.head_count", KvValue::U32(16)),
        ("general.name", KvValue::Str("Qwen2-MoE-A2.7B")),
    ];

    // Build custom GGUF with f32 metadata
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC);
    push_u32(&mut out, GGUF_VERSION);
    push_u64(&mut out, 0); // no tensors
    push_u64(&mut out, kv.len() as u64);

    for (key, value) in &kv {
        match value {
            KvValue::U32(v) => push_kv_u32(&mut out, key, *v),
            KvValue::Str(v) => push_kv_string(&mut out, key, v),
            KvValue::F32(v) => push_kv_f32(&mut out, key, *v),
        }
    }

    // Align
    while out.len() % ALIGNMENT as usize != 0 {
        out.push(0);
    }

    let layout = parse_bytes(out, "mem://metadata".into()).expect("parse");

    // Test all metadata helpers
    assert_eq!(layout.metadata.architecture(), "qwen2moe");
    assert_eq!(layout.metadata.quantization(), "Q4_K_M");
    assert_eq!(layout.metadata.block_count(), Some(28));
    assert_eq!(layout.metadata.expert_count(), Some(64));
    assert_eq!(layout.metadata.expert_used_count(), Some(8));
    assert_eq!(layout.metadata.embedding_length(), Some(2048));
    assert_eq!(layout.metadata.head_count(), Some(16));
    assert_eq!(
        layout.metadata.string("general.name"),
        Some("Qwen2-MoE-A2.7B")
    );

    // Test missing keys return None
    assert_eq!(layout.metadata.string("nonexistent"), None);
    assert_eq!(layout.metadata.float32("nonexistent"), None);
}

#[test]
fn metadata_helpers_with_alternative_keys() {
    let kv = [
        ("general.architecture", KvValue::Str("mixtral")),
        ("mixtral.num_experts", KvValue::U32(8)),
        ("mixtral.num_experts_per_tok", KvValue::U32(2)),
    ];

    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC);
    push_u32(&mut out, GGUF_VERSION);
    push_u64(&mut out, 0);
    push_u64(&mut out, kv.len() as u64);

    for (key, value) in &kv {
        match value {
            KvValue::U32(v) => push_kv_u32(&mut out, key, *v),
            KvValue::Str(v) => push_kv_string(&mut out, key, v),
            KvValue::F32(_) => unreachable!(),
        }
    }

    while out.len() % ALIGNMENT as usize != 0 {
        out.push(0);
    }

    let layout = parse_bytes(out, "mem://alt-keys".into()).expect("parse");

    // Should find alternative keys
    assert_eq!(layout.metadata.expert_count(), Some(8));
    assert_eq!(layout.metadata.expert_used_count(), Some(2));
}

#[test]
fn metadata_helpers_with_unknown_architecture() {
    let kv = [("some.block_count", KvValue::U32(10))];

    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC);
    push_u32(&mut out, GGUF_VERSION);
    push_u64(&mut out, 0);
    push_u64(&mut out, kv.len() as u64);

    for (key, value) in &kv {
        match value {
            KvValue::U32(v) => push_kv_u32(&mut out, key, *v),
            _ => unreachable!(),
        }
    }

    while out.len() % ALIGNMENT as usize != 0 {
        out.push(0);
    }

    let layout = parse_bytes(out, "mem://no-arch".into()).expect("parse");

    // Should return "unknown" and None for arch-specific queries
    assert_eq!(layout.metadata.architecture(), "unknown");
    assert_eq!(layout.metadata.quantization(), "unknown");
    assert_eq!(layout.metadata.block_count(), None);
    assert_eq!(layout.metadata.expert_count(), None);
}

#[test]
fn ggml_type_label_function() {
    use engram_parser::ggml_type_label;

    // Test common types
    assert_eq!(ggml_type_label(0), "F32");
    assert_eq!(ggml_type_label(1), "F16");
    assert_eq!(ggml_type_label(2), "Q4_0");
    assert_eq!(ggml_type_label(3), "Q4_1");
    assert_eq!(ggml_type_label(8), "Q8_0");
    assert_eq!(ggml_type_label(10), "Q2_K");
    assert_eq!(ggml_type_label(12), "Q4_K");
    assert_eq!(ggml_type_label(13), "Q5_K");
    assert_eq!(ggml_type_label(14), "Q6_K");
    assert_eq!(ggml_type_label(21), "IQ3_S");
    assert_eq!(ggml_type_label(30), "BF16");
    assert_eq!(ggml_type_label(31), "Q4_0_4_4");
    assert_ne!(ggml_type_label(31), "IQ3_M");

    // Test unknown type
    assert_eq!(ggml_type_label(999), "unknown");
}

#[test]
fn dtype_enum_comprehensive() {
    use engram_parser::DType;

    // Test F32
    let f32_dtype = DType::from_ggml_type(0);
    assert_eq!(f32_dtype, DType::F32);
    assert_eq!(f32_dtype.ggml_type(), 0);
    assert_eq!(f32_dtype.byte_len_for_elements(100), Some(400));
    assert!(f32_dtype.has_known_byte_layout());

    // Test Q4_K
    let q4k_dtype = DType::from_ggml_type(12);
    assert_eq!(q4k_dtype, DType::Q4_K);
    assert_eq!(q4k_dtype.ggml_type(), 12);
    // Q4_K: 256 elements per block, 144 bytes per block
    assert_eq!(q4k_dtype.byte_len_for_elements(256), Some(144));
    assert_eq!(q4k_dtype.byte_len_for_elements(512), Some(288));
    assert_eq!(q4k_dtype.byte_len_for_elements(100), None); // not aligned

    // Wire 31 is historical Q4_0_4_4 — layout not modeled (Other).
    let t31 = DType::from_ggml_type(31);
    assert_eq!(t31, DType::Other(31));
    assert_eq!(t31.ggml_type(), 31);
    assert_eq!(t31.byte_len_for_elements(256), None);
    assert!(!t31.has_known_byte_layout());
    assert_eq!(t31.label(), "Q4_0_4_4");

    // Test unknown type
    let unknown_dtype = DType::from_ggml_type(999);
    assert_eq!(unknown_dtype, DType::Other(999));
    assert_eq!(unknown_dtype.ggml_type(), 999);
    assert_eq!(unknown_dtype.byte_len_for_elements(100), None);
    assert!(!unknown_dtype.has_known_byte_layout());
}

#[test]
fn dtype_label_method() {
    use engram_parser::DType;

    assert_eq!(DType::F32.label(), "F32");
    assert_eq!(DType::F16.label(), "F16");
    assert_eq!(DType::Q4_0.label(), "Q4_0");
    assert_eq!(DType::Q8_0.label(), "Q8_0");
    assert_eq!(DType::Q4_K.label(), "Q4_K");
    assert_eq!(DType::IQ3_S.label(), "IQ3_S");
    assert_eq!(DType::Other(31).label(), "Q4_0_4_4");
    assert_eq!(DType::BF16.label(), "BF16");
    assert_eq!(DType::Other(999).label(), "unknown");
}

#[test]
fn tensor_with_wire_type_31_fails_closed() {
    // Wire type 31 (Q4_0_4_4) has no modeled byte layout — parse must fail closed.
    let inner = 256;
    let outer = 2;
    let n_experts = 2;
    let dummy_bytes_per_expert = 512;
    let mut payload = Vec::new();
    for i in 0..n_experts {
        for _ in 0..dummy_bytes_per_expert {
            payload.push(i as u8);
        }
    }

    let tensors = [TensorSpec {
        name: "blk.0.ffn_gate_exps.weight",
        dims: vec![inner, outer, n_experts],
        ggml_type: 31, // Q4_0_4_4 — not IQ3_M
        payload,
    }];

    let kv = [("general.architecture", KvValue::Str("testmoe"))];
    let bytes = build_gguf(&kv, &tensors);
    let err = parse_bytes(bytes, "mem://t31".into())
        .expect_err("type 31 must not parse with known layout");
    let msg = err.to_string();
    assert!(
        msg.contains("unknown byte-length")
            || msg.contains("InvalidLayout")
            || msg.contains("ggml_type=31"),
        "unexpected error: {msg}"
    );
}

#[test]
fn rejects_unsupported_gguf_version() {
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC);
    push_u32(&mut out, 2); // version 2 — not supported
    push_u64(&mut out, 0);
    push_u64(&mut out, 0);
    let err = parse_bytes(out, "mem://v2".into()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("unsupported GGUF version") || msg.contains("unsupported GGUF format"),
        "got: {msg}"
    );
}

#[test]
fn rejects_truncated_file() {
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC);
    push_u32(&mut out, GGUF_VERSION);
    // Claim one KV but provide no body → EOF while parsing.
    push_u64(&mut out, 0); // tensor_count
    push_u64(&mut out, 1); // kv_count
    let err = parse_bytes(out, "mem://trunc".into()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("EOF")
            || msg.contains("overflow")
            || msg.contains("unsupported")
            || msg.contains("InvalidLayout")
            || msg.contains("invalid"),
        "got: {msg}"
    );
}

#[test]
fn quantization_falls_back_to_file_type() {
    let kv = [
        ("general.architecture", KvValue::Str("olmoe")),
        ("general.file_type", KvValue::U32(15)),
    ];
    let bytes = build_gguf(&kv, &[]);
    let layout = parse_bytes(bytes, "mem://file-type".into()).expect("parse");
    assert_eq!(layout.metadata.quantization(), "GGUF(15)");

    let kv_f32 = [
        ("general.architecture", KvValue::Str("olmoe")),
        ("general.file_type", KvValue::U32(0)),
    ];
    let layout_f32 = parse_bytes(build_gguf(&kv_f32, &[]), "mem://ft0".into()).unwrap();
    assert_eq!(layout_f32.metadata.quantization(), "F32");

    // Explicit quantization_type wins over file_type.
    let kv_pref = [
        ("general.quantization_type", KvValue::Str("Q4_K_M")),
        ("general.file_type", KvValue::U32(15)),
    ];
    let layout_pref = parse_bytes(build_gguf(&kv_pref, &[]), "mem://pref".into()).unwrap();
    assert_eq!(layout_pref.metadata.quantization(), "Q4_K_M");
}

#[test]
fn parses_iq3_s_tensor_layout() {
    // IQ3_S: 256 elements per block, 110 bytes/block (GGUF wire layout).
    let n = 256usize;
    let payload = vec![0xABu8; 110];
    let tensors = [TensorSpec {
        name: "blk.0.ffn_gate.weight",
        dims: vec![n],
        ggml_type: GGML_IQ3_S,
        payload,
    }];
    let kv = [("general.architecture", KvValue::Str("testmoe"))];
    let layout = parse_bytes(build_gguf(&kv, &tensors), "mem://iq3s".into()).expect("parse");
    let t = layout.tensor("blk.0.ffn_gate.weight").unwrap();
    assert_eq!(t.dtype, DType::IQ3_S);
    assert_eq!(t.byte_len, 110);
    assert_eq!(layout.tensor_bytes(t).unwrap().len(), 110);
}

#[test]
fn extracts_stacked_q8_0_expert_slices() {
    // Q8_0: block size 32, 34 bytes/block. Per-expert: 32 elems → 34 bytes.
    let inner = 32usize;
    let outer = 1usize;
    let n_experts = 3usize;
    let per_expert_bytes = 34usize;
    let mut payload = Vec::with_capacity(n_experts * per_expert_bytes);
    for e in 0..n_experts {
        payload.extend(std::iter::repeat_n(e as u8, per_expert_bytes));
    }
    let tensors = [
        TensorSpec {
            name: "blk.0.ffn_gate_exps.weight",
            dims: vec![inner, outer, n_experts],
            ggml_type: GGML_Q8_0,
            payload: payload.clone(),
        },
        TensorSpec {
            name: "blk.0.ffn_up_exps.weight",
            dims: vec![inner, outer, n_experts],
            ggml_type: GGML_Q8_0,
            payload: payload.clone(),
        },
        TensorSpec {
            name: "blk.0.ffn_down_exps.weight",
            dims: vec![inner, outer, n_experts],
            ggml_type: GGML_Q8_0,
            payload,
        },
    ];
    let kv = [("general.architecture", KvValue::Str("olmoe"))];
    let layout = parse_bytes(build_gguf(&kv, &tensors), "mem://q8-stacked".into()).expect("parse");
    assert_eq!(list_experts(&layout), vec![(0, 0), (0, 1), (0, 2)]);

    for e in 0..n_experts {
        let out = extract_expert(&layout, 0, e).expect("extract");
        let gate = out.gate.as_ref().expect("gate");
        assert!(gate.stacked_slice);
        assert_eq!(gate.bytes.len(), per_expert_bytes);
        assert!(
            gate.bytes.iter().all(|&b| b == e as u8),
            "expert {e} gate bytes should be filled with {e}"
        );
        assert_eq!(gate.dtype, DType::Q8_0);
        assert!(out.is_complete());
    }
}

#[test]
fn extracts_stacked_q4_k_expert_slices() {
    // Q4_K: 256 elems/block, 144 bytes/block. dims [256, 1, 2] experts.
    let inner = 256usize;
    let outer = 1usize;
    let n_experts = 2usize;
    let per_expert_bytes = 144usize;
    let mut payload = Vec::with_capacity(n_experts * per_expert_bytes);
    for e in 0..n_experts {
        payload.extend(std::iter::repeat_n((0x10 + e) as u8, per_expert_bytes));
    }
    let tensors = [TensorSpec {
        name: "blk.0.ffn_gate_exps.weight",
        dims: vec![inner, outer, n_experts],
        ggml_type: GGML_Q4_K,
        payload,
    }];
    let kv = [("general.architecture", KvValue::Str("olmoe"))];
    let layout = parse_bytes(build_gguf(&kv, &tensors), "mem://q4k-stacked".into()).expect("parse");
    let e0 = extract_expert(&layout, 0, 0).unwrap();
    let gate0 = e0.gate.as_ref().unwrap();
    assert_eq!(gate0.bytes.len(), 144);
    assert!(gate0.bytes.iter().all(|&b| b == 0x10));
    assert_eq!(gate0.dtype, DType::Q4_K);

    let e1 = extract_expert(&layout, 0, 1).unwrap();
    let gate1 = e1.gate.as_ref().unwrap();
    assert!(gate1.bytes.iter().all(|&b| b == 0x11));
}

#[test]
fn extracts_underscore_per_expert_tensors() {
    // Alternate naming: ffn_gate_0.weight instead of ffn_gate.0.weight
    let inner = 2usize;
    let outer = 2usize;
    let tensors = [
        TensorSpec {
            name: "blk.0.ffn_gate_0.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: f32_vec_to_le_bytes(&[1.0; 4]),
        },
        TensorSpec {
            name: "blk.0.ffn_up_0.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: f32_vec_to_le_bytes(&[2.0; 4]),
        },
        TensorSpec {
            name: "blk.0.ffn_down_0.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: f32_vec_to_le_bytes(&[3.0; 4]),
        },
    ];
    let kv = [("general.architecture", KvValue::Str("qwen3moe"))];
    let layout = parse_bytes(build_gguf(&kv, &tensors), "mem://uscore".into()).expect("parse");
    let pairs = list_experts(&layout);
    assert!(
        pairs.contains(&(0, 0)),
        "expected (0,0) in list_experts, got {pairs:?}"
    );
    let e0 = extract_expert(&layout, 0, 0).expect("extract underscore expert");
    assert!(e0.is_complete());
    assert_eq!(
        e0.gate.as_ref().unwrap().source_name,
        "blk.0.ffn_gate_0.weight"
    );
}

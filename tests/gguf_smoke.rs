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

// Dtypes.
const GGML_F32: u32 = 0;

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

enum KvValue {
    U32(u32),
    Str(&'static str),
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

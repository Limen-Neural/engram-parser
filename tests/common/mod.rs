// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared test fixtures for synthetic GGUF bytes.

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
pub const GGUF_VERSION: u32 = 3;
pub const ALIGNMENT: u32 = 32;

// Value types.
pub const VT_UINT32: u32 = 4;
pub const VT_STRING: u32 = 8;

// Dtypes (GGUF wire type ids).
pub const GGML_F32: u32 = 0;
pub const GGML_Q8_0: u32 = 8;
pub const GGML_Q4_K: u32 = 12;
pub const GGML_IQ3_S: u32 = 21;

#[allow(dead_code)]
pub enum KvValue {
    U32(u32),
    Str(&'static str),
    F32(f32),
}

pub struct TensorSpec {
    pub name: &'static str,
    pub dims: Vec<usize>,
    pub ggml_type: u32,
    pub payload: Vec<u8>,
}

pub fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn push_string(out: &mut Vec<u8>, s: &str) {
    push_u64(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

pub fn push_kv_u32(out: &mut Vec<u8>, key: &str, v: u32) {
    push_string(out, key);
    push_u32(out, VT_UINT32);
    push_u32(out, v);
}

pub fn push_kv_string(out: &mut Vec<u8>, key: &str, v: &str) {
    push_string(out, key);
    push_u32(out, VT_STRING);
    push_string(out, v);
}

pub fn build_gguf(kv: &[(&str, KvValue)], tensors: &[TensorSpec]) -> Vec<u8> {
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

pub fn f32_vec_to_le_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

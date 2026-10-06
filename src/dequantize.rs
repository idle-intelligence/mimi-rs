//! INT8 → BF16 weight dequantization at load time, combined with HF→internal
//! key remapping for the pocket-tts model.
//!
//! Reads a safetensors buffer that may contain INT8-quantized weight tensors
//! (produced by `quantize.py`) alongside their per-channel BF16 scale factors.
//! Dequantizes every I8 tensor back to BF16, then returns a clean safetensors
//! buffer identical in dtype to the original non-quantized model.
//!
//! Convention produced by `quantize.py`:
//!   - Quantized weight: same tensor name as original, stored as **I8**
//!   - Scale factor:     `{name}_scale`, stored as **BF16**, shape `[out_channels]`
//!
//! Dequantization (per output channel `c`):
//!   `bf16_weight[c, ..] = bf16(i8_weight[c, ..]) * bf16_scale[c]`

use std::collections::HashSet;

/// Maps a HuggingFace tensor name to the internal model name.
///
/// Returns `None` for tensors that should be skipped entirely (unused weights).
/// Returns `Some(remapped_name)` for tensors that should be kept.
pub fn remap_key(name: &str) -> Option<String> {
    if name.contains("flow.w_s_t")
        || name.contains("quantizer.vq")
        || name.contains("quantizer.logvar_proj")
        || name.contains("learnt_padding")
    {
        return None;
    }

    let mut name = name.to_string();
    name = name.replace(
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.speaker_proj_weight",
    );
    name = name.replace(
        "flow_lm.condition_provider.conditioners.transcript_in_segment.",
        "flow_lm.conditioner.",
    );
    name = name.replace("flow_lm.backbone.", "flow_lm.transformer.");
    name = name.replace("flow_lm.flow.", "flow_lm.flow_net.");
    name = name.replace("mimi.model.", "mimi.");

    Some(name)
}

/// Dequantize any INT8 tensors in `buffer` to BF16, remap all tensor names
/// from HuggingFace names to internal model names, and return the resulting
/// safetensors buffer.
///
/// Tensors whose remapped name is `None` (via `remap_key`) are dropped.
/// `_scale` tensors are consumed during dequantization and not emitted.
pub fn dequantize_and_remap(buffer: &[u8]) -> Vec<u8> {
    let st = match safetensors::SafeTensors::deserialize(buffer) {
        Ok(st) => st,
        Err(_) => return buffer.to_vec(),
    };

    let scale_names: HashSet<String> = st
        .iter()
        .filter(|(_, t)| t.dtype() == safetensors::Dtype::I8)
        .map(|(name, _)| format!("{name}_scale"))
        .collect();

    let mut views: Vec<(String, OwnedView)> = Vec::new();

    for (name, tensor) in st.iter() {
        // Skip _scale tensors — consumed during dequantization below.
        if scale_names.contains(name) {
            continue;
        }

        let remapped = match remap_key(name) {
            Some(r) => r,
            None => continue,
        };

        if tensor.dtype() == safetensors::Dtype::I8 {
            let scale_name = format!("{name}_scale");
            let scale_tensor = match st.tensor(&scale_name) {
                Ok(t) => t,
                Err(_) => continue,
            };

            let shape = tensor.shape().to_vec();
            let out_channels = shape[0];
            let elements_per_channel: usize = shape[1..].iter().product();

            let scale_bytes = scale_tensor.data();
            let scales: Vec<half::bf16> = scale_bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
                .collect();

            let i8_data = tensor.data();
            let total = out_channels * elements_per_channel;
            let mut bf16_bytes: Vec<u8> = Vec::with_capacity(total * 2);

            for (ch, &s) in scales.iter().enumerate().take(out_channels) {
                let base = ch * elements_per_channel;
                for i in 0..elements_per_channel {
                    let q = i8_data[base + i] as i8;
                    let val = half::bf16::from_f32(q as f32) * s;
                    bf16_bytes.extend_from_slice(&val.to_le_bytes());
                }
            }

            views.push((
                remapped,
                OwnedView {
                    data: bf16_bytes,
                    shape,
                    dtype: safetensors::Dtype::BF16,
                },
            ));
        } else {
            views.push((
                remapped,
                OwnedView {
                    data: tensor.data().to_vec(),
                    shape: tensor.shape().to_vec(),
                    dtype: tensor.dtype(),
                },
            ));
        }
    }

    let view_refs: Vec<(&str, safetensors::tensor::TensorView<'_>)> = views
        .iter()
        .map(|(name, v)| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(v.dtype, v.shape.clone(), &v.data)
                    .expect("invalid tensor view"),
            )
        })
        .collect();

    safetensors::tensor::serialize(view_refs, &None).expect("serialization failed")
}

struct OwnedView {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: safetensors::Dtype,
}

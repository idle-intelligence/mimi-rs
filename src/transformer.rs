use crate::layer_scale::LayerScale;
use crate::rope::RotaryEmbedding;
use candle_core::{Device, Result, Tensor};
use candle_nn::{LayerNorm, LayerNormConfig, Linear, Module, VarBuilder};

// ---- KV Cache ----

/// Simple append-based KV cache with optional context window trimming.
#[derive(Clone, Debug)]
pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    max_seq_len: usize,
    absolute_offset: usize,
}

impl KvCache {
    pub fn new(max_seq_len: usize) -> Self {
        Self { k: None, v: None, max_seq_len, absolute_offset: 0 }
    }

    pub fn current_seq_len(&self) -> usize {
        match &self.k {
            Some(k) => k.dim(2).unwrap_or(0), // k shape: [b, h, seq, d]
            None => 0,
        }
    }

    /// Append new k, v (shape [b, h, t, d]) and return full (k, v).
    /// Trims to max_seq_len if exceeded.
    pub fn append(
        &mut self,
        new_k: &Tensor,
        new_v: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (k, v) = match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => {
                let k = Tensor::cat(&[prev_k, new_k], 2)?;
                let v = Tensor::cat(&[prev_v, new_v], 2)?;
                (k, v)
            }
            _ => (new_k.clone(), new_v.clone()),
        };

        let seq_len = k.dim(2)?;
        let (k, v) = if seq_len > self.max_seq_len {
            let trim = seq_len - self.max_seq_len;
            (
                k.narrow(2, trim, self.max_seq_len)?.contiguous()?,
                v.narrow(2, trim, self.max_seq_len)?.contiguous()?,
            )
        } else {
            (k, v)
        };

        let new_tokens = new_k.dim(2)?;
        self.absolute_offset += new_tokens;
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }
}

// ---- StreamingMHAState (FlowLM pre-allocated KV cache) ----

/// State for StreamingMultiheadAttention (FlowLM).
/// Uses a growing Vec<Tensor> approach instead of slice_set.
#[derive(Debug, Clone)]
pub struct StreamingMHAState {
    /// Accumulated key tensors (each [B, T_i, H, D])
    pub k_chunks: Vec<Tensor>,
    /// Accumulated value tensors (each [B, T_i, H, D])
    pub v_chunks: Vec<Tensor>,
    /// Current end position (number of tokens seen so far).
    pub current_end: usize,
}

impl StreamingMHAState {
    pub fn new() -> Self {
        Self { k_chunks: Vec::new(), v_chunks: Vec::new(), current_end: 0 }
    }

    /// Create a pre-loaded state from existing k/v tensors and a current end position.
    /// Used for loading voice KV caches.
    pub fn with_kv(k: Tensor, v: Tensor, current_end: usize) -> Self {
        Self { k_chunks: vec![k], v_chunks: vec![v], current_end }
    }

    fn get_kv(&self) -> Result<Option<(Tensor, Tensor)>> {
        if self.k_chunks.is_empty() {
            return Ok(None);
        }
        if self.k_chunks.len() == 1 {
            return Ok(Some((self.k_chunks[0].clone(), self.v_chunks[0].clone())));
        }
        let k_refs: Vec<&Tensor> = self.k_chunks.iter().collect();
        let v_refs: Vec<&Tensor> = self.v_chunks.iter().collect();
        Ok(Some((Tensor::cat(&k_refs, 1)?, Tensor::cat(&v_refs, 1)?)))
    }
}

impl Default for StreamingMHAState {
    fn default() -> Self {
        Self::new()
    }
}

// ---- State types ----

#[derive(Clone, Debug)]
pub enum LayerAttentionState {
    Mimi(KvCache),
    FlowLm(StreamingMHAState),
}

#[derive(Clone, Debug)]
pub struct StreamingTransformerState {
    pub layer_states: Vec<LayerAttentionState>,
}

impl StreamingTransformerState {
    pub fn current_seq_len(&self) -> usize {
        if self.layer_states.is_empty() {
            return 0;
        }
        match &self.layer_states[0] {
            LayerAttentionState::Mimi(cache) => cache.current_seq_len(),
            LayerAttentionState::FlowLm(state) => state.current_end,
        }
    }
}

// ---- Causal mask helper ----

fn causal_mask(num_queries: usize, num_keys: usize, device: &Device) -> Result<Tensor> {
    let shift = num_keys - num_queries;
    let mut data = Vec::with_capacity(num_queries * num_keys);
    for q in 0..num_queries {
        for k in 0..num_keys {
            if k <= q + shift {
                data.push(0f32);
            } else {
                data.push(f32::NEG_INFINITY);
            }
        }
    }
    Tensor::from_vec(data, (num_queries, num_keys), device)
}

// ---- MimiStreamingMultiheadAttention ----
// Uses KV cache with context window.

struct MimiStreamingMHA {
    in_proj: Linear,
    out_proj: Linear,
    embed_dim: usize,
    num_heads: usize,
    context: usize,
}

impl MimiStreamingMHA {
    fn load(vb: VarBuilder, embed_dim: usize, num_heads: usize, context: usize) -> Result<Self> {
        let out_dim = 3 * embed_dim;
        let in_proj = candle_nn::linear_no_bias(embed_dim, out_dim, vb.pp("in_proj"))?;
        let out_proj = candle_nn::linear_no_bias(embed_dim, embed_dim, vb.pp("out_proj"))?;
        Ok(Self { in_proj, out_proj, embed_dim, num_heads, context })
    }

    fn init_state(&self) -> KvCache {
        KvCache::new(self.context)
    }

    fn forward(
        &self,
        query: &Tensor,
        rope: &RotaryEmbedding,
        state: &mut KvCache,
    ) -> Result<Tensor> {
        let (b, t, _) = query.dims3()?;
        let d = self.embed_dim / self.num_heads;
        let offset = state.absolute_offset;

        let projected = self.in_proj.forward(query)?;
        // Reshape to [B, T, 3, H, D]
        let packed = projected.reshape((b, t, 3, self.num_heads, d))?;
        let q = packed.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?; // [B, T, H, D]
        let k = packed.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?;
        let v = packed.narrow(2, 2, 1)?.squeeze(2)?.contiguous()?;

        // RoPE on [B, T, H, D]
        let (q, k) = rope.forward(&q, &k, offset)?;

        // To [B, H, T, D]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // KV cache with context trimming
        let (k, v) = state.append(&k, &v)?;

        // Scaled dot-product attention
        let scale = (d as f64).sqrt().recip();
        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let attn = (attn * scale)?;

        // Causal mask
        let kv_len = k.dim(2)?;
        let mask = causal_mask(t, kv_len, query.device())?;
        // Broadcast mask over [B, H, T, kv_len]
        let mask = mask.reshape((1, 1, t, kv_len))?;
        let attn = attn.broadcast_add(&mask)?;

        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let x = attn.matmul(&v)?;

        let x = x.transpose(1, 2)?.contiguous()?.reshape((b, t, self.embed_dim))?;
        self.out_proj.forward(&x)
    }
}

// ---- StreamingMultiheadAttention (FlowLM) ----

pub struct StreamingMultiheadAttention {
    in_proj: Linear,
    out_proj: Linear,
    pub embed_dim: usize,
    pub num_heads: usize,
}

impl StreamingMultiheadAttention {
    pub fn load(vb: VarBuilder, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let out_dim = 3 * embed_dim;
        let in_proj = candle_nn::linear_no_bias(embed_dim, out_dim, vb.pp("in_proj"))?;
        let out_proj = candle_nn::linear_no_bias(embed_dim, embed_dim, vb.pp("out_proj"))?;
        Ok(Self { in_proj, out_proj, embed_dim, num_heads })
    }

    pub fn init_state(&self) -> StreamingMHAState {
        StreamingMHAState::new()
    }

    pub fn forward(
        &self,
        query: &Tensor,
        rope: &RotaryEmbedding,
        state: &mut StreamingMHAState,
    ) -> Result<Tensor> {
        let (b, t, _) = query.dims3()?;
        let d = self.embed_dim / self.num_heads;
        let offset = state.current_end;

        let projected = self.in_proj.forward(query)?;
        let ed = self.embed_dim;
        let q = projected.narrow(2, 0, ed)?.contiguous()?.reshape((b, t, self.num_heads, d))?;
        let k =
            projected.narrow(2, ed, ed)?.contiguous()?.reshape((b, t, self.num_heads, d))?;
        let v = projected
            .narrow(2, 2 * ed, ed)?
            .contiguous()?
            .reshape((b, t, self.num_heads, d))?;

        // Apply RoPE: q, k are [B, T, H, D]
        let (q, k) = rope.forward(&q, &k, offset)?;

        // Accumulate KV chunks
        state.k_chunks.push(k);
        state.v_chunks.push(v);
        state.current_end += t;

        let (k_full, v_full) = state.get_kv()?.unwrap();
        let kv_len = k_full.dim(1)?;

        // Transpose to [B, H, T, D] for attention
        let q = q.transpose(1, 2)?;
        let k_full = k_full.transpose(1, 2)?;
        let v_full = v_full.transpose(1, 2)?;

        // Causal mask
        let mask = causal_mask(t, kv_len, query.device())?;
        let mask = mask.reshape((1, 1, t, kv_len))?;

        // Scaled dot-product attention
        let scale = (d as f64).sqrt().recip();
        let attn = q.matmul(&k_full.transpose(2, 3)?)?;
        let attn = (attn * scale)?;
        let attn = attn.broadcast_add(&mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let x = attn.matmul(&v_full)?;

        // Back to [B, T, H*D]
        let x = x.transpose(1, 2)?.contiguous()?.reshape((b, t, self.embed_dim))?;
        self.out_proj.forward(&x)
    }
}

// ---- StreamingTransformerLayer ----

enum AttentionKind {
    Mimi(MimiStreamingMHA),
    FlowLm(StreamingMultiheadAttention),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Mimi,
    FlowLm,
}

pub struct StreamingTransformerLayer {
    self_attn: AttentionKind,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    layer_scale_1: Option<LayerScale>,
    layer_scale_2: Option<LayerScale>,
}

impl StreamingTransformerLayer {
    pub fn load(
        vb: VarBuilder,
        d_model: usize,
        num_heads: usize,
        dim_feedforward: usize,
        context: Option<usize>,
        layer_scale: Option<f64>,
        kind: Kind,
    ) -> Result<Self> {
        let self_attn = match kind {
            Kind::Mimi => AttentionKind::Mimi(MimiStreamingMHA::load(
                vb.pp("self_attn"),
                d_model,
                num_heads,
                context.unwrap_or(250),
            )?),
            Kind::FlowLm => AttentionKind::FlowLm(StreamingMultiheadAttention::load(
                vb.pp("self_attn"),
                d_model,
                num_heads,
            )?),
        };

        let ln_cfg = LayerNormConfig { eps: 1e-5, ..Default::default() };
        let norm1 = candle_nn::layer_norm(d_model, ln_cfg, vb.pp("norm1"))?;
        let norm2 = candle_nn::layer_norm(d_model, ln_cfg, vb.pp("norm2"))?;
        let linear1 = candle_nn::linear_no_bias(d_model, dim_feedforward, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear_no_bias(dim_feedforward, d_model, vb.pp("linear2"))?;

        let layer_scale_1 = if layer_scale.is_some() {
            Some(LayerScale::load(vb.pp("layer_scale_1"), d_model)?)
        } else {
            None
        };
        let layer_scale_2 = if layer_scale.is_some() {
            Some(LayerScale::load(vb.pp("layer_scale_2"), d_model)?)
        } else {
            None
        };

        Ok(Self { self_attn, norm1, norm2, linear1, linear2, layer_scale_1, layer_scale_2 })
    }

    pub fn init_state(&self) -> LayerAttentionState {
        match &self.self_attn {
            AttentionKind::Mimi(attn) => LayerAttentionState::Mimi(attn.init_state()),
            AttentionKind::FlowLm(attn) => LayerAttentionState::FlowLm(attn.init_state()),
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        state: &mut LayerAttentionState,
    ) -> Result<Tensor> {
        // Self-attention block: x + layer_scale_1(attn(norm1(x)))
        let norm1_out = self.norm1.forward(x)?;
        let mut attn_out = match (&self.self_attn, state) {
            (AttentionKind::Mimi(attn), LayerAttentionState::Mimi(cache)) => {
                attn.forward(&norm1_out, rope, cache)?
            }
            (AttentionKind::FlowLm(attn), LayerAttentionState::FlowLm(mha_state)) => {
                attn.forward(&norm1_out, rope, mha_state)?
            }
            _ => candle_core::bail!("attention kind and state type mismatch"),
        };
        if let Some(ls) = &self.layer_scale_1 {
            attn_out = ls.forward(&attn_out)?;
        }
        let x = (x + &attn_out)?;

        // FF block: x + layer_scale_2(ff(norm2(x)))
        let norm2_out = self.norm2.forward(&x)?;
        let mut ff_out = self.linear1.forward(&norm2_out)?;
        ff_out = ff_out.gelu_erf()?;
        ff_out = self.linear2.forward(&ff_out)?;
        if let Some(ls) = &self.layer_scale_2 {
            ff_out = ls.forward(&ff_out)?;
        }
        &x + &ff_out
    }
}

// ---- StreamingTransformer ----

pub struct StreamingTransformer {
    pub layers: Vec<StreamingTransformerLayer>,
    rope: RotaryEmbedding,
}

impl StreamingTransformer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: VarBuilder,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        layer_scale: Option<f64>,
        dim_feedforward: usize,
        context: Option<usize>,
        max_period: f64,
        kind: Kind,
    ) -> Result<Self> {
        let head_dim = d_model / num_heads;
        let max_seq_len = 8192;
        let rope =
            RotaryEmbedding::new(head_dim, max_seq_len, max_period, vb.device())?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(StreamingTransformerLayer::load(
                vb.pp(&format!("layers.{i}")),
                d_model,
                num_heads,
                dim_feedforward,
                context,
                layer_scale,
                kind,
            )?);
        }

        Ok(Self { layers, rope })
    }

    pub fn init_state(&self) -> StreamingTransformerState {
        let layer_states = self.layers.iter().map(|l| l.init_state()).collect();
        StreamingTransformerState { layer_states }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        state: &mut StreamingTransformerState,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        for (layer, layer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
            x = layer.forward(&x, &self.rope, layer_state)?;
        }
        Ok(x)
    }
}

// ---- ProjectedTransformer ----

pub struct ProjectedTransformer {
    pub transformer: StreamingTransformer,
    input_proj: Option<Linear>,
    output_projs: Vec<Option<Linear>>,
}

impl ProjectedTransformer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: VarBuilder,
        input_dimension: usize,
        output_dimensions: &[usize],
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        layer_scale: Option<f64>,
        context: usize,
        max_period: f64,
        dim_feedforward: usize,
    ) -> Result<Self> {
        let transformer = StreamingTransformer::load(
            vb.pp("transformer"),
            d_model,
            num_heads,
            num_layers,
            layer_scale,
            dim_feedforward,
            Some(context),
            max_period,
            Kind::Mimi,
        )?;

        let input_proj = if d_model != input_dimension {
            Some(candle_nn::linear(input_dimension, d_model, vb.pp("input_proj"))?)
        } else {
            None
        };

        let mut output_projs = Vec::new();
        for (i, &out_dim) in output_dimensions.iter().enumerate() {
            if d_model == out_dim {
                output_projs.push(None);
            } else {
                let proj =
                    candle_nn::linear(d_model, out_dim, vb.pp(&format!("output_proj.{i}")))?;
                output_projs.push(Some(proj));
            }
        }

        Ok(Self { transformer, input_proj, output_projs })
    }

    pub fn init_state(&self) -> StreamingTransformerState {
        self.transformer.init_state()
    }

    /// Forward pass. Input x is [B, C, T] (conv layout).
    pub fn forward(
        &self,
        x: &Tensor,
        state: &mut StreamingTransformerState,
    ) -> Result<Vec<Tensor>> {
        // [B, C, T] -> [B, T, C]
        let x = x.transpose(1, 2)?.contiguous()?;

        let x = match &self.input_proj {
            Some(proj) => proj.forward(&x)?,
            None => x,
        };

        let z = self.transformer.forward(&x, state)?;

        let mut ys = Vec::with_capacity(self.output_projs.len());
        for proj in &self.output_projs {
            let y = match proj {
                Some(p) => p.forward(&z)?,
                None => z.clone(),
            };
            ys.push(y.transpose(1, 2)?.contiguous()?);
        }
        Ok(ys)
    }
}

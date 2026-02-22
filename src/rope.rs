use candle_core::{Device, Result, Tensor};

/// Rotary position embedding (RoPE).
/// Precomputes cos/sin tables and applies interleaved RoPE.
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        max_period: f64,
        device: &Device,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;
        let mut cos_data = Vec::with_capacity(max_seq_len * half_dim);
        let mut sin_data = Vec::with_capacity(max_seq_len * half_dim);

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let inv_freq = 1.0f64 / max_period.powf(i as f64 / half_dim as f64);
                let angle = pos as f64 * inv_freq;
                cos_data.push(angle.cos() as f32);
                sin_data.push(angle.sin() as f32);
            }
        }

        let cos = Tensor::from_vec(cos_data, (max_seq_len, half_dim), device)?;
        let sin = Tensor::from_vec(sin_data, (max_seq_len, half_dim), device)?;
        Ok(Self { cos, sin })
    }

    /// Apply interleaved RoPE to a tensor of shape [B, T, H, D].
    /// Adjacent pairs (x[..., 2i], x[..., 2i+1]) are rotated by angle theta_i.
    fn apply_rope(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, t, h, d) = x.dims4()?;
        let half_d = d / 2;

        // Get cos/sin for current positions [T, half_d]
        let cos = self.cos.narrow(0, offset, t)?;
        let sin = self.sin.narrow(0, offset, t)?;

        // Reshape x to separate even/odd: [B, T, H, half_d, 2]
        let x = x.reshape((b, t, h, half_d, 2))?;

        // Extract even/odd: both [B, T, H, half_d]
        let x_even = x.narrow(4, 0, 1)?.squeeze(4)?;
        let x_odd = x.narrow(4, 1, 1)?.squeeze(4)?;

        // Broadcast cos/sin: [1, T, 1, half_d]
        let cos = cos.reshape((1, t, 1, half_d))?;
        let sin = sin.reshape((1, t, 1, half_d))?;

        // Rotate: x'_even = x_even * cos - x_odd * sin
        //         x'_odd  = x_even * sin + x_odd * cos
        let out_even = (x_even.broadcast_mul(&cos)? - x_odd.broadcast_mul(&sin)?)?;
        let out_odd = (x_even.broadcast_mul(&sin)? + x_odd.broadcast_mul(&cos)?)?;

        // Interleave back: stack on last dim -> [B, T, H, half_d, 2]
        let out_even = out_even.unsqueeze(4)?;
        let out_odd = out_odd.unsqueeze(4)?;
        let out = Tensor::cat(&[&out_even, &out_odd], 4)?;

        // Reshape back to [B, T, H, D]
        out.reshape((b, t, h, d))
    }

    /// Apply RoPE to q and k tensors in [B, T, H, D] layout.
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let q = self.apply_rope(q, offset)?;
        let k = self.apply_rope(k, offset)?;
        Ok((q, k))
    }
}

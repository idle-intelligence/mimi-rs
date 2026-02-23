use candle_core::{Device, Result, Tensor};

/// Rotary position embedding (RoPE).
/// Precomputes cos/sin tables and computes on the fly for positions beyond the table.
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    half_dim: usize,
    max_period: f64,
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
        Ok(Self { cos, sin, half_dim, max_period })
    }

    /// Compute cos/sin for positions [offset..offset+t], using the pre-computed
    /// table when possible, falling back to on-the-fly computation for large offsets.
    fn get_cos_sin(&self, offset: usize, t: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let table_len = self.cos.dim(0)?;
        if offset + t <= table_len {
            // Fast path: slice from pre-computed table
            Ok((
                self.cos.narrow(0, offset, t)?,
                self.sin.narrow(0, offset, t)?,
            ))
        } else {
            // Compute on the fly for positions beyond the table
            let mut cos_data = Vec::with_capacity(t * self.half_dim);
            let mut sin_data = Vec::with_capacity(t * self.half_dim);
            for pos in offset..offset + t {
                for i in 0..self.half_dim {
                    let inv_freq = 1.0f64 / self.max_period.powf(i as f64 / self.half_dim as f64);
                    let angle = pos as f64 * inv_freq;
                    cos_data.push(angle.cos() as f32);
                    sin_data.push(angle.sin() as f32);
                }
            }
            Ok((
                Tensor::from_vec(cos_data, (t, self.half_dim), device)?,
                Tensor::from_vec(sin_data, (t, self.half_dim), device)?,
            ))
        }
    }

    /// Apply interleaved RoPE to a tensor of shape [B, T, H, D].
    /// Adjacent pairs (x[..., 2i], x[..., 2i+1]) are rotated by angle theta_i.
    fn apply_rope(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, t, h, d) = x.dims4()?;
        let half_d = d / 2;

        // Get cos/sin for current positions [T, half_d]
        let (cos, sin) = self.get_cos_sin(offset, t, x.device())?;

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that on-the-fly RoPE matches the pre-computed table.
    #[test]
    fn on_the_fly_matches_table() -> Result<()> {
        let head_dim = 8;
        let table_size = 16;
        let max_period = 10000.0;
        let dev = Device::Cpu;

        let rope = RotaryEmbedding::new(head_dim, table_size, max_period, &dev)?;

        // Get positions 10..13 from the table (fast path)
        let (cos_table, sin_table) = rope.get_cos_sin(10, 3, &dev)?;

        // Force on-the-fly by requesting same positions but via a fresh rope with tiny table
        let rope_tiny = RotaryEmbedding::new(head_dim, 4, max_period, &dev)?;
        let (cos_fly, sin_fly) = rope_tiny.get_cos_sin(10, 3, &dev)?;

        let cos_diff: f32 = (cos_table - cos_fly)?.abs()?.sum_all()?.to_scalar()?;
        let sin_diff: f32 = (sin_table - sin_fly)?.abs()?.sum_all()?.to_scalar()?;
        assert!(cos_diff < 1e-6, "cos mismatch: {cos_diff}");
        assert!(sin_diff < 1e-6, "sin mismatch: {sin_diff}");
        Ok(())
    }

    /// RoPE must not panic for offsets far beyond the pre-computed table.
    /// This is the bug that caused WASM crashes after ~5.5 minutes of streaming.
    #[test]
    fn large_offset_no_panic() -> Result<()> {
        let head_dim = 64;
        let table_size = 128; // small table
        let max_period = 10000.0;
        let dev = Device::Cpu;

        let rope = RotaryEmbedding::new(head_dim, table_size, max_period, &dev)?;

        // Simulate streaming: offsets way beyond the table
        let x = Tensor::randn(0f32, 1.0, (1, 1, 8, head_dim), &dev)?;
        for offset in [0, 127, 128, 1000, 8192, 50000] {
            let (q, _k) = rope.forward(&x, &x, offset)?;
            assert_eq!(q.dims(), &[1, 1, 8, head_dim]);
        }
        Ok(())
    }
}


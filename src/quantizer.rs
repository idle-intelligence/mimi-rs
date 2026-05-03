//! Residual Vector Quantizer (RVQ) for Mimi.
//!
//! Split architecture: 1 semantic codebook (rvq_first) + N acoustic codebooks (rvq_rest).
//! Each sub-RVQ has input/output Conv1d projections (kernel_size=1) between
//! model dimension (512) and codebook dimension (256).

use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};

/// Single vector quantizer codebook.
pub struct VectorQuantizer {
    /// Codebook embeddings: [n_bins, dim]
    codebook: Tensor,
    /// Pre-computed squared norms: [n_bins]
    codebook_sq_norms: Tensor,
}

impl VectorQuantizer {
    pub fn new(codebook: Tensor) -> Result<Self> {
        // Pre-compute ||c||² for each codebook entry
        let codebook_sq_norms = codebook.sqr()?.sum(1)?;
        Ok(Self { codebook, codebook_sq_norms })
    }

    /// Decode indices back to codebook vectors.
    ///
    /// Input: indices [B, 1, T] (u32)
    /// Returns: quantized [B, dim, T]
    pub fn decode(&self, indices: &Tensor) -> Result<Tensor> {
        let (b, _one, t) = indices.dims3()?;
        let dim = self.codebook.dim(1)?;
        // Flatten to [B*T]
        let flat = indices.flatten_all()?;
        let flat_u32 = flat.to_dtype(DType::U32)?;
        let gathered = self.codebook.index_select(&flat_u32, 0)?; // [B*T, dim]
        gathered
            .reshape((b, t, dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// Quantize input vectors to nearest codebook entries.
    ///
    /// Input: [B, dim, T]
    /// Returns: (quantized [B, dim, T], indices [B, 1, T])
    ///
    /// Uses `argmin(||c||² - 2·x·cᵀ)` — the `||x||²` term is constant across
    /// codebook entries and can be omitted.
    pub fn encode(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, dim, t) = x.dims3()?;

        // Flatten to [B*T, dim] for batch matmul
        let x_flat = x.transpose(1, 2)?.contiguous()?.reshape((b * t, dim))?;

        // dots = x @ codebook.T → [B*T, n_bins]
        let dots = x_flat.matmul(&self.codebook.t()?)?;

        // distances = ||c||² - 2·x·c  (broadcast [n_bins] over [B*T, n_bins])
        let distances = self
            .codebook_sq_norms
            .unsqueeze(0)?
            .broadcast_sub(&(dots * 2.0)?)?;

        // indices = argmin over codebook dimension → [B*T]
        let indices = distances.argmin(1)?;

        // Gather quantized vectors
        let quantized_flat = self.codebook.index_select(&indices, 0)?; // [B*T, dim]
        let quantized = quantized_flat
            .reshape((b, t, dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let indices = indices.to_dtype(DType::U32)?.reshape((b, 1, t))?;

        Ok((quantized, indices))
    }
}

/// Residual Vector Quantizer with input/output projections.
///
/// Projects from model dimension to codebook dimension, applies residual VQ,
/// then projects back.
pub struct ResidualVectorQuantizer {
    /// Project model dim → codebook dim (Conv1d kernel_size=1)
    input_proj: Conv1d,
    /// Project codebook dim → model dim (Conv1d kernel_size=1)
    #[allow(dead_code)]
    output_proj: Conv1d,
    /// Residual VQ codebooks
    quantizers: Vec<VectorQuantizer>,
}

impl ResidualVectorQuantizer {
    pub fn load(
        vb: VarBuilder,
        n_codebooks: usize,
        input_dim: usize,
        codebook_dim: usize,
        codebook_bins: usize,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let input_proj =
            candle_nn::conv1d_no_bias(input_dim, codebook_dim, 1, cfg, vb.pp("input_proj"))?;
        let output_proj =
            candle_nn::conv1d_no_bias(codebook_dim, input_dim, 1, cfg, vb.pp("output_proj"))?;

        let mut quantizers = Vec::with_capacity(n_codebooks);
        for i in 0..n_codebooks {
            let codebook =
                Self::load_codebook(&vb.pp(&format!("layers.{i}.codebook")), codebook_bins, codebook_dim)?;
            quantizers.push(VectorQuantizer::new(codebook)?);
        }

        Ok(Self { input_proj, output_proj, quantizers })
    }

    /// Load and normalize a codebook from embed_sum / max(cluster_usage, 1.0).
    fn load_codebook(
        vb: &VarBuilder,
        n_bins: usize,
        codebook_dim: usize,
    ) -> Result<Tensor> {
        let embed_sum = vb.get((n_bins, codebook_dim), "embed_sum")?;
        let cluster_usage = vb.get(n_bins, "cluster_usage")?;
        let ones = cluster_usage.ones_like()?;
        let usage = cluster_usage.maximum(&ones)?.unsqueeze(1)?;
        embed_sum.broadcast_div(&usage)
    }

    /// Decode: [B, n_q, T] token indices (u32) → [B, input_dim, T].
    ///
    /// Looks up each codebook, sums the quantized vectors, then projects back.
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        self.decode_n(codes, self.quantizers.len())
    }

    /// Decode using only the first `n` codebooks (partial RVQ decode).
    ///
    /// Useful when the model generates fewer codebook tokens than the full RVQ.
    /// Unused codebooks are simply not summed.
    pub fn decode_n(&self, codes: &Tensor, n: usize) -> Result<Tensor> {
        let n = n.min(self.quantizers.len());
        let mut sum: Option<Tensor> = None;
        for (i, vq) in self.quantizers.iter().take(n).enumerate() {
            let code_i = codes.narrow(1, i, 1)?; // [B, 1, T]
            let quantized = vq.decode(&code_i)?; // [B, codebook_dim, T]
            sum = Some(match sum {
                Some(s) => s.add(&quantized)?,
                None => quantized,
            });
        }
        let reconstructed = sum.unwrap();
        self.output_proj.forward(&reconstructed)
    }

    /// Encode: [B, input_dim, T] → [B, n_q, T] token indices (u32).
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        // Project input_dim → codebook_dim
        let projected = self.input_proj.forward(x)?;

        let mut all_codes = Vec::with_capacity(self.quantizers.len());
        let mut residual = projected;

        for quantizer in &self.quantizers {
            let (quantized, indices) = quantizer.encode(&residual)?;
            all_codes.push(indices); // [B, 1, T]
            residual = residual.sub(&quantized)?;
        }

        // Stack: [B, n_q, T]
        Tensor::cat(&all_codes, 1)
    }
}

/// Split Residual Vector Quantizer.
///
/// Split architecture: rvq_first (semantic codebooks) + rvq_rest (acoustic codebooks).
/// Each sub-RVQ independently projects and quantizes the same input.
pub struct SplitResidualVectorQuantizer {
    pub rvq_first: ResidualVectorQuantizer,
    pub rvq_rest: ResidualVectorQuantizer,
    pub n_q: usize,
}

impl SplitResidualVectorQuantizer {
    pub fn load(
        vb: VarBuilder,
        n_q_semantic: usize,
        n_q_acoustic: usize,
        input_dim: usize,
        codebook_dim: usize,
        codebook_bins: usize,
    ) -> Result<Self> {
        let rvq_first = ResidualVectorQuantizer::load(
            vb.pp("semantic_residual_vector_quantizer"),
            n_q_semantic,
            input_dim,
            codebook_dim,
            codebook_bins,
        )?;
        let rvq_rest = ResidualVectorQuantizer::load(
            vb.pp("acoustic_residual_vector_quantizer"),
            n_q_acoustic,
            input_dim,
            codebook_dim,
            codebook_bins,
        )?;
        Ok(Self {
            rvq_first,
            rvq_rest,
            n_q: n_q_semantic + n_q_acoustic,
        })
    }

    /// Encode: [B, dim, T] → [B, n_q, T] token indices (u32).
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let first_codes = self.rvq_first.encode(x)?; // [B, n_first, T]
        let rest_codes = self.rvq_rest.encode(x)?; // [B, n_rest, T]
        Tensor::cat(&[&first_codes, &rest_codes], 1)
    }

    /// Decode: [B, n_q, T] token indices (u32) → [B, dim, T].
    ///
    /// Splits codes into semantic and acoustic, decodes each sub-RVQ, sums results.
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let n_first = self.rvq_first.quantizers.len();
        let first_codes = codes.narrow(1, 0, n_first)?;
        let rest_codes = codes.narrow(1, n_first, self.rvq_rest.quantizers.len())?;
        let first_latent = self.rvq_first.decode(&first_codes)?;
        let rest_latent = self.rvq_rest.decode(&rest_codes)?;
        first_latent.add(&rest_latent)
    }

    /// Decode using only the first `n_total` codebooks (partial decode).
    ///
    /// When the model only generates tokens for a subset of codebooks,
    /// this avoids adding garbage from unused codebook entries.
    pub fn decode_n(&self, codes: &Tensor, n_total: usize) -> Result<Tensor> {
        let n_first = self.rvq_first.quantizers.len();
        let first_codes = codes.narrow(1, 0, n_first.min(n_total))?;
        let first_latent = self.rvq_first.decode_n(&first_codes, n_total.min(n_first))?;

        if n_total <= n_first {
            return Ok(first_latent);
        }

        let n_rest = n_total - n_first;
        let rest_codes = codes.narrow(1, n_first, n_rest)?;
        let rest_latent = self.rvq_rest.decode_n(&rest_codes, n_rest)?;
        first_latent.add(&rest_latent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;

    const DEV: Device = Device::Cpu;

    // ---- VectorQuantizer tests ----

    /// Build a small 4-entry, 3-dim codebook and verify that encode() returns the
    /// correct nearest-neighbor index for each input vector.
    #[test]
    fn vq_nearest_neighbor_indices() -> Result<()> {
        // Codebook: 4 entries in 3-D
        //   0: [1, 0, 0]
        //   1: [0, 1, 0]
        //   2: [0, 0, 1]
        //   3: [1, 1, 1]
        let codebook = Tensor::new(
            &[[1.0f32, 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]],
            &DEV,
        )?;
        let vq = VectorQuantizer::new(codebook)?;

        // Input: [B=1, dim=3, T=4] — each time step is near one codebook entry
        let input = Tensor::new(
            &[[[0.9f32, 0.1, 0.05, 0.8],
               [0.1,    0.9, 0.1,  0.9],
               [0.0,    0.0, 0.85, 0.7]]],
            &DEV,
        )?;
        // Expected nearest:
        //   t=0: [0.9, 0.1, 0.0] → entry 0 [1,0,0]
        //   t=1: [0.1, 0.9, 0.0] → entry 1 [0,1,0]
        //   t=2: [0.05,0.1, 0.85]→ entry 2 [0,0,1]
        //   t=3: [0.8, 0.9, 0.7] → entry 3 [1,1,1]

        let (quantized, indices) = vq.encode(&input)?;

        // Check index values
        let idx_vec: Vec<u32> = indices.flatten_all()?.to_vec1()?;
        assert_eq!(idx_vec, vec![0, 1, 2, 3]);

        // Check that quantized vectors equal the selected codebook entries
        // quantized shape: [1, 3, 4]
        let q_flat = quantized.squeeze(0)?.t()?.contiguous()?; // [4, 3]
        let expected = Tensor::new(
            &[[1.0f32, 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]],
            &DEV,
        )?;
        let diff = q_flat.sub(&expected)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6, "quantized vectors should match codebook entries, diff={diff}");

        Ok(())
    }

    /// Verify output shapes from VectorQuantizer::encode() with batch and time dims.
    #[test]
    fn vq_encode_output_shapes() -> Result<()> {
        let n_bins = 8;
        let dim = 5;
        let codebook = Tensor::randn(0f32, 1.0, (n_bins, dim), &DEV)?;
        let vq = VectorQuantizer::new(codebook)?;

        let b = 2;
        let t = 7;
        let input = Tensor::randn(0f32, 1.0, (b, dim, t), &DEV)?;
        let (quantized, indices) = vq.encode(&input)?;

        assert_eq!(quantized.dims(), &[b, dim, t]);
        assert_eq!(indices.dims(), &[b, 1, t]);
        assert_eq!(indices.dtype(), DType::U32);

        Ok(())
    }

    /// Verify that all returned indices are valid (< n_bins).
    #[test]
    fn vq_indices_in_range() -> Result<()> {
        let n_bins: u32 = 4;
        let dim = 3;
        let codebook = Tensor::randn(0f32, 1.0, (n_bins as usize, dim), &DEV)?;
        let vq = VectorQuantizer::new(codebook)?;

        let input = Tensor::randn(0f32, 1.0, (3, dim, 10), &DEV)?;
        let (_quantized, indices) = vq.encode(&input)?;

        let idx_vec: Vec<u32> = indices.flatten_all()?.to_vec1()?;
        for &idx in &idx_vec {
            assert!(idx < n_bins, "index {idx} should be < n_bins={n_bins}");
        }

        Ok(())
    }

    // ---- ResidualVectorQuantizer tests ----

    /// Helper: build a VarBuilder containing all tensors required by
    /// ResidualVectorQuantizer::load(prefix, n_codebooks, input_dim, codebook_dim, bins).
    ///
    /// Conv1d weights are identity-like (scaled) and codebooks are synthetic.
    fn make_rvq_tensors(
        prefix: &str,
        n_codebooks: usize,
        input_dim: usize,
        codebook_dim: usize,
        codebook_bins: usize,
    ) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();

        // input_proj.weight: [codebook_dim, input_dim, 1]
        // Use identity-like projection when dims match, otherwise random
        let input_w = if input_dim == codebook_dim {
            Tensor::eye(input_dim, DType::F32, &DEV)?.unsqueeze(2)?
        } else {
            Tensor::randn(0f32, 0.1, (codebook_dim, input_dim, 1), &DEV)?
        };
        tensors.insert(format!("{prefix}input_proj.weight"), input_w);

        // output_proj.weight: [input_dim, codebook_dim, 1]
        let output_w = if input_dim == codebook_dim {
            Tensor::eye(input_dim, DType::F32, &DEV)?.unsqueeze(2)?
        } else {
            Tensor::randn(0f32, 0.1, (input_dim, codebook_dim, 1), &DEV)?
        };
        tensors.insert(format!("{prefix}output_proj.weight"), output_w);

        // Codebook tensors for each layer
        for i in 0..n_codebooks {
            // Use well-separated codebook entries for predictable behaviour
            let embed_sum =
                Tensor::randn(0f32, 1.0, (codebook_bins, codebook_dim), &DEV)?;
            let cluster_usage = Tensor::ones((codebook_bins,), DType::F32, &DEV)?;
            tensors.insert(
                format!("{prefix}layers.{i}.codebook.embed_sum"),
                embed_sum,
            );
            tensors.insert(
                format!("{prefix}layers.{i}.codebook.cluster_usage"),
                cluster_usage,
            );
        }

        Ok(tensors)
    }

    /// Build a ResidualVectorQuantizer from synthetic weights and verify the output
    /// shape is [B, n_codebooks, T].
    #[test]
    fn rvq_encode_output_shape() -> Result<()> {
        let n_codebooks = 3;
        let input_dim = 4;
        let codebook_dim = 4;
        let codebook_bins = 8;

        let tensors = make_rvq_tensors("", n_codebooks, input_dim, codebook_dim, codebook_bins)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &DEV);

        let rvq = ResidualVectorQuantizer::load(vb, n_codebooks, input_dim, codebook_dim, codebook_bins)?;

        let b = 1;
        let t = 5;
        let input = Tensor::randn(0f32, 1.0, (b, input_dim, t), &DEV)?;
        let codes = rvq.encode(&input)?;

        assert_eq!(codes.dims(), &[b, n_codebooks, t]);
        assert_eq!(codes.dtype(), DType::U32);

        Ok(())
    }

    /// Verify the overall residual decreases after RVQ encoding.
    /// With random codebooks, individual layers may not always decrease the residual,
    /// but the overall trend should reduce energy.
    #[test]
    fn rvq_residual_decreases() -> Result<()> {
        let n_codebooks = 4;
        let dim = 4; // input_dim == codebook_dim so identity projection
        let codebook_bins = 16;

        let tensors = make_rvq_tensors("", n_codebooks, dim, dim, codebook_bins)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &DEV);

        let rvq = ResidualVectorQuantizer::load(vb, n_codebooks, dim, dim, codebook_bins)?;

        let input = Tensor::randn(0f32, 1.0, (1, dim, 10), &DEV)?;

        // Project through input_proj (identity in this case)
        let projected = rvq.input_proj.forward(&input)?;

        let initial_norm: f32 = projected.sqr()?.sum_all()?.to_scalar()?;

        let mut residual = projected;
        for quantizer in &rvq.quantizers {
            let (quantized, _indices) = quantizer.encode(&residual)?;
            residual = residual.sub(&quantized)?;
        }

        let final_norm: f32 = residual.sqr()?.sum_all()?.to_scalar()?;

        // The final residual should be strictly less than the initial
        assert!(
            final_norm < initial_norm,
            "final residual ({:.6}) should be less than initial ({:.6})",
            final_norm,
            initial_norm,
        );

        Ok(())
    }

    /// Verify that the RVQ encodes correctly with batch size > 1.
    #[test]
    fn rvq_encode_batched() -> Result<()> {
        let n_codebooks = 2;
        let input_dim = 4;
        let codebook_dim = 4;
        let codebook_bins = 8;

        let tensors = make_rvq_tensors("", n_codebooks, input_dim, codebook_dim, codebook_bins)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &DEV);

        let rvq = ResidualVectorQuantizer::load(vb, n_codebooks, input_dim, codebook_dim, codebook_bins)?;

        let b = 3;
        let t = 6;
        let input = Tensor::randn(0f32, 1.0, (b, input_dim, t), &DEV)?;
        let codes = rvq.encode(&input)?;

        assert_eq!(codes.dims(), &[b, n_codebooks, t]);

        // All indices should be valid
        let idx_vec: Vec<u32> = codes.flatten_all()?.to_vec1()?;
        for &idx in &idx_vec {
            assert!(idx < codebook_bins as u32);
        }

        Ok(())
    }

    // ---- SplitResidualVectorQuantizer tests ----

    /// Verify SplitResidualVectorQuantizer produces [B, n_q_semantic + n_q_acoustic, T].
    #[test]
    fn split_rvq_encode_output_shape() -> Result<()> {
        let n_q_semantic = 1;
        let n_q_acoustic = 3;
        let input_dim = 4;
        let codebook_dim = 4;
        let codebook_bins = 8;

        let mut tensors = HashMap::new();
        let sem = make_rvq_tensors(
            "semantic_residual_vector_quantizer.",
            n_q_semantic, input_dim, codebook_dim, codebook_bins,
        )?;
        let aco = make_rvq_tensors(
            "acoustic_residual_vector_quantizer.",
            n_q_acoustic, input_dim, codebook_dim, codebook_bins,
        )?;
        tensors.extend(sem);
        tensors.extend(aco);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &DEV);
        let split = SplitResidualVectorQuantizer::load(
            vb, n_q_semantic, n_q_acoustic, input_dim, codebook_dim, codebook_bins,
        )?;

        assert_eq!(split.n_q, n_q_semantic + n_q_acoustic);

        let b = 1;
        let t = 5;
        let input = Tensor::randn(0f32, 1.0, (b, input_dim, t), &DEV)?;
        let codes = split.encode(&input)?;

        assert_eq!(codes.dims(), &[b, n_q_semantic + n_q_acoustic, t]);
        assert_eq!(codes.dtype(), DType::U32);

        Ok(())
    }

    // ---- Decode round-trip tests ----

    /// VectorQuantizer::decode produces [B, dim, T] from [B, 1, T] u32 indices.
    #[test]
    fn vq_decode_output_shape() -> Result<()> {
        let n_bins = 8usize;
        let dim = 5usize;
        let codebook = Tensor::randn(0f32, 1.0, (n_bins, dim), &DEV)?;
        let vq = VectorQuantizer::new(codebook)?;

        let b = 2usize;
        let t = 7usize;
        // Random valid indices in [0, n_bins)
        let raw: Vec<u32> = (0..(b * t) as u32).map(|i| i % n_bins as u32).collect();
        let indices = Tensor::from_vec(raw, (b, 1, t), &DEV)?;

        let decoded = vq.decode(&indices)?;
        assert_eq!(decoded.dims(), &[b, dim, t]);

        Ok(())
    }

    /// ResidualVectorQuantizer::decode produces [B, input_dim, T] from [B, n_q, T] u32 codes.
    #[test]
    fn rvq_decode_output_shape() -> Result<()> {
        let n_codebooks = 3usize;
        let input_dim = 4usize;
        let codebook_dim = 4usize;
        let codebook_bins = 8usize;

        let tensors = make_rvq_tensors("", n_codebooks, input_dim, codebook_dim, codebook_bins)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &DEV);
        let rvq = ResidualVectorQuantizer::load(vb, n_codebooks, input_dim, codebook_dim, codebook_bins)?;

        let b = 2usize;
        let t = 10usize;
        let raw: Vec<u32> = (0..(b * n_codebooks * t) as u32)
            .map(|i| i % codebook_bins as u32)
            .collect();
        let codes = Tensor::from_vec(raw, (b, n_codebooks, t), &DEV)?;

        let decoded = rvq.decode(&codes)?;
        assert_eq!(decoded.dims(), &[b, input_dim, t]);

        Ok(())
    }

    /// ResidualVectorQuantizer::decode_n with n < n_codebooks still produces [B, input_dim, T].
    #[test]
    fn rvq_decode_n_partial() -> Result<()> {
        let n_codebooks = 4usize;
        let input_dim = 4usize;
        let codebook_dim = 4usize;
        let codebook_bins = 8usize;

        let tensors = make_rvq_tensors("", n_codebooks, input_dim, codebook_dim, codebook_bins)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &DEV);
        let rvq = ResidualVectorQuantizer::load(vb, n_codebooks, input_dim, codebook_dim, codebook_bins)?;

        let b = 1usize;
        let t = 5usize;
        // Only supply 2 codebooks worth of codes
        let n_partial = 2usize;
        let raw: Vec<u32> = (0..(b * n_partial * t) as u32)
            .map(|i| i % codebook_bins as u32)
            .collect();
        let codes = Tensor::from_vec(raw, (b, n_partial, t), &DEV)?;

        let decoded = rvq.decode_n(&codes, n_partial)?;
        assert_eq!(decoded.dims(), &[b, input_dim, t]);

        Ok(())
    }

    /// SplitResidualVectorQuantizer::decode produces [B, input_dim, T].
    #[test]
    fn split_rvq_decode_output_shape() -> Result<()> {
        let n_q_semantic = 1usize;
        let n_q_acoustic = 3usize;
        let n_q = n_q_semantic + n_q_acoustic;
        let input_dim = 4usize;
        let codebook_dim = 4usize;
        let codebook_bins = 8usize;

        let mut tensors = HashMap::new();
        tensors.extend(make_rvq_tensors(
            "semantic_residual_vector_quantizer.",
            n_q_semantic, input_dim, codebook_dim, codebook_bins,
        )?);
        tensors.extend(make_rvq_tensors(
            "acoustic_residual_vector_quantizer.",
            n_q_acoustic, input_dim, codebook_dim, codebook_bins,
        )?);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &DEV);
        let split = SplitResidualVectorQuantizer::load(
            vb, n_q_semantic, n_q_acoustic, input_dim, codebook_dim, codebook_bins,
        )?;

        let b = 1usize;
        let t = 10usize;
        let raw: Vec<u32> = (0..(b * n_q * t) as u32)
            .map(|i| i % codebook_bins as u32)
            .collect();
        let codes = Tensor::from_vec(raw, (b, n_q, t), &DEV)?;

        let decoded = split.decode(&codes)?;
        assert_eq!(decoded.dims(), &[b, input_dim, t]);

        Ok(())
    }

    /// SplitResidualVectorQuantizer::decode_n with n_total <= n_q_semantic (only first sub-RVQ).
    #[test]
    fn split_rvq_decode_n_semantic_only() -> Result<()> {
        let n_q_semantic = 1usize;
        let n_q_acoustic = 3usize;
        let input_dim = 4usize;
        let codebook_dim = 4usize;
        let codebook_bins = 8usize;

        let mut tensors = HashMap::new();
        tensors.extend(make_rvq_tensors(
            "semantic_residual_vector_quantizer.",
            n_q_semantic, input_dim, codebook_dim, codebook_bins,
        )?);
        tensors.extend(make_rvq_tensors(
            "acoustic_residual_vector_quantizer.",
            n_q_acoustic, input_dim, codebook_dim, codebook_bins,
        )?);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &DEV);
        let split = SplitResidualVectorQuantizer::load(
            vb, n_q_semantic, n_q_acoustic, input_dim, codebook_dim, codebook_bins,
        )?;

        let b = 1usize;
        let t = 8usize;
        // Only 1 codebook (semantic only)
        let raw: Vec<u32> = (0..(b * n_q_semantic * t) as u32)
            .map(|i| i % codebook_bins as u32)
            .collect();
        let codes = Tensor::from_vec(raw, (b, n_q_semantic, t), &DEV)?;

        let decoded = split.decode_n(&codes, n_q_semantic)?;
        assert_eq!(decoded.dims(), &[b, input_dim, t]);

        Ok(())
    }
}

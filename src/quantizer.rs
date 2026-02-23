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
}

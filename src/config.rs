/// Configuration for the Mimi audio codec.
pub struct MimiConfig {
    pub channels: usize,
    pub sample_rate: usize,
    pub frame_rate: usize,
    pub dimension: usize,
    pub quantizer_dimension: usize,
    pub quantizer_output_dimension: usize,
    /// Number of RVQ codebooks (0 = DummyQuantizer for TTS, 32 = SplitRVQ for full Mimi).
    pub num_codebooks: usize,
    /// Number of bins per codebook (e.g. 2048).
    pub codebook_bins: usize,
    /// Internal codebook dimension (e.g. 256).
    pub codebook_dim: usize,
    /// Number of semantic codebooks in SplitRVQ (typically 1).
    pub num_codebooks_semantic: usize,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub residual_kernel_size: usize,
    pub dilation_base: usize,
    pub compress: usize,
    pub transformer_d_model: usize,
    pub transformer_num_heads: usize,
    pub transformer_num_layers: usize,
    pub transformer_layer_scale: f64,
    pub transformer_context: usize,
    pub transformer_max_period: f64,
    pub transformer_dim_feedforward: usize,
}

impl MimiConfig {
    /// Returns the Mimi config for the v202601 TTS model checkpoint (pocket-tts).
    ///
    /// Uses DummyQuantizer (output projection only, no RVQ codebooks).
    pub fn v202601() -> Self {
        Self {
            channels: 1,
            sample_rate: 24000,
            frame_rate: 12,
            dimension: 512,
            quantizer_dimension: 32,
            quantizer_output_dimension: 512,
            num_codebooks: 0,
            codebook_bins: 0,
            codebook_dim: 0,
            num_codebooks_semantic: 0,
            n_filters: 64,
            n_residual_layers: 1,
            ratios: vec![6, 5, 4],
            kernel_size: 7,
            last_kernel_size: 3,
            residual_kernel_size: 3,
            dilation_base: 2,
            compress: 2,
            transformer_d_model: 512,
            transformer_num_heads: 8,
            transformer_num_layers: 2,
            transformer_layer_scale: 0.01,
            transformer_context: 250,
            transformer_max_period: 10000.0,
            transformer_dim_feedforward: 2048,
        }
    }

    /// Returns the config for the full Mimi v1.0.0 codec model.
    ///
    /// Uses SplitRVQ: 1 semantic + 31 acoustic codebooks.
    pub fn mimi_v1_0_0() -> Self {
        Self {
            channels: 1,
            sample_rate: 24000,
            // True frame rate is 12.5 Hz; use 12 as integer approximation.
            // Downsample stride computation: encoder_frame_rate=25, 25/12 = 2 (correct).
            frame_rate: 12,
            dimension: 512,
            quantizer_dimension: 32,
            quantizer_output_dimension: 512,
            num_codebooks: 32,
            codebook_bins: 2048,
            codebook_dim: 256,
            num_codebooks_semantic: 1,
            n_filters: 64,
            n_residual_layers: 1,
            ratios: vec![8, 6, 5, 4],
            kernel_size: 7,
            last_kernel_size: 3,
            residual_kernel_size: 3,
            dilation_base: 2,
            compress: 2,
            transformer_d_model: 512,
            transformer_num_heads: 8,
            transformer_num_layers: 8,
            transformer_layer_scale: 0.01,
            transformer_context: 250,
            transformer_max_period: 10000.0,
            transformer_dim_feedforward: 2048,
        }
    }
}

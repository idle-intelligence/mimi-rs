/// Configuration for the Mimi audio codec.
pub struct MimiConfig {
    pub channels: usize,
    pub sample_rate: usize,
    pub frame_rate: usize,
    pub dimension: usize,
    pub quantizer_dimension: usize,
    pub quantizer_output_dimension: usize,
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
    /// Returns the default Mimi config for the v202601 model checkpoint.
    pub fn v202601() -> Self {
        Self {
            channels: 1,
            sample_rate: 24000,
            frame_rate: 12,
            dimension: 512,
            quantizer_dimension: 32,
            quantizer_output_dimension: 512,
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
}

use crate::gguf_loader::GgufTensors;
use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};

/// Simplified quantizer that only provides output projection for TTS.
pub struct DummyQuantizer {
    output_proj: Conv1d,
    pub dimension: usize,
    pub output_dimension: usize,
}

impl DummyQuantizer {
    pub fn load(vb: VarBuilder, dimension: usize, output_dimension: usize) -> Result<Self> {
        let vb = vb.pp("output_proj");
        let cfg = Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1, ..Default::default() };
        // kernel_size=1 conv with no bias
        let has_bias = vb.contains_tensor("bias");
        let output_proj = if has_bias {
            candle_nn::conv1d(dimension, output_dimension, 1, cfg, vb)?
        } else {
            candle_nn::conv1d_no_bias(dimension, output_dimension, 1, cfg, vb)?
        };
        Ok(Self { output_proj, dimension, output_dimension })
    }

    pub fn load_gguf(
        gguf: &mut GgufTensors,
        prefix: &str,
        dimension: usize,
        output_dimension: usize,
    ) -> Result<Self> {
        let cfg = Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1, ..Default::default() };
        let output_proj = gguf.conv1d(&format!("{prefix}.output_proj"), dimension, output_dimension, 1, cfg)?;
        Ok(Self { output_proj, dimension, output_dimension })
    }

    /// Forward pass: Conv1d with kernel_size=1.
    /// Input: [B, dimension, T] -> Output: [B, output_dimension, T]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.output_proj.forward(x)
    }
}

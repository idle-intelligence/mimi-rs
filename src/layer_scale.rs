use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

pub struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    /// Construct from a pre-loaded scale tensor (e.g. from GGUF).
    pub fn new(scale: Tensor) -> Self {
        Self { scale }
    }

    pub fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        let scale = vb.get((channels,), "scale")?;
        Ok(Self { scale })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.scale)
    }
}

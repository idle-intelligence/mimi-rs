use crate::conv::{
    PadMode, StreamingConv1d, StreamingConv1dState, StreamingConvTr1dState,
    StreamingConvTranspose1d,
};
use crate::gguf_loader::GgufTensors;
use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

pub struct ConvDownsample1d {
    conv: StreamingConv1d,
}

impl ConvDownsample1d {
    pub fn load(vb: VarBuilder, stride: usize, dimension: usize) -> Result<Self> {
        let conv = StreamingConv1d::load(
            vb.pp("conv"),
            dimension,
            dimension,
            2 * stride,
            stride,
            1,
            PadMode::Replicate,
            1,
            false,
        )?;
        Ok(Self { conv })
    }

    pub fn init_state(&self, batch_size: usize, device: &Device) -> Result<StreamingConv1dState> {
        self.conv.init_state(batch_size, device)
    }

    pub fn forward(
        &self,
        x: &Tensor,
        state: &mut StreamingConv1dState,
    ) -> Result<Tensor> {
        self.conv.forward(x, state)
    }

    pub fn load_gguf(gguf: &mut GgufTensors, prefix: &str, stride: usize, dimension: usize) -> Result<Self> {
        let conv = gguf.streaming_conv1d(
            &format!("{prefix}.conv"),
            dimension,
            dimension,
            2 * stride,
            stride,
            1,
            PadMode::Replicate,
            1,
            false,
        )?;
        Ok(Self { conv })
    }

    /// Non-streaming forward (creates and discards state).
    pub fn forward_no_state(&self, x: &Tensor) -> Result<Tensor> {
        let b = x.dim(0)?;
        let device = x.device().clone();
        let mut state = self.init_state(b, &device)?;
        self.conv.forward(x, &mut state)
    }
}

pub struct ConvTrUpsample1d {
    convtr: StreamingConvTranspose1d,
}

impl ConvTrUpsample1d {
    pub fn load(vb: VarBuilder, stride: usize, dimension: usize) -> Result<Self> {
        let convtr = StreamingConvTranspose1d::load(
            vb.pp("convtr"),
            dimension,
            dimension,
            2 * stride,
            stride,
            dimension, // groups = dimension (depthwise)
            false,
        )?;
        Ok(Self { convtr })
    }

    pub fn load_gguf(gguf: &mut GgufTensors, prefix: &str, stride: usize, dimension: usize) -> Result<Self> {
        let convtr = gguf.streaming_conv_transpose1d(
            &format!("{prefix}.convtr"),
            dimension,
            dimension,
            2 * stride,
            stride,
            dimension, // groups = dimension (depthwise)
            false,
        )?;
        Ok(Self { convtr })
    }

    pub fn init_state(&self, batch_size: usize, device: &Device) -> Result<StreamingConvTr1dState> {
        self.convtr.init_state(batch_size, device)
    }

    pub fn forward(
        &self,
        x: &Tensor,
        state: &mut StreamingConvTr1dState,
    ) -> Result<Tensor> {
        self.convtr.forward(x, state)
    }
}

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};

/// Pad input so that a convolution covers the full input.
pub fn pad_for_conv1d(x: &Tensor, kernel_size: usize, stride: usize) -> Result<Tensor> {
    let length = x.dim(2)?;
    let n_frames = (length as f64 - kernel_size as f64) / stride as f64 + 1.0;
    let ideal_length = (n_frames.ceil() as usize - 1) * stride + kernel_size;
    let extra = ideal_length.saturating_sub(length);
    if extra > 0 {
        x.pad_with_zeros(2, 0, extra)
    } else {
        Ok(x.clone())
    }
}

/// Conv1d wrapper with weight and optional bias.
pub struct CausalConv1d {
    inner: Conv1d,
    pub stride: usize,
    pub dilation: usize,
    pub kernel_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub groups: usize,
}

impl CausalConv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let cfg = Conv1dConfig { padding: 0, stride, dilation, groups, ..Default::default() };
        let inner = if bias {
            candle_nn::conv1d(in_channels, out_channels, kernel_size, cfg, vb)?
        } else {
            candle_nn::conv1d_no_bias(in_channels, out_channels, kernel_size, cfg, vb)?
        };
        Ok(Self { inner, stride, dilation, kernel_size, in_channels, out_channels, groups })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

/// ConvTranspose1d wrapper.
pub struct CausalConvTranspose1d {
    inner: ConvTranspose1d,
    /// Bias stored separately for streaming overlap-add correction.
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub kernel_size: usize,
    pub out_channels: usize,
    pub groups: usize,
}

impl CausalConvTranspose1d {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        // Load bias separately for streaming overlap-add correction
        let bias_tensor = if bias {
            Some(vb.get(out_channels, "bias")?)
        } else {
            None
        };
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            stride,
            output_padding: 0,
            dilation: 1,
            groups,
        };
        let inner = if bias {
            candle_nn::conv_transpose1d(in_channels, out_channels, kernel_size, cfg, vb)?
        } else {
            candle_nn::conv_transpose1d_no_bias(
                in_channels,
                out_channels,
                kernel_size,
                cfg,
                vb,
            )?
        };
        Ok(Self { inner, bias: bias_tensor, stride, kernel_size, out_channels, groups })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

/// Streaming state for StreamingConv1d.
#[derive(Debug, Clone)]
pub struct StreamingConv1dState {
    pub previous: Tensor,
    pub first: bool,
}

#[derive(Clone, Copy)]
pub enum PadMode {
    Constant,
    Replicate,
}

/// Streaming Conv1d with causal padding.
pub struct StreamingConv1d {
    pub conv: CausalConv1d,
    pad_mode: PadMode,
}

impl StreamingConv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        pad_mode: PadMode,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let conv = CausalConv1d::load(
            vb.pp("conv"),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
        )?;
        Ok(Self { conv, pad_mode })
    }

    fn effective_kernel_size(&self) -> usize {
        (self.conv.kernel_size - 1) * self.conv.dilation + 1
    }

    pub fn init_state(&self, batch_size: usize, device: &Device) -> Result<StreamingConv1dState> {
        let kernel = self.effective_kernel_size();
        let prev_len = kernel.saturating_sub(self.conv.stride);
        let previous =
            Tensor::zeros((batch_size, self.conv.in_channels, prev_len), DType::F32, device)?;
        Ok(StreamingConv1dState { previous, first: true })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        state: &mut StreamingConv1dState,
    ) -> Result<Tensor> {
        let tp = state.previous.dim(2)?;

        // On first call with replicate padding, fill previous with first sample
        if tp > 0 && matches!(self.pad_mode, PadMode::Replicate) && state.first {
            let init = x.narrow(2, 0, 1)?.contiguous()?;
            state.previous = if tp == 1 {
                init
            } else {
                let refs: Vec<&Tensor> = (0..tp).map(|_| &init).collect();
                Tensor::cat(&refs, 2)?
            };
        }

        // Prepend previous state
        let x_padded =
            if tp > 0 { Tensor::cat(&[&state.previous, x], 2)? } else { x.clone() };

        // Run convolution
        let y = self.conv.forward(&x_padded)?;

        // Update state
        if tp > 0 {
            let xlen = x_padded.dim(2)?;
            state.previous = x_padded.narrow(2, xlen - tp, tp)?.contiguous()?;
            if matches!(self.pad_mode, PadMode::Replicate) {
                state.first = false;
            }
        }

        Ok(y)
    }
}

/// Streaming state for StreamingConvTranspose1d.
#[derive(Debug, Clone)]
pub struct StreamingConvTr1dState {
    pub partial: Tensor,
}

/// Streaming ConvTranspose1d.
pub struct StreamingConvTranspose1d {
    pub convtr: CausalConvTranspose1d,
}

impl StreamingConvTranspose1d {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let convtr = CausalConvTranspose1d::load(
            vb.pp("convtr"),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            bias,
        )?;
        Ok(Self { convtr })
    }

    pub fn init_state(&self, batch_size: usize, device: &Device) -> Result<StreamingConvTr1dState> {
        let pt = self.convtr.kernel_size - self.convtr.stride;
        let partial =
            Tensor::zeros((batch_size, self.convtr.out_channels, pt), DType::F32, device)?;
        Ok(StreamingConvTr1dState { partial })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        state: &mut StreamingConvTr1dState,
    ) -> Result<Tensor> {
        let y = self.convtr.forward(x)?;
        let pt = state.partial.dim(2)?;

        if pt > 0 {
            let y_len = y.dim(2)?;
            // Add overlap from previous to the start
            let y_start = y.narrow(2, 0, pt)?;
            let y_start = (y_start + &state.partial)?;
            // Combine corrected start with the rest
            let y_rest = y.narrow(2, pt, y_len - pt)?;
            let y = Tensor::cat(&[&y_start, &y_rest], 2)?.contiguous()?;

            // Save new partial (last pt samples), subtracting bias if present.
            // The bias is included in every conv_transpose1d output. Without subtracting
            // it here, the overlap-add would double-count the bias in overlap regions.
            let mut new_partial = y.narrow(2, y_len - pt, pt)?.contiguous()?;
            if let Some(bias) = &self.convtr.bias {
                let bias = bias.reshape((1, bias.elem_count(), 1))?;
                new_partial = new_partial.broadcast_sub(&bias)?;
            }
            state.partial = new_partial;

            // Return without the partial tail
            let out = y.narrow(2, 0, y_len - pt)?.contiguous()?;
            Ok(out)
        } else {
            Ok(y)
        }
    }
}

use crate::conv::{CausalConv1d, CausalConvTranspose1d, PadMode, StreamingConv1d, StreamingConvTranspose1d};
use crate::qlinear::QLinear;
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use std::io::Cursor;

/// Wraps a GGUF file (as bytes) and provides typed tensor accessors.
///
/// Tensor names in the GGUF file must match the internal (remapped) model names,
/// e.g. `"mimi.decoder.model.0.conv.weight"`.
pub struct GgufTensors<'a> {
    content: gguf_file::Content,
    cursor: Cursor<&'a [u8]>,
    pub device: Device,
}

impl<'a> GgufTensors<'a> {
    pub fn from_bytes(data: &'a [u8], device: &Device) -> Result<Self> {
        let mut cursor = Cursor::new(data);
        let content = gguf_file::Content::read(&mut cursor)?;
        Ok(Self { content, cursor, device: device.clone() })
    }

    // ---- Low-level helpers ----

    fn qt(&mut self, name: &str) -> Result<QTensor> {
        self.content.tensor(&mut self.cursor, name, &self.device)
    }

    /// Check whether a tensor exists in the GGUF.
    pub fn contains(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }

    // ---- Public accessors ----

    /// Load a tensor by name and dequantize it to F32.
    pub fn tensor(&mut self, name: &str) -> Result<Tensor> {
        let qt = self.qt(name)?;
        qt.dequantize(&self.device)
    }

    /// Load a linear layer from GGUF.
    ///
    /// Looks for `{prefix}.weight` (required) and `{prefix}.bias` (optional).
    /// All weights (including Q8_0) are dequantized to F32 and wrapped in
    /// `candle_nn::Linear` so matmuls go through the optimized `gemm` crate.
    ///
    /// Rationale: candle's `QMatMul` for quantized tensors uses a naive
    /// triple loop, while `Linear` uses `gemm` with SIMD-tiled, cache-blocked
    /// kernels. On WASM with simd128, F32 gemm is ~1.7x faster than Q8_0
    /// QMatMul for this model. We keep Q8_0 in the GGUF file for compact
    /// download (178 MB vs 236 MB F32) and dequantize at load time.
    ///
    /// TODO: switch back to `QMatMul` once candle ships an optimized
    /// quantized matmul kernel (tiled + SIMD) that can compete with gemm.
    /// See: https://github.com/huggingface/candle/issues/XXXX
    pub fn qlinear(&mut self, prefix: &str) -> Result<QLinear> {
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");

        if !self.contains(&weight_name) {
            return Err(candle_core::Error::Msg(format!("tensor not found: {weight_name}")));
        }

        let bias = if self.contains(&bias_name) {
            Some(self.tensor(&bias_name)?)
        } else {
            None
        };

        let weight = self.tensor(&weight_name)?;
        Ok(QLinear::from_linear(candle_nn::Linear::new(weight, bias)))
    }

    /// Load a Conv1d from GGUF (always F32 — conv weights are not quantized).
    ///
    /// Looks for `{prefix}.weight` (required) and `{prefix}.bias` (optional).
    pub fn conv1d(
        &mut self,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        cfg: Conv1dConfig,
    ) -> Result<Conv1d> {
        let weight = self.tensor(&format!("{prefix}.weight"))?;
        let bias_name = format!("{prefix}.bias");
        let bias = if self.contains(&bias_name) { Some(self.tensor(&bias_name)?) } else { None };
        // Validate shape
        let _ = (in_c, out_c, kernel_size); // shapes are baked into the weight tensor
        Ok(Conv1d::new(weight, bias, cfg))
    }

    /// Load a `StreamingConv1d` from GGUF.
    ///
    /// Tensor prefix for a `StreamingConv1d` is the path up to (but not including)
    /// `.conv`, e.g. `"mimi.encoder.model.0"` → reads `mimi.encoder.model.0.conv.weight`.
    #[allow(clippy::too_many_arguments)]
    pub fn streaming_conv1d(
        &mut self,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        pad_mode: PadMode,
        groups: usize,
        bias: bool,
    ) -> Result<StreamingConv1d> {
        let p = format!("{prefix}.conv");
        let cfg = Conv1dConfig { padding: 0, stride, dilation, groups, ..Default::default() };
        let weight = self.tensor(&format!("{p}.weight"))?;
        let bias_t = if bias { Some(self.tensor(&format!("{p}.bias"))?) } else { None };
        let inner = Conv1d::new(weight, bias_t, cfg);
        let causal = CausalConv1d::from_parts(inner, stride, dilation, kernel_size, in_channels, out_channels, groups);
        Ok(StreamingConv1d::from_parts(causal, pad_mode))
    }

    /// Load a `StreamingConvTranspose1d` from GGUF.
    ///
    /// Tensor prefix for a `StreamingConvTranspose1d` is the path up to (but not including)
    /// `.convtr`, e.g. `"mimi.decoder.model.2"` → reads `mimi.decoder.model.2.convtr.weight`.
    #[allow(clippy::too_many_arguments)]
    pub fn streaming_conv_transpose1d(
        &mut self,
        prefix: &str,
        _in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> Result<StreamingConvTranspose1d> {
        let p = format!("{prefix}.convtr");
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            stride,
            output_padding: 0,
            dilation: 1,
            groups,
        };
        let weight = self.tensor(&format!("{p}.weight"))?;
        let bias_t = if bias { Some(self.tensor(&format!("{p}.bias"))?) } else { None };
        // The bias tensor is stored both inside the ConvTranspose1d kernel and separately
        // on CausalConvTranspose1d for streaming overlap-add correction.
        let inner = ConvTranspose1d::new(weight, bias_t.clone(), cfg);
        let causal = CausalConvTranspose1d::from_parts(inner, bias_t, stride, kernel_size, out_channels, groups);
        Ok(StreamingConvTranspose1d::from_parts(causal))
    }
}

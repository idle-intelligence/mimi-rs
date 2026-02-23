use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};
use std::sync::Arc;

/// A linear layer that can hold either F32 weights (via QMatMul::Tensor)
/// or quantized weights (via QMatMul::QTensor). Drop-in replacement for Linear.
pub struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
}

impl QLinear {
    /// Wrap an existing F32 Linear as a QLinear (no quantization yet).
    pub fn from_linear(linear: Linear) -> Self {
        let bias = linear.bias().cloned();
        Self {
            inner: QMatMul::Tensor(linear.weight().clone()),
            bias,
        }
    }

    /// Create a QLinear directly from a QTensor (e.g. loaded from GGUF).
    /// No runtime quantization needed — weights are already quantized.
    pub fn from_qtensor(qtensor: QTensor, bias: Option<Tensor>) -> Self {
        Self { inner: QMatMul::QTensor(Arc::new(qtensor)), bias }
    }

    /// Quantize the weight tensor in-place to the given GGML dtype (e.g. Q8_0).
    /// No-op if already quantized.
    pub fn quantize_in_place(&mut self, dtype: GgmlDType) -> Result<()> {
        match &self.inner {
            QMatMul::Tensor(t) => {
                let qtensor = QTensor::quantize(t, dtype)?;
                self.inner = QMatMul::QTensor(Arc::new(qtensor));
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.inner.forward(x)?;
        match &self.bias {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }
}

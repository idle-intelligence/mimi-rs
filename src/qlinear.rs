use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};
use std::sync::Arc;

/// A linear layer that can hold either F32 weights (via `candle_nn::Linear`,
/// using the optimized `gemm` matmul) or quantized weights (via `QMatMul`).
///
/// Important: the F32 path uses `Linear` directly, NOT `QMatMul::Tensor`,
/// because `QMatMul::Tensor::forward` bypasses `gemm` and is much slower.
enum Inner {
    Linear(Linear),
    Quantized { qmatmul: QMatMul, bias: Option<Tensor> },
}

pub struct QLinear {
    inner: Inner,
}

impl QLinear {
    /// Wrap an existing F32 Linear — matmuls go through optimized `gemm`.
    pub fn from_linear(linear: Linear) -> Self {
        Self { inner: Inner::Linear(linear) }
    }

    /// Create from a QTensor (e.g. loaded from GGUF).
    /// Matmuls go through candle's quantized kernel.
    pub fn from_qtensor(qtensor: QTensor, bias: Option<Tensor>) -> Self {
        Self {
            inner: Inner::Quantized {
                qmatmul: QMatMul::QTensor(Arc::new(qtensor)),
                bias,
            },
        }
    }

    /// Quantize F32 weights in-place to the given GGML dtype (e.g. Q8_0).
    /// No-op if already quantized.
    pub fn quantize_in_place(&mut self, dtype: GgmlDType) -> Result<()> {
        if let Inner::Linear(linear) = &self.inner {
            let qtensor = QTensor::quantize(linear.weight(), dtype)?;
            self.inner = Inner::Quantized {
                qmatmul: QMatMul::QTensor(Arc::new(qtensor)),
                bias: linear.bias().cloned(),
            };
        }
        Ok(())
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.inner {
            Inner::Linear(linear) => linear.forward(x),
            Inner::Quantized { qmatmul, bias } => {
                let out = qmatmul.forward(x)?;
                match bias {
                    Some(b) => out.broadcast_add(b),
                    None => Ok(out),
                }
            }
        }
    }
}

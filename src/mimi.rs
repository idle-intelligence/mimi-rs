use crate::config::MimiConfig;
use crate::conv::{pad_for_conv1d, PadMode, StreamingConv1dState, StreamingConvTr1dState};
use crate::dummy_quantizer::DummyQuantizer;
use crate::gguf_loader::GgufTensors;
use crate::quantizer::SplitResidualVectorQuantizer;
use crate::resample::{ConvDownsample1d, ConvTrUpsample1d};
use crate::seanet::{SEANetDecoder, SEANetDecoderState, SEANetEncoder, SEANetEncoderState};
use crate::transformer::{ProjectedTransformer, StreamingTransformerState};
use candle_core::quantized::GgmlDType;
use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

/// Either a DummyQuantizer (TTS, output projection only) or a full SplitRVQ (encoder).
pub enum QuantizerKind {
    Dummy(DummyQuantizer),
    SplitRvq(SplitResidualVectorQuantizer),
}

pub struct MimiModel {
    encoder: Option<SEANetEncoder>,
    decoder: Option<SEANetDecoder>,
    encoder_transformer: Option<ProjectedTransformer>,
    decoder_transformer: Option<ProjectedTransformer>,
    pub quantizer: QuantizerKind,
    downsample: Option<ConvDownsample1d>,
    upsample: Option<ConvTrUpsample1d>,
    frame_rate: usize,
    _encoder_frame_rate: f64,
    pub sample_rate: usize,
    _dimension: usize,
}

/// Decoder-only streaming state (used by TTS).
#[derive(Debug, Clone)]
pub struct MimiState {
    _encoder_state: Option<SEANetEncoderState>,
    decoder_state: SEANetDecoderState,
    _encoder_transformer_state: Option<StreamingTransformerState>,
    decoder_transformer_state: StreamingTransformerState,
    _downsample_state: Option<StreamingConv1dState>,
    upsample_state: Option<StreamingConvTr1dState>,
}

/// Encoder-only streaming state (used by STT).
#[derive(Debug, Clone)]
pub struct MimiEncoderState {
    pub encoder_state: SEANetEncoderState,
    pub encoder_transformer_state: StreamingTransformerState,
    pub downsample_state: Option<StreamingConv1dState>,
}

impl MimiModel {
    pub fn load(vb: VarBuilder, cfg: &MimiConfig) -> Result<Self> {
        let pad_mode = PadMode::Constant;

        let encoder = SEANetEncoder::load(
            vb.pp("encoder"),
            cfg.channels,
            cfg.dimension,
            cfg.n_filters,
            cfg.n_residual_layers,
            &cfg.ratios,
            cfg.kernel_size,
            cfg.last_kernel_size,
            cfg.residual_kernel_size,
            cfg.dilation_base,
            pad_mode,
            cfg.compress,
        )?;

        let decoder = SEANetDecoder::load(
            vb.pp("decoder"),
            cfg.channels,
            cfg.dimension,
            cfg.n_filters,
            cfg.n_residual_layers,
            &cfg.ratios,
            cfg.kernel_size,
            cfg.last_kernel_size,
            cfg.residual_kernel_size,
            cfg.dilation_base,
            pad_mode,
            cfg.compress,
        )?;

        let output_dimensions = vec![cfg.dimension];
        let encoder_transformer = ProjectedTransformer::load(
            vb.pp("encoder_transformer"),
            cfg.dimension,
            &output_dimensions,
            cfg.transformer_d_model,
            cfg.transformer_num_heads,
            cfg.transformer_num_layers,
            Some(cfg.transformer_layer_scale),
            cfg.transformer_context,
            cfg.transformer_max_period,
            cfg.transformer_dim_feedforward,
        )?;

        let decoder_transformer = ProjectedTransformer::load(
            vb.pp("decoder_transformer"),
            cfg.dimension,
            &output_dimensions,
            cfg.transformer_d_model,
            cfg.transformer_num_heads,
            cfg.transformer_num_layers,
            Some(cfg.transformer_layer_scale),
            cfg.transformer_context,
            cfg.transformer_max_period,
            cfg.transformer_dim_feedforward,
        )?;

        // Load quantizer based on config: SplitRVQ if num_codebooks > 0, else DummyQuantizer.
        let quantizer = if cfg.num_codebooks > 0 {
            let n_acoustic = cfg.num_codebooks - cfg.num_codebooks_semantic;
            let split_rvq = SplitResidualVectorQuantizer::load(
                vb.pp("quantizer"),
                cfg.num_codebooks_semantic,
                n_acoustic,
                cfg.dimension,
                cfg.codebook_dim,
                cfg.codebook_bins,
            )?;
            QuantizerKind::SplitRvq(split_rvq)
        } else {
            let dummy = DummyQuantizer::load(
                vb.pp("quantizer"),
                cfg.quantizer_dimension,
                cfg.quantizer_output_dimension,
            )?;
            QuantizerKind::Dummy(dummy)
        };

        let hop_length: usize = cfg.ratios.iter().product();
        let encoder_frame_rate = cfg.sample_rate as f64 / hop_length as f64;

        let (downsample, upsample) =
            if (encoder_frame_rate - cfg.frame_rate as f64).abs() > 0.01 {
                let downsample_stride = (encoder_frame_rate / cfg.frame_rate as f64) as usize;
                let ds = ConvDownsample1d::load(
                    vb.pp("downsample"),
                    downsample_stride,
                    cfg.dimension,
                )?;
                let us = ConvTrUpsample1d::load(
                    vb.pp("upsample"),
                    downsample_stride,
                    cfg.dimension,
                )?;
                (Some(ds), Some(us))
            } else {
                (None, None)
            };

        Ok(Self {
            encoder: Some(encoder),
            decoder: Some(decoder),
            encoder_transformer: Some(encoder_transformer),
            decoder_transformer: Some(decoder_transformer),
            quantizer,
            downsample,
            upsample,
            frame_rate: cfg.frame_rate,
            _encoder_frame_rate: encoder_frame_rate,
            sample_rate: cfg.sample_rate,
            _dimension: cfg.dimension,
        })
    }

    /// Load encoder-only model (no decoder weights needed).
    ///
    /// Used by STT which only needs: SEANetEncoder + encoder transformer + downsample + RVQ quantizer.
    /// Decoder/upsample fields are set to None.
    pub fn load_encoder_only(vb: VarBuilder, cfg: &MimiConfig) -> Result<Self> {
        let pad_mode = PadMode::Constant;

        let encoder = SEANetEncoder::load(
            vb.pp("encoder"),
            cfg.channels,
            cfg.dimension,
            cfg.n_filters,
            cfg.n_residual_layers,
            &cfg.ratios,
            cfg.kernel_size,
            cfg.last_kernel_size,
            cfg.residual_kernel_size,
            cfg.dilation_base,
            pad_mode,
            cfg.compress,
        )?;

        let output_dimensions = vec![cfg.dimension];
        let encoder_transformer = ProjectedTransformer::load(
            vb.pp("encoder_transformer"),
            cfg.dimension,
            &output_dimensions,
            cfg.transformer_d_model,
            cfg.transformer_num_heads,
            cfg.transformer_num_layers,
            Some(cfg.transformer_layer_scale),
            cfg.transformer_context,
            cfg.transformer_max_period,
            cfg.transformer_dim_feedforward,
        )?;

        // Load quantizer based on config
        let quantizer = if cfg.num_codebooks > 0 {
            let n_acoustic = cfg.num_codebooks - cfg.num_codebooks_semantic;
            let split_rvq = SplitResidualVectorQuantizer::load(
                vb.pp("quantizer"),
                cfg.num_codebooks_semantic,
                n_acoustic,
                cfg.dimension,
                cfg.codebook_dim,
                cfg.codebook_bins,
            )?;
            QuantizerKind::SplitRvq(split_rvq)
        } else {
            let dummy = DummyQuantizer::load(
                vb.pp("quantizer"),
                cfg.quantizer_dimension,
                cfg.quantizer_output_dimension,
            )?;
            QuantizerKind::Dummy(dummy)
        };

        let hop_length: usize = cfg.ratios.iter().product();
        let encoder_frame_rate = cfg.sample_rate as f64 / hop_length as f64;

        let downsample =
            if (encoder_frame_rate - cfg.frame_rate as f64).abs() > 0.01 {
                let downsample_stride = (encoder_frame_rate / cfg.frame_rate as f64) as usize;
                Some(ConvDownsample1d::load(
                    vb.pp("downsample"),
                    downsample_stride,
                    cfg.dimension,
                )?)
            } else {
                None
            };

        Ok(Self {
            encoder: Some(encoder),
            decoder: None,
            encoder_transformer: Some(encoder_transformer),
            decoder_transformer: None,
            quantizer,
            downsample,
            upsample: None,
            frame_rate: cfg.frame_rate,
            _encoder_frame_rate: encoder_frame_rate,
            sample_rate: cfg.sample_rate,
            _dimension: cfg.dimension,
        })
    }

    /// Load model from GGUF, skipping components whose tensors are absent.
    ///
    /// Encoder/encoder_transformer are loaded only if their tensors exist in the
    /// GGUF. This allows a TTS-only GGUF (decoder + DummyQuantizer) to skip the
    /// encoder entirely, saving ~52 MB / 28% of the file.
    pub fn load_gguf(gguf: &mut GgufTensors, prefix: &str, cfg: &MimiConfig) -> Result<Self> {
        let pad_mode = PadMode::Constant;

        // Encoder: load only if tensors are present
        let has_encoder = gguf.contains(&format!("{prefix}.encoder.model.0.conv.weight"));
        let encoder = if has_encoder {
            Some(SEANetEncoder::load_gguf(
                gguf,
                &format!("{prefix}.encoder"),
                cfg.channels,
                cfg.dimension,
                cfg.n_filters,
                cfg.n_residual_layers,
                &cfg.ratios,
                cfg.kernel_size,
                cfg.last_kernel_size,
                cfg.residual_kernel_size,
                cfg.dilation_base,
                pad_mode,
                cfg.compress,
            )?)
        } else {
            None
        };

        // Decoder: load only if tensors are present
        let has_decoder = gguf.contains(&format!("{prefix}.decoder.model.0.conv.weight"));
        let decoder = if has_decoder {
            Some(SEANetDecoder::load_gguf(
                gguf,
                &format!("{prefix}.decoder"),
                cfg.channels,
                cfg.dimension,
                cfg.n_filters,
                cfg.n_residual_layers,
                &cfg.ratios,
                cfg.kernel_size,
                cfg.last_kernel_size,
                cfg.residual_kernel_size,
                cfg.dilation_base,
                pad_mode,
                cfg.compress,
            )?)
        } else {
            None
        };

        let output_dimensions = vec![cfg.dimension];

        let has_enc_transformer = gguf.contains(
            &format!("{prefix}.encoder_transformer.transformer.layers.0.self_attn.in_proj.weight"),
        );
        let encoder_transformer = if has_enc_transformer {
            Some(ProjectedTransformer::load_gguf(
                gguf,
                &format!("{prefix}.encoder_transformer"),
                cfg.dimension,
                &output_dimensions,
                cfg.transformer_d_model,
                cfg.transformer_num_heads,
                cfg.transformer_num_layers,
                Some(cfg.transformer_layer_scale),
                cfg.transformer_context,
                cfg.transformer_max_period,
                cfg.transformer_dim_feedforward,
            )?)
        } else {
            None
        };

        let has_dec_transformer = gguf.contains(
            &format!("{prefix}.decoder_transformer.transformer.layers.0.self_attn.in_proj.weight"),
        );
        let decoder_transformer = if has_dec_transformer {
            Some(ProjectedTransformer::load_gguf(
                gguf,
                &format!("{prefix}.decoder_transformer"),
                cfg.dimension,
                &output_dimensions,
                cfg.transformer_d_model,
                cfg.transformer_num_heads,
                cfg.transformer_num_layers,
                Some(cfg.transformer_layer_scale),
                cfg.transformer_context,
                cfg.transformer_max_period,
                cfg.transformer_dim_feedforward,
            )?)
        } else {
            None
        };

        let quantizer = if cfg.num_codebooks > 0 {
            candle_core::bail!("SplitRVQ GGUF loading not implemented; use VarBuilder-based load() for full codec")
        } else {
            let dummy = DummyQuantizer::load_gguf(
                gguf,
                &format!("{prefix}.quantizer"),
                cfg.quantizer_dimension,
                cfg.quantizer_output_dimension,
            )?;
            QuantizerKind::Dummy(dummy)
        };

        let hop_length: usize = cfg.ratios.iter().product();
        let encoder_frame_rate = cfg.sample_rate as f64 / hop_length as f64;

        let (downsample, upsample) =
            if (encoder_frame_rate - cfg.frame_rate as f64).abs() > 0.01 {
                let downsample_stride = (encoder_frame_rate / cfg.frame_rate as f64) as usize;
                let ds_name = format!("{prefix}.downsample.conv.conv.weight");
                let us_name = format!("{prefix}.upsample.convtr.convtr.weight");
                let downsample = if gguf.contains(&ds_name) {
                    Some(ConvDownsample1d::load_gguf(
                        gguf,
                        &format!("{prefix}.downsample"),
                        downsample_stride,
                        cfg.dimension,
                    )?)
                } else {
                    None
                };
                let upsample = if gguf.contains(&us_name) {
                    Some(ConvTrUpsample1d::load_gguf(
                        gguf,
                        &format!("{prefix}.upsample"),
                        downsample_stride,
                        cfg.dimension,
                    )?)
                } else {
                    None
                };
                (downsample, upsample)
            } else {
                (None, None)
            };

        Ok(Self {
            encoder,
            decoder,
            encoder_transformer,
            decoder_transformer,
            quantizer,
            downsample,
            upsample,
            frame_rate: cfg.frame_rate,
            _encoder_frame_rate: encoder_frame_rate,
            sample_rate: cfg.sample_rate,
            _dimension: cfg.dimension,
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    pub fn frame_size(&self) -> usize {
        self.sample_rate / self.frame_rate
    }

    pub fn quantize_encoder_transformer(&mut self, dtype: GgmlDType) -> Result<()> {
        match &mut self.encoder_transformer {
            Some(et) => et.quantize_weights(dtype),
            None => Ok(()),
        }
    }

    pub fn quantize_decoder_transformer(&mut self, dtype: GgmlDType) -> Result<()> {
        match &mut self.decoder_transformer {
            Some(dt) => dt.quantize_weights(dtype),
            None => Ok(()),
        }
    }

    /// Apply the quantizer output projection (DummyQuantizer only).
    /// Input: [B, quantizer_dim, T] -> [B, output_dim, T].
    pub fn quantizer_forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.quantizer {
            QuantizerKind::Dummy(q) => q.forward(x),
            QuantizerKind::SplitRvq(_) => {
                candle_core::bail!("quantizer_forward not supported for SplitRVQ; use quantize_to_codes instead")
            }
        }
    }

    /// Encode latent to token IDs (SplitRVQ only).
    /// Input: [B, dim, T'] latent → Output: [B, n_q, T'] u32 token indices.
    pub fn quantize_to_codes(&self, latent: &Tensor) -> Result<Tensor> {
        match &self.quantizer {
            QuantizerKind::SplitRvq(q) => q.encode(latent),
            QuantizerKind::Dummy(_) => {
                candle_core::bail!("quantize_to_codes requires SplitRVQ quantizer (num_codebooks > 0)")
            }
        }
    }

    pub fn init_state(&self, batch_size: usize, device: &Device) -> Result<MimiState> {
        let decoder = self.decoder.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("init_state requires decoder (use load(), not load_encoder_only())".into()))?;
        let decoder_transformer = self.decoder_transformer.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("init_state requires decoder_transformer".into()))?;
        let upsample_state = match &self.upsample {
            Some(us) => Some(us.init_state(batch_size, device)?),
            None => None,
        };
        let _downsample_state = match &self.downsample {
            Some(ds) => Some(ds.init_state(batch_size, device)?),
            None => None,
        };
        let _encoder_state = match &self.encoder {
            Some(enc) => Some(enc.init_state(batch_size, device)?),
            None => None,
        };
        let _encoder_transformer_state = self.encoder_transformer.as_ref().map(|t| t.init_state());
        let s = MimiState {
            _encoder_state,
            decoder_state: decoder.init_state(batch_size, device)?,
            _encoder_transformer_state,
            decoder_transformer_state: decoder_transformer.init_state(),
            _downsample_state,
            upsample_state,
        };
        Ok(s)
    }

    /// Initialize encoder-only streaming state.
    pub fn init_encoder_state(
        &self,
        batch_size: usize,
        device: &Device,
    ) -> Result<MimiEncoderState> {
        let encoder = self.encoder.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("init_encoder_state requires encoder".into()))?;
        let encoder_transformer = self.encoder_transformer.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("init_encoder_state requires encoder_transformer".into()))?;
        let downsample_state = match &self.downsample {
            Some(ds) => Some(ds.init_state(batch_size, device)?),
            None => None,
        };
        Ok(MimiEncoderState {
            encoder_state: encoder.init_state(batch_size, device)?,
            encoder_transformer_state: encoder_transformer.init_state(),
            downsample_state,
        })
    }

    /// Encode audio to latent (non-streaming). Returns [B, C, T'].
    pub fn encode_to_latent(&self, x: &Tensor) -> Result<Tensor> {
        let encoder = self.encoder.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("encode_to_latent requires encoder".into()))?;
        let encoder_transformer = self.encoder_transformer.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("encode_to_latent requires encoder_transformer".into()))?;

        let device = x.device().clone();
        let frame_size = self.frame_size();
        let x = pad_for_conv1d(x, frame_size, frame_size)?;

        let batch = x.dim(0)?;
        let mut enc_state = encoder.init_state(batch, &device)?;
        let emb = encoder.forward(&x, &mut enc_state)?;

        let mut et_state = encoder_transformer.init_state();
        let outs = encoder_transformer.forward(&emb, &mut et_state)?;
        let emb = &outs[0];

        // Downsample to frame rate
        match &self.downsample {
            Some(ds) => ds.forward_no_state(emb),
            None => Ok(emb.clone()),
        }
    }

    /// Streaming encode: [B, 1, T] audio → [B, C, T'] latent.
    ///
    /// Runs SEANetEncoder → encoder transformer → downsample, all streaming.
    /// May return a tensor with T'=0 if not enough audio has accumulated.
    pub fn encode_streaming(
        &self,
        audio: &Tensor,
        state: &mut MimiEncoderState,
    ) -> Result<Tensor> {
        let encoder = self.encoder.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("encode_streaming requires encoder".into()))?;
        let encoder_transformer = self.encoder_transformer.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("encode_streaming requires encoder_transformer".into()))?;

        // SEANet encoder (streaming via conv buffers)
        let emb = encoder.forward(audio, &mut state.encoder_state)?;

        // Check if encoder produced any output frames
        let t = emb.dim(2)?;
        if t == 0 {
            return Ok(emb);
        }

        // Encoder transformer (streaming via KV cache)
        let outs = encoder_transformer
            .forward(&emb, &mut state.encoder_transformer_state)?;
        let emb = &outs[0];

        // Downsample to target frame rate (streaming via conv buffer)
        match (&self.downsample, &mut state.downsample_state) {
            (Some(ds), Some(ds_state)) => ds.forward(emb, ds_state),
            _ => Ok(emb.clone()),
        }
    }

    /// Decode from latent to audio (streaming). Input: [B, C, T'].
    ///
    /// Requires a full model (loaded via `load()`, not `load_encoder_only()`).
    pub fn decode_from_latent(
        &self,
        latent: &Tensor,
        state: &mut MimiState,
    ) -> Result<Tensor> {
        let decoder = self.decoder.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("decode requires decoder (use load(), not load_encoder_only())".into()))?;
        let decoder_transformer = self.decoder_transformer.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("decode requires decoder_transformer".into()))?;
        // Upsample to encoder frame rate
        let emb = match (&self.upsample, &mut state.upsample_state) {
            (Some(us), Some(us_state)) => {
                us.forward(latent, us_state)?
            }
            _ => latent.clone(),
        };
        let outs = decoder_transformer.forward(&emb, &mut state.decoder_transformer_state)?;
        let audio = decoder.forward(&outs[0], &mut state.decoder_state)?;
        Ok(audio)
    }
}

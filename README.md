# mimi-rs

Rust implementation of the [Mimi](https://huggingface.co/kyutai/mimi) audio codec by Kyutai Labs, built on [candle](https://github.com/huggingface/candle).

Shared library used by [tts-web](https://github.com/idle-intelligence/tts-web) and stt-web for streaming audio encoding/decoding in the browser via WASM.

## Components

- **SEANet** encoder/decoder for audio compression
- **Streaming transformer** with KV cache (Mimi-style context window) and growing KV (FlowLM-style)
- **QLinear** for post-load F32 → Q8_0 weight quantization via candle's `QMatMul`
- **Rotary embeddings**, layer scaling, residual convolutions, resampling

## Usage

```toml
[dependencies]
mimi-rs = { git = "https://github.com/idle-intelligence/mimi-rs.git" }
```

## License

Model weights: [Kyutai license](https://huggingface.co/kyutai/mimi). Code: MIT.

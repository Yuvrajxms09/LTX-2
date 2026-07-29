# Chunked Avatar Prototype

The avatar runner is an isolated experimental pipeline for evaluating whether
LTX-2.3 can produce low-latency, continuous talking-head chunks from a reference
image and external TTS audio. Existing LTX pipelines and CLIs are unchanged.

It uses a distilled one-stage denoising pass at the requested output resolution:

1. Encode the prompt once for the complete session.
2. Encode the relevant external-audio window for each chunk and keep its latent frozen.
3. Generate the first chunk from the reference image.
4. Generate following chunks from the previous decoded tail.
5. Remove the repeated overlap from both video and audio before writing each playable MP4.

This is chunked inference, not causal token streaming. A chunk becomes playable
only after that chunk's denoising, VAE decode, and encoding complete.
Mono TTS input is duplicated to stereo at the audio-VAE boundary, matching the
two-channel checkpoint contract without changing the waveform content.

## Configuration

Copy [`avatar.example.toml`](avatar.example.toml) and update its paths. The
settings intended for avatar experiments are exposed directly:

- `generation_frames`: frames computed by every model invocation; must be `8*k+1`.
- `overlap_frames`: prior tail frames conditioned into the next invocation.
- `reference_strength` and `overlap_strength`: image-conditioning strengths.
- `frame_rate`, resolution, seed behavior, and the complete distilled sigma schedule.
- optional, loader-compatible LoRA paths and strengths. No LoRA is required for
  the external driving-audio baseline.
- CPU/disk model offload, FP8 quantization, `torch.compile`, and warm
  transformer reuse.
- MP4 quality and diagnostic detail.

For a 49-frame generation with a 9-frame overlap at 25 FPS, the first chunk
emits 1.96 seconds and each following full chunk adds 1.60 seconds.

Validate paths and inspect the exact chunk/audio plan without loading models:

```bash
python -m ltx_pipelines.avatar.runner \
    --config packages/ltx-pipelines/docs/avatar.example.toml \
    --dry-run
```

Run inference:

```bash
python -m ltx_pipelines.avatar.runner \
    --config avatar.toml
```

Override common experimental values without editing the file:

```bash
python -m ltx_pipelines.avatar.runner \
    --config avatar.toml \
    --set generation.generation_frames=33 \
    --set generation.overlap_frames=8 \
    --set generation.max_chunks=2
```

Values containing spaces should use TOML quoting:

```bash
--set 'input.prompt="A front-facing avatar speaking naturally"'
```

## Outputs and diagnostics

The output directory contains:

- `chunk_0000.mp4`, `chunk_0001.mp4`, ...: independently playable emitted chunks.
- `conditioning/`: exact decoded tail frames used to condition later chunks.
- `manifest.json`: chunk plan, progress, output paths, wall throughput, and real-time factor.
- `metrics.jsonl`: machine-readable phase, denoising-step, memory, and run events.
- `avatar.log`: human-readable execution log.

Phase metrics include model construction, prompt encoding, audio decode, audio
VAE encoding, image conditioning, denoising, VAE decode, video encoding, and
audio muxing. CUDA synchronization is enabled by default so timings describe
completed GPU work rather than asynchronous dispatch. Disable it only for
minimum-overhead runs.

The complete runner executes under `torch.inference_mode()`, matching Scope and
the official LTX inference CLIs. `run_started` records whether gradients and
inference mode are active. Before the first denoising step of every chunk,
`denoising_preflight` records shape, dtype, device, gradient, and inference
metadata for the video latent, frozen audio latent, and sigma schedule. Failed
runs persist the complete Python traceback in both `manifest.json` and
`metrics.jsonl`.

The manifest snapshots the effective configuration and runtime metadata,
including Python, PyTorch, CUDA, cuDNN, GPU model, compute capability, and total
VRAM. Initialization, model-loading, and prompt failures are persisted with
their exception type and message rather than leaving a stale running manifest.

Per-chunk metrics distinguish:

- `generated_fps`: all frames computed divided by complete chunk wall time.
- `emitted_fps`: non-overlap frames delivered divided by complete chunk wall time.
- `denoising_fps`: all generated frames divided by transformer denoising time only.
- `real_time_factor`: complete chunk wall time divided by emitted media duration.
- `deadline_margin_seconds`: playback time gained or lost while computing the chunk.
- `meets_realtime_deadline`: direct steady-state scheduling pass/fail.
- `phase_seconds`: immediately usable per-chunk timing breakdown.

Real-time scheduling requires `real_time_factor <= 1` for following chunks.
`time_to_first_chunk_seconds` in the manifest measures startup through the first
playable MP4 and therefore includes model and prompt preparation.

Set `diagnostics.tensor_statistics = true` only while investigating numerical
issues. It calculates reductions over denoised video tensors at every step and
therefore changes measured latency.

## Interpretation

Start with two or three chunks and inspect:

- identity drift across `chunk_0000.mp4` and `chunk_0001.mp4`;
- visible pose or lighting jumps at chunk boundaries;
- lip timing around the overlap boundary;
- whether following-chunk `real_time_factor` stays at or below 1;
- first-run compilation versus steady-state denoising-step durations;
- peak allocated and reserved VRAM.

The manifest also records overlap MAE/RMSE/PSNR by comparing each regenerated
overlap against the exact prior tail, plus boundary MAE/RMSE between the prior
last frame and the first newly emitted frame. These are diagnostic signals for
ranking configurations; they do not replace watching the transition.

The runner refuses to write into a non-empty output directory by default, which
prevents metrics and chunks from different experiments being mixed. Set
`output.allow_existing = true` only when intentionally replacing colliding run
artifacts; unrelated files are never deleted.

Use the official monolithic `ltx-2.3-22b-distilled-1.1.safetensors`
checkpoint. When BF16 does not fit in VRAM, set `model.offload = "cpu"` to
stream layers from system RAM without quantizing weights. Use `fp8-cast` only
when reducing model memory is more important than preserving the BF16 baseline.
The separated transformer-only FP8 checkpoint used by Scope does not contain
the text projection and audio/video VAE weights this pipeline loads from the
checkpoint. Daydream's `talkvid-3k` ID-LoRA is for reference-speaker audio
conditioning and is not required for external TTS driving mode.

If overlap conditioning is visually stable but slower than real time, reduce
`generation_frames`, benchmark compilation and FP8 independently, and separate
model time from encoding time using phase metrics. If transitions remain
unstable across reasonable overlap strengths and lengths, use the repository's
video-extension LoRA training mode rather than hiding the discontinuity in the
delivery layer.

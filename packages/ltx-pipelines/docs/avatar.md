# Chunked Avatar Prototype

The avatar runner is an isolated experimental pipeline for evaluating whether
LTX-2.3 can produce low-latency, continuous talking-head chunks from a reference
image and external TTS audio. Existing LTX pipelines and CLIs are unchanged.

It uses a distilled one-stage denoising pass at the requested output resolution:

1. Encode the prompt once for the complete session.
2. Encode the relevant external-audio window for each chunk and keep its latent frozen.
3. Generate the first chunk from the reference image.
4. Generate following chunks from either the previous decoded tail or an exact clean latent prefix.
5. In latent-prefix mode, discard the causally reinterpreted first extension latent and
   fuse the remaining latent overlap into one persistent timeline.
6. Decode that timeline once into the authoritative combined output. Independently
   decoded chunk MP4s remain available for delivery and diagnostics.

This is chunked inference, not causal token streaming. A chunk becomes playable
only after that chunk's denoising, VAE decode, and encoding complete.
Mono TTS input is duplicated to stereo at the audio-VAE boundary, matching the
two-channel checkpoint contract without changing the waveform content.

## Configuration

Copy [`avatar.example.toml`](avatar.example.toml) and update its paths. The
settings intended for avatar experiments are exposed directly:

- `generation_frames`: frames computed by every model invocation; must be `8*k+1`.
- `overlap_frames`: prior tail frames conditioned into the next invocation.
- `continuation_mode`:
  - `image-keyframes` preserves the original decoded-tail experiment.
  - `latent-prefix` carries the previous denoised latent tail directly and bypasses
    PNG serialization and video-VAE re-encoding between chunks.
- `reference_strength` and `overlap_strength`: first-frame and temporal-prefix strengths.
- `identity_anchor_strength`: when greater than zero in `latent-prefix` mode,
  encodes the original portrait once and appends it to every continuation chunk
  as a model-native keyframe at temporal index `-1`. The cached negative-time
  reference is outside the generated timeline and is intended to reduce identity
  drift. Start with `0.25` or `0.5`; `0` disables it.
- `frame_rate`, resolution, seed behavior, and the complete distilled sigma schedule.
- optional, loader-compatible LoRA paths and strengths. No LoRA is required for
  the external driving-audio baseline.
- CPU/disk model offload, FP8 quantization, `torch.compile`, and warm
  transformer reuse.
- MP4 quality and diagnostic detail.

Latent-prefix overlap must be `8*k+1` because of the causal video-VAE layout.
Use 17 overlap frames for three temporal latents or 25 overlap frames for four.
Use `overlap_strength = 1.0` for an exact clean prefix. The official LTX
extension sampler uses `0.5` as its default soft overlap strength; test it only
after the exact baseline if motion through the overlap looks frozen or ghosted.

For a 121-frame generation with a 17-frame latent prefix at 25 FPS, the first
chunk emits 4.84 seconds and each following full chunk adds 4.16 seconds.
The final duration is rounded to the nearest `8*k+1` frame count representable
by the causal video VAE. For the 14.4457-second test audio, that is 361 frames
or 14.44 seconds.

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
    --set 'generation.continuation_mode="latent-prefix"' \
    --set generation.generation_frames=121 \
    --set generation.overlap_frames=17 \
    --set generation.max_chunks=2
```

Values containing spaces should use TOML quoting:

```bash
--set 'input.prompt="A front-facing avatar speaking naturally"'
```

## Outputs and diagnostics

The output directory contains:

- `chunk_0000.mp4`, `chunk_0001.mp4`, ...: independently playable emitted chunks.
- `combined.mp4`: in `latent-prefix` mode, one decode of the fused canonical
  latent timeline. Use this file for visual quality and seam assessment.
- `conditioning/`: decoded tail frames used by `image-keyframes` mode.
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

In `latent-prefix` mode, `latent_prefix_mae`, `latent_prefix_rmse`, and
`latent_prefix_max_abs` verify that strength-1 prefix latents survived denoising
unchanged. Pixel overlap and boundary metrics remain necessary because exact
latent preservation does not by itself guarantee a seamless causal-VAE decode.
Each chunk also records `latent_fusion`: the discarded causal-boundary latent,
the number of blended overlap latents, and the accumulated timeline length.
`identity_anchor_enabled` records whether the cached negative-index portrait
was supplied. The `identity_anchor_ready` event records its fixed index,
strength, and cache status.

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

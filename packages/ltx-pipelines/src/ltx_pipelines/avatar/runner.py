from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import tempfile
import time
import traceback
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import Any

import av
import torch

from ltx_core.conditioning import ConditioningItem
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.transformer import X0Model
from ltx_core.types import Audio
from ltx_pipelines.avatar.config import AvatarConfig, load_avatar_config
from ltx_pipelines.avatar.media import (
    FrameWindow,
    continuity_metrics,
    fuse_causal_latent_extension,
    latent_prefix_metrics,
    save_tail_frames,
    stereo_audio_window,
)
from ltx_pipelines.avatar.metrics import MetricsRecorder, execution_snapshot
from ltx_pipelines.avatar.pipeline import AvatarA2VidPipeline, AvatarChunkResult, AvatarPromptContext
from ltx_pipelines.avatar.planning import AvatarChunk, latent_frames_for_pixel_prefix, plan_avatar_chunks
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import OffloadMode

logger = logging.getLogger(__name__)


def _require_inference_runtime() -> None:
    state = execution_snapshot()
    if state["grad_enabled"] or not state["inference_mode_enabled"]:
        raise RuntimeError(
            "Avatar inference must run inside torch.inference_mode(); "
            f"grad_enabled={state['grad_enabled']}, "
            f"inference_mode_enabled={state['inference_mode_enabled']}"
        )


def probe_audio_duration(path: str) -> float:
    container = av.open(path)
    try:
        stream = next((candidate for candidate in container.streams if candidate.type == "audio"), None)
        if stream is None:
            raise ValueError(f"No audio stream found in {path!r}")
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        duration = 0.0
        for frame in container.decode(stream):
            if frame.pts is not None:
                duration = max(duration, float(frame.pts * stream.time_base) + frame.samples / frame.sample_rate)
        if duration <= 0:
            raise ValueError(f"Could not determine audio duration for {path!r}")
        return duration
    finally:
        container.close()


def _conditioning_inputs(
    config: AvatarConfig,
    chunk: AvatarChunk,
    previous_tail: list[str],
) -> list[ImageConditioningInput]:
    if chunk.index == 0 or config.generation.continuation_mode == "reference-reset":
        return [
            ImageConditioningInput(
                path=config.input.image_path,
                frame_idx=0,
                strength=config.generation.reference_strength,
            )
        ]
    if config.generation.continuation_mode == "latent-prefix":
        return []
    if len(previous_tail) < chunk.overlap_frames:
        raise RuntimeError(
            f"Chunk {chunk.index} requires {chunk.overlap_frames} overlap frames, "
            f"but only {len(previous_tail)} are available"
        )
    selected = previous_tail[-chunk.overlap_frames :] if chunk.overlap_frames else []
    return [
        ImageConditioningInput(
            path=path,
            frame_idx=frame_index,
            strength=config.generation.overlap_strength,
        )
        for frame_index, path in enumerate(selected)
    ]


def _chunk_prompts(config: AvatarConfig, chunks: list[AvatarChunk]) -> tuple[str, ...]:
    configured = config.input.chunk_prompts
    if not configured:
        return (config.input.prompt,) * len(chunks)
    if len(configured) != len(chunks):
        raise ValueError(f"input.chunk_prompts has {len(configured)} entries, but the run plans {len(chunks)} chunks")
    return configured


def _encode_prompt_contexts(
    pipeline: AvatarA2VidPipeline,
    config: AvatarConfig,
    prompts: tuple[str, ...],
    recorder: MetricsRecorder,
) -> tuple[AvatarPromptContext, ...]:
    unique_prompts = tuple(dict.fromkeys(prompts))
    if config.input.enhance_prompt and len(unique_prompts) > 1:
        raise ValueError("input.enhance_prompt cannot be combined with multiple distinct chunk prompts")
    with recorder.phase("prompt_encoding", prompt_count=len(unique_prompts)):
        encoded = pipeline.encode_prompts(
            prompts=unique_prompts,
            enhance_prompt=config.input.enhance_prompt,
            image_path=config.input.image_path,
            seed=config.generation.seed,
        )
    contexts_by_prompt = dict(zip(unique_prompts, encoded, strict=True))
    return tuple(contexts_by_prompt[prompt] for prompt in prompts)


def _latent_prefix_for_chunk(
    config: AvatarConfig,
    chunk: AvatarChunk,
    previous_latent_tail: torch.Tensor | None,
) -> torch.Tensor | None:
    if chunk.index == 0 or config.generation.continuation_mode != "latent-prefix":
        return None
    if previous_latent_tail is None:
        raise RuntimeError(f"Chunk {chunk.index} requires a latent prefix, but no previous latent tail is available")
    required_frames = latent_frames_for_pixel_prefix(chunk.overlap_frames)
    if previous_latent_tail.shape[2] != required_frames:
        raise RuntimeError(
            f"Chunk {chunk.index} requires {required_frames} prefix latents, "
            f"but received {previous_latent_tail.shape[2]}"
        )
    return previous_latent_tail


def _continuation_tail(config: AvatarConfig, latent: torch.Tensor) -> torch.Tensor | None:
    if config.generation.continuation_mode != "latent-prefix":
        return None
    latent_frames = latent_frames_for_pixel_prefix(config.generation.overlap_frames)
    if latent.shape[2] < latent_frames:
        raise RuntimeError(f"Generated latent has {latent.shape[2]} frames; cannot retain {latent_frames}")
    return latent[:, :, -latent_frames:].detach().clone()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def _configure_logging(output_dir: Path, level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level.upper())
    if not root.handlers:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)
    file_handler = logging.FileHandler(output_dir / "avatar.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


class _TransformerContext(AbstractContextManager[X0Model]):
    def __init__(
        self,
        pipeline: AvatarA2VidPipeline,
        recorder: MetricsRecorder,
        enabled: bool,
    ) -> None:
        self._pipeline = pipeline
        self._recorder = recorder
        self._enabled = enabled
        self._context: AbstractContextManager[X0Model] | None = None
        self._transformer: X0Model | None = None

    def __enter__(self) -> X0Model | None:
        if not self._enabled:
            return None
        with self._recorder.phase("transformer_build", warm=True):
            self._context = self._pipeline.stage.model_context()
            self._transformer = self._context.__enter__()
        return self._transformer

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._context is None:
            return None
        with self._recorder.phase("transformer_release", warm=True):
            return self._context.__exit__(exc_type, exc_value, traceback)


def _build_pipeline(config: AvatarConfig, recorder: MetricsRecorder) -> AvatarA2VidPipeline:
    quantization = (
        QuantizationKind(config.model.quantization).to_policy(config.model.checkpoint_path)
        if config.model.quantization is not None
        else None
    )
    loras = tuple(
        LoraPathStrengthAndSDOps(lora.path, lora.strength, LTXV_LORA_COMFY_RENAMING_MAP) for lora in config.model.loras
    )
    with recorder.phase("pipeline_construction"):
        return AvatarA2VidPipeline(
            checkpoint_path=config.model.checkpoint_path,
            gemma_root=config.model.gemma_root,
            loras=loras,
            quantization=quantization,
            compilation_config=config.model.compile,
            offload_mode=OffloadMode(config.model.offload),
        )


def _encode_identity_anchor(
    pipeline: AvatarA2VidPipeline,
    config: AvatarConfig,
    recorder: MetricsRecorder,
) -> list[ConditioningItem]:
    strength = config.generation.identity_anchor_strength
    if strength == 0:
        return []
    image = ImageConditioningInput(
        path=config.input.image_path,
        frame_idx=-1,
        strength=strength,
    )
    with recorder.phase("identity_anchor_encoding", frame_idx=-1, strength=strength):
        conditionings = pipeline.encode_image_conditionings(
            images=[image],
            height=config.generation.height,
            width=config.generation.width,
        )
    if len(conditionings) != 1:
        raise RuntimeError(f"Expected one identity-anchor conditioning, got {len(conditionings)}")
    recorder.emit(
        "identity_anchor_ready",
        image_path=config.input.image_path,
        frame_idx=-1,
        strength=strength,
        cached=True,
    )
    return conditionings


def _generate_with_transformer(
    pipeline: AvatarA2VidPipeline,
    transformer: X0Model,
    context: AvatarPromptContext,
    config: AvatarConfig,
    chunk: AvatarChunk,
    images: list[ImageConditioningInput],
    prefix_latent: torch.Tensor | None,
    identity_conditionings: list[ConditioningItem],
    recorder: MetricsRecorder,
) -> AvatarChunkResult:
    result = pipeline.generate_chunk(
        transformer=transformer,
        context=context,
        images=images,
        prefix_latent=prefix_latent,
        prefix_strength=config.generation.overlap_strength,
        identity_conditionings=identity_conditionings,
        audio_path=config.input.audio_path,
        audio_start_time=chunk.audio_start_seconds,
        seed=config.generation.seed + chunk.index * config.generation.seed_stride,
        height=config.generation.height,
        width=config.generation.width,
        num_frames=chunk.generation_frames,
        frame_rate=config.generation.frame_rate,
        sigmas=torch.tensor(config.generation.sigmas),
        recorder=recorder,
        chunk_index=chunk.index,
        log_denoising_steps=config.diagnostics.log_denoising_steps,
        tensor_statistics=config.diagnostics.tensor_statistics,
    )
    return result


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _runtime_metadata(device: torch.device) -> dict[str, object]:
    metadata: dict[str, object] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
                "cudnn_version": torch.backends.cudnn.version(),
                "total_vram_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
        )
    return metadata


def _generate_chunk(
    pipeline: AvatarA2VidPipeline,
    warm_transformer: X0Model | None,
    context: AvatarPromptContext,
    config: AvatarConfig,
    chunk: AvatarChunk,
    images: list[ImageConditioningInput],
    prefix_latent: torch.Tensor | None,
    identity_conditionings: list[ConditioningItem],
    recorder: MetricsRecorder,
) -> AvatarChunkResult:
    if warm_transformer is not None:
        return _generate_with_transformer(
            pipeline,
            warm_transformer,
            context,
            config,
            chunk,
            images,
            prefix_latent,
            identity_conditionings,
            recorder,
        )
    with (
        recorder.phase("transformer_build_and_release", chunk_index=chunk.index),
        pipeline.stage.model_context() as transformer,
    ):
        return _generate_with_transformer(
            pipeline,
            transformer,
            context,
            config,
            chunk,
            images,
            prefix_latent,
            identity_conditionings,
            recorder,
        )


def _merge_latent_chunk(
    timeline: torch.Tensor | None,
    generated_latent: torch.Tensor,
    prefix_latent_frames: int,
    chunk_index: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    history_frames = 0 if timeline is None else timeline.shape[2]
    merged = (
        generated_latent.detach().clone()
        if timeline is None
        else fuse_causal_latent_extension(
            history=timeline,
            extension=generated_latent,
            prefix_latent_frames=prefix_latent_frames,
        )
    )
    return merged, {
        "history_latent_frames_before": history_frames,
        "generated_latent_frames": generated_latent.shape[2],
        "discarded_causal_boundary_latents": 0 if chunk_index == 0 else 1,
        "fused_overlap_latents": 0 if chunk_index == 0 else prefix_latent_frames - 1,
        "timeline_latent_frames_after": merged.shape[2],
    }


def _encode_combined_latent_output(
    pipeline: AvatarA2VidPipeline,
    config: AvatarConfig,
    chunks: list[AvatarChunk],
    latent_timeline: torch.Tensor,
    emitted_audio: list[torch.Tensor],
    audio_sampling_rate: int,
    output_dir: Path,
    recorder: MetricsRecorder,
) -> dict[str, Any]:
    emitted_frames = sum(chunk.emitted_frames for chunk in chunks)
    target_latent_frames = latent_frames_for_pixel_prefix(emitted_frames)
    if latent_timeline.shape[2] < target_latent_frames:
        raise RuntimeError(
            f"Accumulated timeline has {latent_timeline.shape[2]} latents; expected at least {target_latent_frames}"
        )
    latent_timeline = latent_timeline[:, :, :target_latent_frames]
    required_audio_samples = round(emitted_frames / config.generation.frame_rate * audio_sampling_rate)
    waveform = torch.cat(emitted_audio, dim=-1)
    if waveform.shape[-1] < required_audio_samples:
        waveform = torch.nn.functional.pad(waveform, (0, required_audio_samples - waveform.shape[-1]))
    else:
        waveform = waveform[..., :required_audio_samples]

    combined_path = output_dir / "combined.mp4"
    generator = torch.Generator(device=pipeline.device).manual_seed(config.generation.seed)
    combined_frames = FrameWindow(
        pipeline.video_decoder(latent_timeline, generator=generator),
        drop_frames=0,
        emit_frames=emitted_frames,
        tail_frames=0,
    )
    with recorder.phase("combined_latent_decode_encode_mux"):
        encode_video(
            video=combined_frames,
            fps=int(config.generation.frame_rate),
            audio=Audio(waveform=waveform, sampling_rate=audio_sampling_rate),
            output_path=str(combined_path),
            video_chunks_number=1,
            crf=config.output.crf,
            preset=config.output.preset,
        )
    return {
        "path": str(combined_path),
        "frames": emitted_frames,
        "latent_frames": target_latent_frames,
        "duration_seconds": emitted_frames / config.generation.frame_rate,
        "assembly": "causal-latent-fusion",
        "decoded_once": True,
    }


def _run_chunks(  # noqa: PLR0913, PLR0915
    pipeline: AvatarA2VidPipeline,
    contexts: tuple[AvatarPromptContext, ...],
    prompts: tuple[str, ...],
    config: AvatarConfig,
    chunks: list[AvatarChunk],
    output_dir: Path,
    conditioning_dir: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    recorder: MetricsRecorder,
    run_started: float,
    identity_conditionings: list[ConditioningItem],
) -> None:
    if len(contexts) != len(chunks) or len(prompts) != len(chunks):
        raise ValueError("Every planned chunk requires exactly one prompt context")
    previous_tail: list[str] = []
    previous_tail_tensors: tuple[torch.Tensor, ...] = ()
    previous_latent_tail: torch.Tensor | None = None
    latent_timeline: torch.Tensor | None = None
    emitted_audio: list[torch.Tensor] = []
    audio_sampling_rate: int | None = None
    prefix_latent_frames = (
        latent_frames_for_pixel_prefix(config.generation.overlap_frames)
        if config.generation.continuation_mode == "latent-prefix"
        else 0
    )
    with _TransformerContext(pipeline, recorder, config.model.warm_transformer) as warm_transformer:
        for chunk in chunks:
            recorder.reset_peak_memory()
            chunk_started = time.perf_counter()
            images = _conditioning_inputs(config, chunk, previous_tail)
            prefix_latent = _latent_prefix_for_chunk(config, chunk, previous_latent_tail)
            logger.info(
                "Generating chunk %d: source %.3fs, %d frames, overlap %d, emit %d, continuation %s",
                chunk.index,
                chunk.audio_start_seconds,
                chunk.generation_frames,
                chunk.overlap_frames,
                chunk.emitted_frames,
                config.generation.continuation_mode,
            )
            result = _generate_chunk(
                pipeline,
                warm_transformer,
                contexts[chunk.index],
                config,
                chunk,
                images,
                prefix_latent,
                identity_conditionings,
                recorder,
            )
            frame_window = FrameWindow(
                result.video,
                drop_frames=chunk.overlap_frames,
                emit_frames=chunk.emitted_frames,
                tail_frames=config.generation.overlap_frames,
            )
            chunk_audio = stereo_audio_window(
                result.audio,
                drop_frames=chunk.overlap_frames,
                emit_frames=chunk.emitted_frames,
                frame_rate=chunk.frame_rate,
            )
            if config.generation.continuation_mode == "latent-prefix":
                if audio_sampling_rate is None:
                    audio_sampling_rate = chunk_audio.sampling_rate
                elif audio_sampling_rate != chunk_audio.sampling_rate:
                    raise RuntimeError("Chunk audio sampling rate changed during generation")
                emitted_audio.append(chunk_audio.waveform.detach().to(device="cpu").clone())
            chunk_path = output_dir / f"chunk_{chunk.index:04d}.mp4"
            with recorder.phase("decode_encode_mux", chunk_index=chunk.index):
                encode_video(
                    video=frame_window,
                    fps=int(config.generation.frame_rate),
                    audio=chunk_audio,
                    output_path=str(chunk_path),
                    video_chunks_number=1,
                    crf=config.output.crf,
                    preset=config.output.preset,
                )
            continuity = continuity_metrics(
                previous_tail=previous_tail_tensors,
                reconstructed_overlap=frame_window.dropped,
                first_emitted=frame_window.first_emitted,
            )
            continuity.update(latent_prefix_metrics(prefix_latent, result.latent))
            latent_fusion: dict[str, int] = {}
            if config.generation.continuation_mode == "latent-prefix":
                latent_timeline, latent_fusion = _merge_latent_chunk(
                    timeline=latent_timeline,
                    generated_latent=result.latent,
                    prefix_latent_frames=prefix_latent_frames,
                    chunk_index=chunk.index,
                )
            previous_tail_tensors = frame_window.tail
            previous_latent_tail = _continuation_tail(
                config,
                latent_timeline if latent_timeline is not None else result.latent,
            )
            previous_tail = (
                save_tail_frames(previous_tail_tensors, conditioning_dir, chunk.index)
                if config.generation.continuation_mode == "image-keyframes"
                else []
            )
            elapsed = time.perf_counter() - chunk_started
            phase_seconds = recorder.phase_totals(chunk.index)
            denoising_seconds = phase_seconds.get("denoising")
            deadline_seconds = chunk.emitted_duration_seconds
            chunk_record = {
                **asdict(chunk),
                "path": str(chunk_path),
                "wall_seconds": elapsed,
                "generated_fps": chunk.generation_frames / elapsed,
                "emitted_fps": chunk.emitted_frames / elapsed,
                "denoising_fps": (
                    chunk.generation_frames / denoising_seconds
                    if denoising_seconds is not None and denoising_seconds > 0
                    else None
                ),
                "real_time_factor": elapsed / deadline_seconds,
                "deadline_seconds": deadline_seconds,
                "deadline_margin_seconds": deadline_seconds - elapsed,
                "meets_realtime_deadline": elapsed <= deadline_seconds,
                "phase_seconds": phase_seconds,
                "continuation_mode": config.generation.continuation_mode,
                "prompt": prompts[chunk.index],
                "conditioning_images": [image.path for image in images],
                "identity_anchor_enabled": bool(identity_conditionings),
                "continuity": continuity,
                "latent_fusion": latent_fusion,
            }
            manifest["completed_chunks"].append(chunk_record)
            if len(manifest["completed_chunks"]) == 1:
                manifest["time_to_first_chunk_seconds"] = time.perf_counter() - run_started
            _write_json(manifest_path, manifest)
            recorder.emit("chunk_completed", **chunk_record, **recorder.hardware_snapshot())

    if config.generation.continuation_mode != "latent-prefix":
        return
    if latent_timeline is None or audio_sampling_rate is None or not emitted_audio:
        raise RuntimeError("Latent-prefix generation completed without an accumulated timeline")

    manifest["combined_output"] = _encode_combined_latent_output(
        pipeline=pipeline,
        config=config,
        chunks=chunks,
        latent_timeline=latent_timeline,
        emitted_audio=emitted_audio,
        audio_sampling_rate=audio_sampling_rate,
        output_dir=output_dir,
        recorder=recorder,
    )
    _write_json(manifest_path, manifest)
    recorder.emit("combined_output_completed", **manifest["combined_output"], **recorder.hardware_snapshot())


@torch.inference_mode()
def run_avatar(config: AvatarConfig) -> Path:
    _require_inference_runtime()
    output_dir = Path(config.output.directory)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.output.allow_existing:
        raise FileExistsError(
            f"Output directory {output_dir} is not empty. Choose a new directory or set output.allow_existing=true."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(output_dir, config.diagnostics.log_level)
    metrics_path = output_dir / "metrics.jsonl" if config.diagnostics.jsonl_metrics else None
    device = _device()
    recorder = MetricsRecorder(metrics_path, device=device, synchronize_cuda=config.diagnostics.synchronize_cuda)
    runtime = _runtime_metadata(device)
    recorder.emit(
        "run_started",
        config=asdict(config),
        runtime=runtime,
        execution=execution_snapshot(),
        **recorder.hardware_snapshot(),
    )
    run_started = time.perf_counter()
    manifest: dict[str, Any] = {
        "status": "initializing",
        "config": asdict(config),
        "runtime": runtime,
        "completed_chunks": [],
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    temporary_context: tempfile.TemporaryDirectory[str] | None = None
    try:
        duration = probe_audio_duration(config.input.audio_path)
        chunks = plan_avatar_chunks(
            audio_duration_seconds=duration,
            frame_rate=config.generation.frame_rate,
            generation_frames=config.generation.generation_frames,
            overlap_frames=config.generation.overlap_frames,
            max_chunks=config.generation.max_chunks,
            align_total_frames=config.generation.continuation_mode == "latent-prefix",
        )
        prompts = _chunk_prompts(config, chunks)
        manifest.update(
            {
                "status": "running",
                "audio_duration_seconds": duration,
                "planned_chunks": [asdict(chunk) for chunk in chunks],
                "chunk_prompts": list(prompts),
            }
        )
        _write_json(manifest_path, manifest)
        pipeline = _build_pipeline(config, recorder)
        contexts = _encode_prompt_contexts(pipeline, config, prompts, recorder)
        identity_conditionings = _encode_identity_anchor(pipeline, config, recorder)
        temporary_context = (
            None
            if config.output.save_conditioning_frames
            else tempfile.TemporaryDirectory(prefix="ltx-avatar-conditioning-", dir=output_dir)
        )
        conditioning_dir = output_dir / "conditioning" if temporary_context is None else Path(temporary_context.name)
        _run_chunks(
            pipeline,
            contexts,
            prompts,
            config,
            chunks,
            output_dir,
            conditioning_dir,
            manifest,
            manifest_path,
            recorder,
            run_started,
            identity_conditionings,
        )
        manifest["status"] = "completed"
        manifest["wall_seconds"] = time.perf_counter() - run_started
        manifest.setdefault("time_to_first_chunk_seconds", None)
        _write_json(manifest_path, manifest)
        recorder.emit("run_completed", **manifest, **recorder.hardware_snapshot())
        return manifest_path
    except Exception as error:
        formatted_traceback = traceback.format_exc()
        manifest["status"] = "failed"
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": formatted_traceback,
        }
        manifest["wall_seconds"] = time.perf_counter() - run_started
        _write_json(manifest_path, manifest)
        recorder.emit(
            "run_failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=formatted_traceback,
            execution=execution_snapshot(),
            **recorder.hardware_snapshot(),
        )
        raise
    finally:
        if temporary_context is not None:
            temporary_context.cleanup()


def _dry_run(config: AvatarConfig) -> None:
    duration = probe_audio_duration(config.input.audio_path)
    chunks = plan_avatar_chunks(
        audio_duration_seconds=duration,
        frame_rate=config.generation.frame_rate,
        generation_frames=config.generation.generation_frames,
        overlap_frames=config.generation.overlap_frames,
        max_chunks=config.generation.max_chunks,
        align_total_frames=config.generation.continuation_mode == "latent-prefix",
    )
    prompts = _chunk_prompts(config, chunks)
    sys.stdout.write(
        json.dumps(
            {
                "audio_duration_seconds": duration,
                "config": asdict(config),
                "chunks": [asdict(chunk) for chunk in chunks],
                "chunk_prompts": list(prompts),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configurable chunked LTX-2 avatar inference.")
    parser.add_argument("--config", required=True, help="Path to the avatar TOML configuration.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override one top-level config value. Repeat as needed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the chunk plan without models.",
    )
    args = parser.parse_args()
    config = load_avatar_config(args.config, overrides=tuple(args.overrides))
    if args.dry_run:
        _dry_run(config)
        return
    manifest_path = run_avatar(config)
    logger.info("Avatar run complete: %s", manifest_path)


if __name__ == "__main__":
    main()

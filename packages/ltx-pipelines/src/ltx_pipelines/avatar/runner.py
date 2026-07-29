from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import Any

import av
import torch

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.transformer import X0Model
from ltx_core.types import Audio
from ltx_pipelines.avatar.config import AvatarConfig, load_avatar_config
from ltx_pipelines.avatar.media import FrameWindow, continuity_metrics, save_tail_frames, stereo_audio_window
from ltx_pipelines.avatar.metrics import MetricsRecorder
from ltx_pipelines.avatar.pipeline import AvatarA2VidPipeline, AvatarPromptContext
from ltx_pipelines.avatar.planning import AvatarChunk, plan_avatar_chunks
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.quantization_factory import QuantizationKind

logger = logging.getLogger(__name__)


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
    if chunk.index == 0:
        return [
            ImageConditioningInput(
                path=config.input.image_path,
                frame_idx=0,
                strength=config.generation.reference_strength,
            )
        ]
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
        )


def _generate_with_transformer(
    pipeline: AvatarA2VidPipeline,
    transformer: X0Model,
    context: AvatarPromptContext,
    config: AvatarConfig,
    chunk: AvatarChunk,
    images: list[ImageConditioningInput],
    recorder: MetricsRecorder,
) -> tuple[Iterator[torch.Tensor], Audio]:
    result = pipeline.generate_chunk(
        transformer=transformer,
        context=context,
        images=images,
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
    return result.video, result.audio


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
    recorder: MetricsRecorder,
) -> tuple[Iterator[torch.Tensor], Audio]:
    if warm_transformer is not None:
        return _generate_with_transformer(
            pipeline,
            warm_transformer,
            context,
            config,
            chunk,
            images,
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
            recorder,
        )


def _run_chunks(
    pipeline: AvatarA2VidPipeline,
    context: AvatarPromptContext,
    config: AvatarConfig,
    chunks: list[AvatarChunk],
    output_dir: Path,
    conditioning_dir: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    recorder: MetricsRecorder,
    run_started: float,
) -> None:
    previous_tail: list[str] = []
    previous_tail_tensors: tuple[torch.Tensor, ...] = ()
    with _TransformerContext(pipeline, recorder, config.model.warm_transformer) as warm_transformer:
        for chunk in chunks:
            recorder.reset_peak_memory()
            chunk_started = time.perf_counter()
            images = _conditioning_inputs(config, chunk, previous_tail)
            logger.info(
                "Generating chunk %d: source %.3fs, %d frames, overlap %d, emit %d",
                chunk.index,
                chunk.audio_start_seconds,
                chunk.generation_frames,
                chunk.overlap_frames,
                chunk.emitted_frames,
            )
            video, audio = _generate_chunk(
                pipeline,
                warm_transformer,
                context,
                config,
                chunk,
                images,
                recorder,
            )
            frame_window = FrameWindow(
                video,
                drop_frames=chunk.overlap_frames,
                emit_frames=chunk.emitted_frames,
                tail_frames=config.generation.overlap_frames,
            )
            chunk_audio = stereo_audio_window(
                audio,
                drop_frames=chunk.overlap_frames,
                emit_frames=chunk.emitted_frames,
                frame_rate=chunk.frame_rate,
            )
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
            previous_tail_tensors = frame_window.tail
            previous_tail = save_tail_frames(previous_tail_tensors, conditioning_dir, chunk.index)
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
                "conditioning_images": [image.path for image in images],
                "continuity": continuity,
            }
            manifest["completed_chunks"].append(chunk_record)
            if len(manifest["completed_chunks"]) == 1:
                manifest["time_to_first_chunk_seconds"] = time.perf_counter() - run_started
            _write_json(manifest_path, manifest)
            recorder.emit("chunk_completed", **chunk_record, **recorder.hardware_snapshot())


def run_avatar(config: AvatarConfig) -> Path:
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
    recorder.emit("run_started", config=asdict(config), runtime=runtime, **recorder.hardware_snapshot())
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
        )
        manifest.update(
            {
                "status": "running",
                "audio_duration_seconds": duration,
                "planned_chunks": [asdict(chunk) for chunk in chunks],
            }
        )
        _write_json(manifest_path, manifest)
        pipeline = _build_pipeline(config, recorder)
        with recorder.phase("prompt_encoding"):
            context = pipeline.encode_prompt(
                prompt=config.input.prompt,
                enhance_prompt=config.input.enhance_prompt,
                image_path=config.input.image_path,
                seed=config.generation.seed,
            )
        temporary_context = (
            None
            if config.output.save_conditioning_frames
            else tempfile.TemporaryDirectory(prefix="ltx-avatar-conditioning-", dir=output_dir)
        )
        conditioning_dir = output_dir / "conditioning" if temporary_context is None else Path(temporary_context.name)
        _run_chunks(
            pipeline,
            context,
            config,
            chunks,
            output_dir,
            conditioning_dir,
            manifest,
            manifest_path,
            recorder,
            run_started,
        )
        manifest["status"] = "completed"
        manifest["wall_seconds"] = time.perf_counter() - run_started
        manifest.setdefault("time_to_first_chunk_seconds", None)
        _write_json(manifest_path, manifest)
        recorder.emit("run_completed", **manifest, **recorder.hardware_snapshot())
        return manifest_path
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        manifest["wall_seconds"] = time.perf_counter() - run_started
        _write_json(manifest_path, manifest)
        recorder.emit("run_failed", error_type=type(error).__name__, error=str(error), **recorder.hardware_snapshot())
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
    )
    sys.stdout.write(
        json.dumps(
            {
                "audio_duration_seconds": duration,
                "config": asdict(config),
                "chunks": [asdict(chunk) for chunk in chunks],
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

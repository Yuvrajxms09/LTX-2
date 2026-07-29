from __future__ import annotations

import json
import logging
import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch

from ltx_core.model.transformer import X0Model
from ltx_core.types import LatentState
from ltx_pipelines.utils.types import DenoisedLatentResult, Denoiser

logger = logging.getLogger(__name__)


def _rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class MetricsRecorder:
    def __init__(
        self,
        output_path: Path | None,
        device: torch.device,
        synchronize_cuda: bool,
    ) -> None:
        self._output_path = output_path
        self._device = device
        self._synchronize_cuda = synchronize_cuda
        self._phase_durations: dict[tuple[int | None, str], list[float]] = {}
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding="utf-8")

    @property
    def device(self) -> torch.device:
        return self._device

    def synchronize(self) -> None:
        if self._synchronize_cuda and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def reset_peak_memory(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)

    def hardware_snapshot(self) -> dict[str, int | str]:
        snapshot: dict[str, int | str] = {
            "device": str(self._device),
            "process_peak_rss_bytes": _rss_bytes(),
        }
        if self._device.type == "cuda":
            snapshot.update(
                {
                    "gpu_name": torch.cuda.get_device_name(self._device),
                    "gpu_allocated_bytes": torch.cuda.memory_allocated(self._device),
                    "gpu_reserved_bytes": torch.cuda.memory_reserved(self._device),
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(self._device),
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(self._device),
                }
            )
        return snapshot

    def emit(self, event: str, **values: object) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **values,
        }
        if self._output_path is not None:
            with self._output_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, default=_json_default, sort_keys=True))
                output.write("\n")
        logger.debug("metric %s", json.dumps(record, default=_json_default, sort_keys=True))

    def phase_totals(self, chunk_index: int | None) -> dict[str, float]:
        return {
            phase: sum(durations)
            for (recorded_chunk, phase), durations in self._phase_durations.items()
            if recorded_chunk == chunk_index
        }

    @contextmanager
    def phase(self, name: str, **dimensions: object) -> Iterator[None]:
        self.synchronize()
        started = time.perf_counter()
        try:
            yield
        except Exception as error:
            self.synchronize()
            self.emit(
                "phase",
                phase=name,
                status="failed",
                duration_seconds=time.perf_counter() - started,
                error_type=type(error).__name__,
                error=str(error),
                **dimensions,
                **self.hardware_snapshot(),
            )
            raise
        self.synchronize()
        duration = time.perf_counter() - started
        chunk_index = dimensions.get("chunk_index")
        if chunk_index is not None and not isinstance(chunk_index, int):
            raise TypeError("chunk_index metric dimension must be an integer")
        self._phase_durations.setdefault((chunk_index, name), []).append(duration)
        self.emit(
            "phase",
            phase=name,
            status="completed",
            duration_seconds=duration,
            **dimensions,
            **self.hardware_snapshot(),
        )


def _tensor_statistics(tensor: torch.Tensor) -> dict[str, float | list[int] | str]:
    values = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": values.mean().item(),
        "std": values.std().item(),
        "min": values.min().item(),
        "max": values.max().item(),
        "norm": values.norm().item(),
    }


class TimedDenoiser(Denoiser):
    def __init__(
        self,
        denoiser: Denoiser,
        recorder: MetricsRecorder,
        chunk_index: int,
        enabled: bool,
        tensor_statistics: bool,
    ) -> None:
        self._denoiser = denoiser
        self._recorder = recorder
        self._chunk_index = chunk_index
        self._enabled = enabled
        self._tensor_statistics = tensor_statistics

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
        if not self._enabled:
            return self._denoiser(transformer, video_state, audio_state, sigmas, step_index)

        self._recorder.synchronize()
        started = time.perf_counter()
        video_result, audio_result = self._denoiser(
            transformer,
            video_state,
            audio_state,
            sigmas,
            step_index,
        )
        self._recorder.synchronize()
        values: dict[str, Any] = {
            "chunk_index": self._chunk_index,
            "step_index": step_index,
            "sigma": sigmas[step_index].item(),
            "next_sigma": sigmas[step_index + 1].item(),
            "duration_seconds": time.perf_counter() - started,
            **self._recorder.hardware_snapshot(),
        }
        if self._tensor_statistics and video_result is not None:
            values["video_denoised"] = _tensor_statistics(video_result.denoised)
        self._recorder.emit("denoising_step", **values)
        return video_result, audio_result

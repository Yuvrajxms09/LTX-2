from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import torch
from PIL import Image

from ltx_core.types import Audio


def fuse_causal_latent_extension(
    history: torch.Tensor,
    extension: torch.Tensor,
    prefix_latent_frames: int,
) -> torch.Tensor:
    """Merge an LTX extension into a canonical causal-VAE latent timeline."""
    if history.ndim != 5 or extension.ndim != 5:
        raise ValueError("Expected latent tensors shaped (B,C,F,H,W)")
    if history.shape[:2] != extension.shape[:2] or history.shape[3:] != extension.shape[3:]:
        raise ValueError("History and extension latents have incompatible batch, channel, or spatial shapes")
    if prefix_latent_frames < 2:
        raise ValueError("Causal latent extension requires at least two prefix latent frames")
    if history.shape[2] < prefix_latent_frames:
        raise ValueError("History is shorter than the requested latent prefix")
    if extension.shape[2] <= prefix_latent_frames:
        raise ValueError("Extension must contain new latent frames beyond its prefix")

    # An ordinary eight-frame tail latent is interpreted as the one-frame latent
    # at index zero of a fresh LTX sequence. It cannot be appended to the
    # canonical timeline; official LTX extension code drops it before fusion.
    extension = extension[:, :, 1:]
    overlap = prefix_latent_frames - 1
    alpha = torch.linspace(
        1.0,
        0.0,
        overlap + 2,
        device=history.device,
        dtype=torch.float32,
    )[1:-1].view(1, 1, overlap, 1, 1)
    blended = (alpha * history[:, :, -overlap:].float() + (1.0 - alpha) * extension[:, :, :overlap].float()).to(
        history.dtype
    )
    return torch.cat(
        [
            history[:, :, :-overlap],
            blended,
            extension[:, :, overlap:],
        ],
        dim=2,
    )


def stereo_audio_window(audio: Audio, drop_frames: int, emit_frames: int, frame_rate: float) -> Audio:
    start = round(drop_frames / frame_rate * audio.sampling_rate)
    end = start + round(emit_frames / frame_rate * audio.sampling_rate)
    waveform = audio.waveform[..., start:end]
    if waveform.ndim != 2:
        raise ValueError(f"Expected channel-first audio after decoding, got shape {tuple(waveform.shape)}")
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[1] == 1:
        waveform = waveform.T.repeat(2, 1)
    elif waveform.shape[0] != 2:
        raise ValueError(f"Expected mono or stereo audio, got shape {tuple(waveform.shape)}")
    return Audio(waveform=waveform, sampling_rate=audio.sampling_rate)


class FrameWindow(Iterator[torch.Tensor]):
    """Drop overlap frames, cap emitted frames, and retain a small continuation tail."""

    def __init__(
        self,
        chunks: Iterator[torch.Tensor],
        drop_frames: int,
        emit_frames: int,
        tail_frames: int,
    ) -> None:
        self._chunks = iter(chunks)
        self._drop_remaining = drop_frames
        self._emit_remaining = emit_frames
        self._tail: deque[torch.Tensor] = deque(maxlen=tail_frames)
        self._dropped: list[torch.Tensor] = []
        self._first_emitted: torch.Tensor | None = None

    @property
    def tail(self) -> tuple[torch.Tensor, ...]:
        return tuple(self._tail)

    @property
    def dropped(self) -> tuple[torch.Tensor, ...]:
        return tuple(self._dropped)

    @property
    def first_emitted(self) -> torch.Tensor | None:
        return self._first_emitted

    def __iter__(self) -> FrameWindow:
        return self

    def __next__(self) -> torch.Tensor:
        while self._emit_remaining > 0:
            chunk = next(self._chunks)
            if chunk.ndim != 4:
                raise ValueError(f"Expected decoded frames shaped (F,H,W,C), got {tuple(chunk.shape)}")
            cpu_chunk = chunk.detach().to(device="cpu", dtype=torch.float32)
            self._tail.extend(frame.clone() for frame in cpu_chunk)
            if self._drop_remaining:
                dropped = min(self._drop_remaining, cpu_chunk.shape[0])
                self._dropped.extend(frame.clone() for frame in cpu_chunk[:dropped])
                cpu_chunk = cpu_chunk[dropped:]
                self._drop_remaining -= dropped
            if cpu_chunk.shape[0] == 0:
                continue
            emitted = min(self._emit_remaining, cpu_chunk.shape[0])
            self._emit_remaining -= emitted
            if self._first_emitted is None:
                self._first_emitted = cpu_chunk[0].clone()
            return cpu_chunk[:emitted]
        close = getattr(self._chunks, "close", None)
        if close is not None:
            close()
        raise StopIteration


def save_tail_frames(frames: tuple[torch.Tensor, ...], directory: Path, chunk_index: int) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for frame_index, frame in enumerate(frames):
        pixels = frame.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
        path = directory / f"chunk_{chunk_index:04d}_tail_{frame_index:03d}.png"
        Image.fromarray(pixels).save(path)
        paths.append(str(path))
    return paths


def continuity_metrics(
    previous_tail: tuple[torch.Tensor, ...],
    reconstructed_overlap: tuple[torch.Tensor, ...],
    first_emitted: torch.Tensor | None,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    matched = min(len(previous_tail), len(reconstructed_overlap))
    if matched:
        previous = torch.stack(previous_tail[-matched:]).float()
        reconstructed = torch.stack(reconstructed_overlap[:matched]).float()
        difference = reconstructed - previous
        mse = difference.square().mean().item()
        metrics.update(
            {
                "overlap_compared_frames": matched,
                "overlap_mae": difference.abs().mean().item(),
                "overlap_rmse": math.sqrt(mse),
                "overlap_psnr_db": 120.0 if mse == 0 else -10.0 * math.log10(mse),
            }
        )
    if previous_tail and first_emitted is not None:
        boundary_difference = first_emitted.float() - previous_tail[-1].float()
        metrics.update(
            {
                "boundary_mae": boundary_difference.abs().mean().item(),
                "boundary_rmse": boundary_difference.square().mean().sqrt().item(),
            }
        )
    return metrics


def latent_prefix_metrics(
    expected_prefix: torch.Tensor | None,
    generated_latent: torch.Tensor,
) -> dict[str, float | int]:
    if expected_prefix is None:
        return {}
    if generated_latent.ndim != 5 or expected_prefix.ndim != 5:
        raise ValueError("Expected latent tensors shaped (B,C,F,H,W)")
    prefix_frames = expected_prefix.shape[2]
    batch_channels_match = generated_latent.shape[:2] == expected_prefix.shape[:2]
    spatial_shape_matches = generated_latent.shape[3:] == expected_prefix.shape[3:]
    if not batch_channels_match or not spatial_shape_matches:
        raise ValueError("Generated latent and expected prefix have incompatible batch, channel, or spatial shapes")
    if generated_latent.shape[2] < prefix_frames:
        raise ValueError("Generated latent is shorter than the expected prefix")
    difference = generated_latent[:, :, :prefix_frames].float() - expected_prefix.float()
    return {
        "latent_prefix_frames": prefix_frames,
        "latent_prefix_mae": difference.abs().mean().item(),
        "latent_prefix_rmse": difference.square().mean().sqrt().item(),
        "latent_prefix_max_abs": difference.abs().max().item(),
    }

import pytest
import torch

from ltx_pipelines.avatar.media import FrameWindow, continuity_metrics, latent_prefix_metrics
from ltx_pipelines.avatar.planning import latent_frames_for_pixel_prefix, plan_avatar_chunks


def test_latent_frames_for_pixel_prefix_matches_causal_vae_layout() -> None:
    assert latent_frames_for_pixel_prefix(17) == 3
    assert latent_frames_for_pixel_prefix(25) == 4


def test_latent_frames_for_pixel_prefix_rejects_unaligned_frame_count() -> None:
    with pytest.raises(ValueError, match=r"8\*k \+ 1"):
        latent_frames_for_pixel_prefix(8)


def test_plan_avatar_chunks_rewinds_audio_and_only_emits_unique_frames() -> None:
    chunks = plan_avatar_chunks(
        audio_duration_seconds=14.44,
        frame_rate=25,
        generation_frames=49,
        overlap_frames=9,
    )

    assert len(chunks) == 9
    assert chunks[0].generation_start_frame == 0
    assert chunks[0].emitted_frames == 49
    assert chunks[1].generation_start_frame == 40
    assert chunks[1].emitted_start_frame == 49
    assert chunks[1].overlap_frames == 9
    assert chunks[-1].emitted_frames == 32
    assert sum(chunk.emitted_frames for chunk in chunks) == 361


def test_plan_avatar_chunks_honors_max_chunks() -> None:
    chunks = plan_avatar_chunks(14.44, 25, 49, 9, max_chunks=2)

    assert len(chunks) == 2
    assert sum(chunk.emitted_frames for chunk in chunks) == 89


def test_frame_window_drops_overlap_caps_output_and_keeps_tail() -> None:
    frames = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1)
    window = FrameWindow(iter([frames[:4], frames[4:]]), drop_frames=2, emit_frames=5, tail_frames=3)

    emitted = torch.cat(list(window))

    assert emitted.flatten().tolist() == [2, 3, 4, 5, 6]
    assert [frame.item() for frame in window.dropped] == [0, 1]
    assert window.first_emitted is not None
    assert window.first_emitted.item() == 2
    assert [frame.item() for frame in window.tail] == [7, 8, 9]


def test_continuity_metrics_compare_overlap_and_boundary() -> None:
    previous = tuple(torch.full((1, 1, 1), value) for value in (0.0, 0.5))
    reconstructed = tuple(torch.full((1, 1, 1), value) for value in (0.0, 0.75))

    metrics = continuity_metrics(previous, reconstructed, first_emitted=torch.ones(1, 1, 1))

    assert metrics["overlap_compared_frames"] == 2
    assert metrics["overlap_mae"] == 0.125
    assert metrics["boundary_mae"] == 0.5


def test_latent_prefix_metrics_verify_the_clean_prefix_survived_denoising() -> None:
    prefix = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 1, 2)
    generated = torch.cat([prefix, torch.full((1, 2, 2, 1, 2), 100.0)], dim=2)

    metrics = latent_prefix_metrics(prefix, generated)

    assert metrics == {
        "latent_prefix_frames": 3,
        "latent_prefix_mae": 0.0,
        "latent_prefix_rmse": 0.0,
        "latent_prefix_max_abs": 0.0,
    }

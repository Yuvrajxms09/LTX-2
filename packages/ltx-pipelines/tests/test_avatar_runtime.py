from unittest.mock import MagicMock, Mock

import pytest
import torch

from ltx_pipelines.avatar.config import (
    AvatarConfig,
    GenerationConfig,
    InputConfig,
    ModelConfig,
    OutputConfig,
)
from ltx_pipelines.avatar.metrics import execution_snapshot, tensor_snapshot
from ltx_pipelines.avatar.planning import AvatarChunk
from ltx_pipelines.avatar.runner import (
    _continuation_tail,
    _encode_identity_anchor,
    _latent_prefix_for_chunk,
    _require_inference_runtime,
)


def test_require_inference_runtime_rejects_autograd_execution() -> None:
    with pytest.raises(RuntimeError, match=r"torch\.inference_mode"):
        _require_inference_runtime()


def test_require_inference_runtime_accepts_inference_execution() -> None:
    with torch.inference_mode():
        _require_inference_runtime()


def test_runtime_diagnostics_identify_inference_tensors() -> None:
    with torch.inference_mode():
        tensor = torch.ones(2, 3)
        execution = execution_snapshot()
        metadata = tensor_snapshot(tensor)

    assert execution == {
        "grad_enabled": False,
        "inference_mode_enabled": True,
    }
    assert metadata["shape"] == [2, 3]
    assert metadata["requires_grad"] is False
    assert metadata["is_inference"] is True


def _latent_prefix_config() -> AvatarConfig:
    return AvatarConfig(
        model=ModelConfig(checkpoint_path="model", gemma_root="gemma"),
        input=InputConfig(image_path="image", audio_path="audio", prompt="prompt"),
        generation=GenerationConfig(
            generation_frames=121,
            overlap_frames=17,
            continuation_mode="latent-prefix",
        ),
        output=OutputConfig(directory="output"),
    )


def test_continuation_tail_carries_exact_generated_latents() -> None:
    latent = torch.arange(1 * 2 * 16 * 1 * 1, dtype=torch.float32).reshape(1, 2, 16, 1, 1)

    tail = _continuation_tail(_latent_prefix_config(), latent)

    assert tail is not None
    assert tail.shape == (1, 2, 3, 1, 1)
    torch.testing.assert_close(tail, latent[:, :, -3:])
    assert tail.data_ptr() != latent.data_ptr()


def test_latent_prefix_for_chunk_reuses_exact_tail() -> None:
    config = _latent_prefix_config()
    chunk = AvatarChunk(
        index=1,
        generation_start_frame=104,
        generation_frames=121,
        overlap_frames=17,
        emitted_start_frame=121,
        emitted_frames=104,
        frame_rate=25.0,
    )
    tail = torch.randn(1, 128, 3, 16, 16)

    prefix = _latent_prefix_for_chunk(config, chunk, tail)

    assert prefix is tail


def test_identity_anchor_is_cached_at_negative_temporal_index() -> None:
    config = AvatarConfig(
        model=ModelConfig(checkpoint_path="model", gemma_root="gemma"),
        input=InputConfig(image_path="image", audio_path="audio", prompt="prompt"),
        generation=GenerationConfig(
            generation_frames=121,
            overlap_frames=17,
            continuation_mode="latent-prefix",
            identity_anchor_strength=0.5,
        ),
        output=OutputConfig(directory="output"),
    )
    pipeline = Mock()
    conditioning = object()
    pipeline.encode_image_conditionings.return_value = [conditioning]
    recorder = MagicMock()

    result = _encode_identity_anchor(pipeline, config, recorder)

    assert result == [conditioning]
    image = pipeline.encode_image_conditionings.call_args.kwargs["images"][0]
    assert image.path == "image"
    assert image.frame_idx == -1
    assert image.strength == 0.5
    recorder.emit.assert_called_once_with(
        "identity_anchor_ready",
        image_path="image",
        frame_idx=-1,
        strength=0.5,
        cached=True,
    )

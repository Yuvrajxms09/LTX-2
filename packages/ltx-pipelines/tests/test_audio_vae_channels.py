import pytest
import torch

from ltx_core.model.audio_vae.audio_vae import _match_audio_channels
from ltx_core.types import Audio


def test_match_audio_channels_expands_mono_for_stereo_encoder() -> None:
    waveform = torch.tensor([[[0.0, 0.25, -0.5]]])

    result = _match_audio_channels(Audio(waveform=waveform, sampling_rate=16_000), target_channels=2)

    assert result.waveform.shape == (1, 2, 3)
    torch.testing.assert_close(result.waveform[:, 0], waveform[:, 0])
    torch.testing.assert_close(result.waveform[:, 1], waveform[:, 0])


def test_match_audio_channels_preserves_matching_input() -> None:
    audio = Audio(waveform=torch.zeros(1, 2, 4), sampling_rate=16_000)

    assert _match_audio_channels(audio, target_channels=2) is audio


def test_match_audio_channels_rejects_ambiguous_multichannel_input() -> None:
    audio = Audio(waveform=torch.zeros(1, 6, 4), sampling_rate=48_000)

    with pytest.raises(ValueError, match="only mono input"):
        _match_audio_channels(audio, target_channels=2)

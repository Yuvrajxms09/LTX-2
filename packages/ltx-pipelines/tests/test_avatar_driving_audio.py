import torch

from ltx_core.model.transformer import Modality
from ltx_core.types import LatentState
from ltx_pipelines.avatar.pipeline import DrivingAudioDenoiser


class _CapturingTransformer:
    def __init__(self) -> None:
        self.video: Modality | None = None
        self.audio: Modality | None = None

    def __call__(
        self,
        *,
        video: Modality,
        audio: Modality,
        perturbations: object | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.video = video
        self.audio = audio
        assert perturbations is None
        return video.latent, audio.latent


def _latent_state(denoise_value: float) -> LatentState:
    latent = torch.ones(1, 2, 3)
    return LatentState(
        latent=latent,
        denoise_mask=torch.full_like(latent, denoise_value),
        positions=torch.zeros_like(latent),
        clean_latent=latent.clone(),
    )


def test_driving_audio_is_presented_as_clean_during_video_denoising() -> None:
    transformer = _CapturingTransformer()
    denoiser = DrivingAudioDenoiser(
        video_context=torch.ones(1, 2, 4),
        audio_context=torch.ones(1, 2, 4),
    )

    denoiser(
        transformer,  # type: ignore[arg-type]
        video_state=_latent_state(1.0),
        audio_state=_latent_state(0.0),
        sigmas=torch.tensor([0.75, 0.0]),
        step_index=0,
    )

    assert transformer.video is not None
    assert transformer.audio is not None
    torch.testing.assert_close(transformer.video.sigma, torch.tensor(0.75))
    torch.testing.assert_close(transformer.video.timesteps, torch.full((1, 2, 3), 0.75))
    torch.testing.assert_close(transformer.audio.sigma, torch.tensor(0.0))
    torch.testing.assert_close(transformer.audio.timesteps, torch.zeros(1, 2, 3))

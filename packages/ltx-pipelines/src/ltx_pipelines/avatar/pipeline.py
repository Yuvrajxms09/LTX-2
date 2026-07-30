from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import ConditioningItem, VideoConditionByLatentIndex
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.transformer import X0Model
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import TilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_pipelines.avatar.metrics import MetricsRecorder, TimedDenoiser, tensor_snapshot
from ltx_pipelines.utils.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.blocks import (
    AudioConditioner,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    get_device,
    modality_from_latent_state,
)
from ltx_pipelines.utils.media_io import decode_audio_from_file
from ltx_pipelines.utils.types import DenoisedLatentResult, ModalitySpec, OffloadMode


@dataclass(frozen=True)
class AvatarPromptContext:
    video: torch.Tensor
    audio: torch.Tensor


@dataclass(frozen=True)
class AvatarChunkResult:
    video: Iterator[torch.Tensor]
    audio: Audio
    latent: torch.Tensor


class DrivingAudioDenoiser:
    """Denoise video while exposing fixed driving audio as a clean modality."""

    def __init__(self, video_context: torch.Tensor, audio_context: torch.Tensor) -> None:
        self._video_context = video_context
        self._audio_context = audio_context

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
        if video_state is None or audio_state is None:
            raise ValueError("Driving-audio avatar denoising requires both video and audio latent states")

        video_sigma = sigmas[step_index]
        audio_sigma = torch.zeros_like(video_sigma)
        video = modality_from_latent_state(video_state, self._video_context, video_sigma)
        audio = modality_from_latent_state(audio_state, self._audio_context, audio_sigma)
        denoised_video, denoised_audio = transformer(video=video, audio=audio, perturbations=None)
        return (
            DenoisedLatentResult.result_or_none(denoised_video),
            DenoisedLatentResult.result_or_none(denoised_audio),
        )


class AvatarA2VidPipeline:
    """One-stage distilled image-and-audio-to-video pipeline for avatar experiments.

    The external audio latent is frozen during denoising, so speech drives the
    generated video without being regenerated. Model construction remains
    controllable by the caller: use ``stage.model_context()`` and pass the
    resulting transformer to ``generate_chunk`` to keep the transformer warm
    across chunks.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self.prompt_encoder = PromptEncoder(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.audio_conditioner = AudioConditioner(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.stage = DiffusionStage.from_checkpoint(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            loras=loras,
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.video_decoder = VideoDecoder(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )

    def encode_prompt(
        self,
        prompt: str,
        enhance_prompt: bool,
        image_path: str,
        seed: int,
    ) -> AvatarPromptContext:
        return self.encode_prompts(
            prompts=(prompt,),
            enhance_prompt=enhance_prompt,
            image_path=image_path,
            seed=seed,
        )[0]

    def encode_prompts(
        self,
        prompts: tuple[str, ...],
        enhance_prompt: bool,
        image_path: str,
        seed: int,
    ) -> tuple[AvatarPromptContext, ...]:
        encoded = self.prompt_encoder(
            list(prompts),
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=image_path,
            enhance_prompt_seed=seed,
        )
        return tuple(self._prompt_context(item) for item in encoded)

    def encode_image_conditionings(
        self,
        images: list[ImageConditioningInput],
        height: int,
        width: int,
    ) -> list[ConditioningItem]:
        return self.image_conditioner(
            lambda encoder: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=encoder,
                dtype=self.dtype,
                device=self.device,
            )
        )

    @staticmethod
    def _prompt_context(encoded: EmbeddingsProcessorOutput) -> AvatarPromptContext:
        return AvatarPromptContext(video=encoded.video_encoding, audio=encoded.audio_encoding)

    @staticmethod
    def _pad_audio(audio: Audio, duration_seconds: float) -> Audio:
        required_samples = round(duration_seconds * audio.sampling_rate)
        current_samples = audio.waveform.shape[-1]
        if current_samples >= required_samples:
            waveform = audio.waveform[..., :required_samples]
        else:
            waveform = torch.nn.functional.pad(audio.waveform, (0, required_samples - current_samples))
        return Audio(waveform=waveform, sampling_rate=audio.sampling_rate)

    def generate_chunk(  # noqa: PLR0913
        self,
        transformer: X0Model,
        context: AvatarPromptContext,
        images: list[ImageConditioningInput],
        prefix_latent: torch.Tensor | None,
        prefix_strength: float,
        identity_conditionings: list[ConditioningItem],
        audio_path: str,
        audio_start_time: float,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        sigmas: torch.Tensor,
        recorder: MetricsRecorder,
        chunk_index: int,
        log_denoising_steps: bool,
        tensor_statistics: bool,
        tiling_config: TilingConfig | None = None,
    ) -> AvatarChunkResult:
        assert_resolution(height=height, width=width, is_two_stage=False)
        if (num_frames - 1) % 8 != 0:
            raise ValueError("num_frames must satisfy frames = 8*k + 1")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        generation_duration = num_frames / frame_rate

        with recorder.phase("audio_decode", chunk_index=chunk_index):
            decoded_audio = decode_audio_from_file(
                audio_path,
                self.device,
                start_time=audio_start_time,
                max_duration=generation_duration,
            )
            if decoded_audio is None:
                raise ValueError(f"No audio was decoded from {audio_path!r} at {audio_start_time:.3f}s")
            decoded_audio = self._pad_audio(decoded_audio, generation_duration)

        with recorder.phase("audio_conditioning", chunk_index=chunk_index):
            encoded_audio = self.audio_conditioner(lambda encoder: vae_encode_audio(decoded_audio, encoder, None))
            required_audio_frames = AudioLatentShape.from_duration(
                batch=1,
                duration=generation_duration,
                channels=8,
                mel_bins=16,
            ).frames
            encoded_audio = encoded_audio[:, :, :required_audio_frames]
            if encoded_audio.shape[2] != required_audio_frames:
                raise RuntimeError(
                    f"Audio encoder returned {encoded_audio.shape[2]} latent frames; expected {required_audio_frames}"
                )
            waveform = decoded_audio.waveform.detach().float()
            latent_values = encoded_audio.detach().float()
            recorder.emit(
                "driving_audio_ready",
                chunk_index=chunk_index,
                audio_start_time=audio_start_time,
                duration_seconds=generation_duration,
                sampling_rate=decoded_audio.sampling_rate,
                source_channels=decoded_audio.waveform.shape[1],
                waveform_rms=waveform.square().mean().sqrt().item(),
                waveform_peak=waveform.abs().max().item(),
                encoded_audio=tensor_snapshot(encoded_audio),
                encoded_audio_mean=latent_values.mean().item(),
                encoded_audio_std=latent_values.std().item(),
                encoded_audio_min=latent_values.min().item(),
                encoded_audio_max=latent_values.max().item(),
                audio_sigma_policy="zero_clean",
            )

        if prefix_latent is None:
            with recorder.phase("image_conditioning", chunk_index=chunk_index, image_count=len(images)):
                conditionings = self.encode_image_conditionings(images, height, width)
        else:
            if images:
                raise ValueError("A latent-prefix chunk cannot also use image keyframe conditioning")
            target_shape = VideoLatentShape.from_pixel_shape(
                VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=frame_rate)
            )
            expected_spatial_shape = target_shape.to_torch_shape()
            if prefix_latent.ndim != 5:
                raise ValueError(f"Expected prefix latent shaped (B,C,F,H,W), got {tuple(prefix_latent.shape)}")
            if (
                prefix_latent.shape[0] != expected_spatial_shape[0]
                or prefix_latent.shape[1] != expected_spatial_shape[1]
                or prefix_latent.shape[3:] != expected_spatial_shape[3:]
            ):
                raise ValueError(
                    f"Prefix latent shape {tuple(prefix_latent.shape)} is incompatible with target "
                    f"{tuple(expected_spatial_shape)}"
                )
            if prefix_latent.shape[2] >= target_shape.frames:
                raise ValueError("Prefix latent must leave at least one temporal latent to generate")
            prefix_latent = prefix_latent.to(device=self.device, dtype=self.dtype)
            conditionings = [
                VideoConditionByLatentIndex(
                    latent=prefix_latent,
                    strength=prefix_strength,
                    latent_idx=0,
                )
            ]
            conditionings.extend(identity_conditionings)
            recorder.emit(
                "latent_prefix_ready",
                chunk_index=chunk_index,
                pixel_frames=(prefix_latent.shape[2] - 1) * 8 + 1,
                latent=tensor_snapshot(prefix_latent),
                strength=prefix_strength,
                identity_anchor_count=len(identity_conditionings),
            )

        denoiser = TimedDenoiser(
            DrivingAudioDenoiser(context.video, context.audio),
            recorder=recorder,
            chunk_index=chunk_index,
            enabled=log_denoising_steps,
            tensor_statistics=tensor_statistics,
        )
        sigmas = sigmas.to(dtype=torch.float32, device=self.device)
        with recorder.phase(
            "denoising",
            chunk_index=chunk_index,
            steps=len(sigmas) - 1,
            frames=num_frames,
            width=width,
            height=height,
        ):
            video_state, _ = self.stage.run(
                transformer=transformer,
                denoiser=denoiser,
                sigmas=sigmas,
                noiser=noiser,
                width=width,
                height=height,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(context=context.video, conditionings=conditionings),
                audio=ModalitySpec(
                    context=context.audio,
                    frozen=True,
                    noise_scale=0.0,
                    initial_latent=encoded_audio,
                ),
            )
        if video_state is None:
            raise RuntimeError("Avatar diffusion stage returned no video latent")

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        original_audio = Audio(
            waveform=decoded_audio.waveform.squeeze(0),
            sampling_rate=decoded_audio.sampling_rate,
        )
        return AvatarChunkResult(video=decoded_video, audio=original_audio, latent=video_state.latent)

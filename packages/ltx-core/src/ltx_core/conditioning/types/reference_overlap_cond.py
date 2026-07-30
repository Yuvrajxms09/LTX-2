"""Source-tagged overlap reference conditioning for identity IC-LoRAs."""

import torch

from ltx_core.conditioning.item import ConditioningItem
from ltx_core.conditioning.mask_utils import update_attention_mask
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState


class VideoConditionByReferenceOverlap(ConditioningItem):
    """Append a clean reference frame on the target frame-zero positional grid."""

    def __init__(
        self,
        latent: torch.Tensor,
        source_id: float = 2.0,
        phase_scale: float = 1.0,
        strength: float = 1.0,
    ) -> None:
        self.latent = latent
        self.source_phase = float(source_id) * float(phase_scale)
        self.strength = strength

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.latent)
        reference_token_count = tokens.shape[1]
        if reference_token_count > latent_state.positions.shape[2]:
            raise ValueError(
                f"Reference has {reference_token_count} tokens, but target frame-zero grid "
                f"contains at most {latent_state.positions.shape[2]} tokens"
            )

        positions = latent_state.positions[:, :, :reference_token_count].clone()
        denoise_mask = torch.full(
            (*tokens.shape[:2], 1),
            1.0 - self.strength,
            device=self.latent.device,
            dtype=self.latent.dtype,
        )
        attention_mask = update_attention_mask(
            latent_state=latent_state,
            attention_mask=None,
            num_noisy_tokens=latent_tools.target_shape.token_count(),
            num_new_tokens=reference_token_count,
            batch_size=tokens.shape[0],
            device=self.latent.device,
            dtype=self.latent.dtype,
        )

        return LatentState(
            latent=torch.cat([latent_state.latent, torch.zeros_like(tokens)], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=attention_mask,
            reference_overlap_token_count=reference_token_count,
            reference_source_phase=self.source_phase,
        )

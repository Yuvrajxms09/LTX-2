"""RoPE source-phase tagging for overlap reference tokens."""

import torch


def apply_source_phase(
    positional_embeddings: tuple[torch.Tensor, torch.Tensor],
    reference_token_count: int,
    source_phase: float,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the trailing reference tokens by a source-specific RoPE phase."""
    if reference_token_count <= 0 or source_phase == 0.0:
        return positional_embeddings

    cosine, sine = positional_embeddings
    if reference_token_count > cosine.shape[-2]:
        raise ValueError(
            f"Reference token count {reference_token_count} exceeds positional sequence length {cosine.shape[-2]}"
        )

    dimensions = cosine.shape[-1]
    indices = torch.arange(dimensions, device=cosine.device, dtype=torch.float32)
    phase = source_phase * theta ** (-indices / float(dimensions))
    phase_cosine = phase.cos().to(cosine.dtype)
    phase_sine = phase.sin().to(sine.dtype)
    reference_slice = (..., slice(cosine.shape[-2] - reference_token_count, None), slice(None))

    rotated_cosine = cosine.clone()
    rotated_sine = sine.clone()
    original_cosine = cosine[reference_slice]
    original_sine = sine[reference_slice]
    rotated_cosine[reference_slice] = original_cosine * phase_cosine - original_sine * phase_sine
    rotated_sine[reference_slice] = original_sine * phase_cosine + original_cosine * phase_sine
    return rotated_cosine, rotated_sine

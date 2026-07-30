import torch

from ltx_core.model.transformer.source_phase import apply_source_phase


def test_source_phase_changes_only_trailing_reference_tokens() -> None:
    cosine = torch.ones(1, 2, 5, 4)
    sine = torch.zeros_like(cosine)

    rotated_cosine, rotated_sine = apply_source_phase(
        (cosine, sine),
        reference_token_count=2,
        source_phase=2.0,
    )

    torch.testing.assert_close(rotated_cosine[..., :3, :], cosine[..., :3, :])
    torch.testing.assert_close(rotated_sine[..., :3, :], sine[..., :3, :])
    assert not torch.equal(rotated_cosine[..., 3:, :], cosine[..., 3:, :])
    assert not torch.equal(rotated_sine[..., 3:, :], sine[..., 3:, :])


def test_source_phase_zero_is_noop() -> None:
    cosine = torch.randn(1, 2, 5, 4)
    sine = torch.randn_like(cosine)

    result = apply_source_phase(
        (cosine, sine),
        reference_token_count=2,
        source_phase=0.0,
    )

    assert result[0] is cosine
    assert result[1] is sine

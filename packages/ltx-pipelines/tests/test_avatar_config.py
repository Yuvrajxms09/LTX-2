from pathlib import Path

import pytest

from ltx_pipelines.avatar.config import load_avatar_config


def _write_config(tmp_path: Path) -> Path:
    for name in ("model.safetensors", "avatar.png", "speech.wav", "talking.safetensors"):
        (tmp_path / name).touch()
    (tmp_path / "gemma").mkdir()
    config = tmp_path / "avatar.toml"
    config.write_text(
        """
[model]
checkpoint_path = "model.safetensors"
gemma_root = "gemma"
quantization = "fp8-scaled-mm"
offload = "cpu"
compile = { mode = "reduce-overhead", dynamic = true }

[[model.loras]]
path = "talking.safetensors"
strength = 0.8

[input]
image_path = "avatar.png"
audio_path = "speech.wav"
prompt = "A speaking avatar"
chunk_prompts = ["First window", "Second window"]

[generation]
generation_frames = 49
overlap_frames = 9

[output]
directory = "output"
""",
        encoding="utf-8",
    )
    return config


def test_load_avatar_config_resolves_paths_and_compile_settings(tmp_path: Path) -> None:
    config = load_avatar_config(_write_config(tmp_path))

    assert config.model.checkpoint_path == str((tmp_path / "model.safetensors").resolve())
    assert config.model.loras[0].strength == 0.8
    assert config.model.offload == "cpu"
    assert config.model.compile is not None
    assert config.model.compile.mode == "reduce-overhead"
    assert config.input.chunk_prompts == ("First window", "Second window")
    assert config.output.directory == str((tmp_path / "output").resolve())


def test_load_avatar_config_applies_overrides(tmp_path: Path) -> None:
    config = load_avatar_config(
        _write_config(tmp_path),
        overrides=("generation.generation_frames=33", "generation.overlap_frames=8"),
    )

    assert config.generation.generation_frames == 33
    assert config.generation.overlap_frames == 8


def test_load_avatar_config_accepts_exact_latent_prefix_mode(tmp_path: Path) -> None:
    config = load_avatar_config(
        _write_config(tmp_path),
        overrides=(
            'generation.continuation_mode="latent-prefix"',
            "generation.generation_frames=121",
            "generation.overlap_frames=17",
        ),
    )

    assert config.generation.continuation_mode == "latent-prefix"
    assert config.generation.overlap_frames == 17


def test_load_avatar_config_accepts_reference_reset_without_overlap(tmp_path: Path) -> None:
    config = load_avatar_config(
        _write_config(tmp_path),
        overrides=(
            'generation.continuation_mode="reference-reset"',
            "generation.overlap_frames=0",
        ),
    )

    assert config.generation.continuation_mode == "reference-reset"
    assert config.generation.overlap_frames == 0


def test_load_avatar_config_rejects_reference_reset_overlap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be 0"):
        load_avatar_config(
            _write_config(tmp_path),
            overrides=('generation.continuation_mode="reference-reset"',),
        )


def test_load_avatar_config_rejects_non_aligned_latent_prefix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"8\*k \+ 1"):
        load_avatar_config(
            _write_config(tmp_path),
            overrides=(
                'generation.continuation_mode="latent-prefix"',
                "generation.overlap_frames=8",
            ),
        )


def test_load_avatar_config_accepts_soft_latent_prefix_strength(tmp_path: Path) -> None:
    config = load_avatar_config(
        _write_config(tmp_path),
        overrides=(
            'generation.continuation_mode="latent-prefix"',
            "generation.overlap_frames=17",
            "generation.overlap_strength=0.5",
        ),
    )

    assert config.generation.overlap_strength == 0.5


def test_load_avatar_config_rejects_latent_prefix_strength_above_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"overlap_strength must be at most 1\.0"):
        load_avatar_config(
            _write_config(tmp_path),
            overrides=(
                'generation.continuation_mode="latent-prefix"',
                "generation.overlap_frames=17",
                "generation.overlap_strength=1.1",
            ),
        )


def test_load_avatar_config_accepts_negative_identity_anchor(tmp_path: Path) -> None:
    config = load_avatar_config(
        _write_config(tmp_path),
        overrides=(
            'generation.continuation_mode="latent-prefix"',
            "generation.overlap_frames=17",
            "generation.identity_anchor_strength=0.5",
        ),
    )

    assert config.generation.identity_anchor_strength == 0.5


def test_load_avatar_config_rejects_identity_anchor_without_latent_prefix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires latent-prefix"):
        load_avatar_config(
            _write_config(tmp_path),
            overrides=("generation.identity_anchor_strength=0.5",),
        )


def test_load_avatar_config_rejects_identity_anchor_strength_above_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        load_avatar_config(
            _write_config(tmp_path),
            overrides=(
                'generation.continuation_mode="latent-prefix"',
                "generation.overlap_frames=17",
                "generation.identity_anchor_strength=1.1",
            ),
        )


def test_load_avatar_config_rejects_unknown_settings(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace("overlap_frames = 9", "overlap_frames = 9\novrelap = 8"))

    with pytest.raises(ValueError, match="ovrelap"):
        load_avatar_config(path)


def test_load_avatar_config_rejects_invalid_model_frame_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"8\*k \+ 1"):
        load_avatar_config(_write_config(tmp_path), overrides=("generation.generation_frames=48",))


def test_load_avatar_config_rejects_invalid_offload_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"model\.offload"):
        load_avatar_config(_write_config(tmp_path), overrides=('model.offload="gpu"',))

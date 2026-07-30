from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from ltx_core.model.transformer.compiling import CompilationConfig

_SECTIONS = {"model", "input", "generation", "output", "diagnostics"}
_DEFAULT_DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)


@dataclass(frozen=True)
class LoraConfig:
    path: str
    strength: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    checkpoint_path: str
    gemma_root: str
    loras: tuple[LoraConfig, ...] = ()
    quantization: str | None = None
    offload: str = "none"
    compile: CompilationConfig | None = None
    warm_transformer: bool = True


@dataclass(frozen=True)
class InputConfig:
    image_path: str
    audio_path: str
    prompt: str
    enhance_prompt: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    width: int = 512
    height: int = 512
    frame_rate: float = 25.0
    generation_frames: int = 49
    overlap_frames: int = 9
    continuation_mode: str = "image-keyframes"
    reference_strength: float = 1.0
    overlap_strength: float = 1.0
    identity_anchor_strength: float = 0.0
    seed: int = 10
    seed_stride: int = 1
    sigmas: tuple[float, ...] = _DEFAULT_DISTILLED_SIGMAS
    max_chunks: int | None = None


@dataclass(frozen=True)
class OutputConfig:
    directory: str
    crf: int = 19
    preset: str = "veryfast"
    save_conditioning_frames: bool = True
    allow_existing: bool = False


@dataclass(frozen=True)
class DiagnosticsConfig:
    log_level: str = "INFO"
    jsonl_metrics: bool = True
    synchronize_cuda: bool = True
    log_denoising_steps: bool = True
    tensor_statistics: bool = False


@dataclass(frozen=True)
class AvatarConfig:
    model: ModelConfig
    input: InputConfig
    generation: GenerationConfig
    output: OutputConfig
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    def validate(self) -> None:
        self._validate_generation()
        if self.model.quantization not in {None, "fp8-cast", "fp8-scaled-mm"}:
            raise ValueError("model.quantization must be fp8-cast, fp8-scaled-mm, or omitted")
        if self.model.offload not in {"none", "cpu", "disk"}:
            raise ValueError("model.offload must be none, cpu, or disk")
        if not 0 <= self.output.crf <= 51:
            raise ValueError("output.crf must be between 0 and 51")
        if self.diagnostics.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("diagnostics.log_level is invalid")

    def _validate_generation(self) -> None:
        generation = self.generation
        if generation.width <= 0 or generation.width % 32 != 0:
            raise ValueError("generation.width must be positive and divisible by 32")
        if generation.height <= 0 or generation.height % 32 != 0:
            raise ValueError("generation.height must be positive and divisible by 32")
        if generation.frame_rate <= 0:
            raise ValueError("generation.frame_rate must be positive")
        if not float(generation.frame_rate).is_integer():
            raise ValueError("generation.frame_rate must be a whole number because the MP4 encoder uses integer FPS")
        if generation.generation_frames < 9 or (generation.generation_frames - 1) % 8 != 0:
            raise ValueError("generation.generation_frames must satisfy frames = 8*k + 1 and be at least 9")
        if not 0 <= generation.overlap_frames < generation.generation_frames:
            raise ValueError("generation.overlap_frames must be in [0, generation_frames)")
        if generation.overlap_frames >= generation.generation_frames - 1:
            raise ValueError("generation.overlap_frames must leave at least two newly generated frames")
        if generation.continuation_mode not in {"image-keyframes", "latent-prefix"}:
            raise ValueError("generation.continuation_mode must be image-keyframes or latent-prefix")
        if generation.continuation_mode == "latent-prefix":
            self._validate_latent_prefix()
        if generation.reference_strength < 0 or generation.overlap_strength < 0:
            raise ValueError("conditioning strengths must be non-negative")
        self._validate_identity_anchor()
        if generation.seed_stride < 0:
            raise ValueError("generation.seed_stride must be non-negative")
        if generation.max_chunks is not None and generation.max_chunks < 1:
            raise ValueError("generation.max_chunks must be at least 1")
        self._validate_sigma_schedule()

    def _validate_sigma_schedule(self) -> None:
        sigmas = self.generation.sigmas
        if len(sigmas) < 2 or sigmas[-1] != 0:
            raise ValueError("generation.sigmas must contain at least two values and end at 0")
        if any(left < right for left, right in pairwise(sigmas)):
            raise ValueError("generation.sigmas must be monotonically non-increasing")

    def _validate_latent_prefix(self) -> None:
        generation = self.generation
        if generation.overlap_frames < 1 or (generation.overlap_frames - 1) % 8 != 0:
            raise ValueError("generation.overlap_frames must satisfy frames = 8*k + 1 in latent-prefix mode")
        prefix_latent_frames = (generation.overlap_frames - 1) // 8 + 1
        generation_latent_frames = (generation.generation_frames - 1) // 8 + 1
        if prefix_latent_frames >= generation_latent_frames:
            raise ValueError("latent-prefix overlap must leave at least one temporal latent to generate")
        if generation.overlap_strength > 1.0:
            raise ValueError("generation.overlap_strength must be at most 1.0 in latent-prefix mode")

    def _validate_identity_anchor(self) -> None:
        generation = self.generation
        if not 0 <= generation.identity_anchor_strength <= 1:
            raise ValueError("generation.identity_anchor_strength must be between 0 and 1")
        if generation.identity_anchor_strength > 0 and generation.continuation_mode != "latent-prefix":
            raise ValueError("generation.identity_anchor_strength requires latent-prefix continuation")


def _expect_table(data: dict[str, Any], key: str, *, required: bool = True) -> dict[str, Any]:
    value = data.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"[{key}] must be a TOML table")
    return value


def _reject_unknown(table: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"Unknown {section} setting(s): {', '.join(unknown)}")


def _required_string(table: dict[str, Any], key: str, section: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section}.{key} must be a non-empty string")
    return value


def _resolve_path(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _parse_compile(value: object) -> CompilationConfig | None:
    if value is None or value is False:
        return None
    if value is True:
        return CompilationConfig()
    if not isinstance(value, dict):
        raise ValueError("model.compile must be a boolean or table")
    allowed = {"mode", "backend", "fullgraph", "dynamic", "inductor_config", "dynamo_config"}
    _reject_unknown(value, allowed, "model.compile")
    return CompilationConfig(**value)


def _parse_loras(value: object, base_dir: Path) -> tuple[LoraConfig, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("model.loras must be an array of tables")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"model.loras[{index}] must be a table")
        _reject_unknown(item, {"path", "strength"}, f"model.loras[{index}]")
        path = _required_string(item, "path", f"model.loras[{index}]")
        result.append(LoraConfig(path=_resolve_path(path, base_dir), strength=float(item.get("strength", 1.0))))
    return tuple(result)


def _parse_toml_value(raw: str) -> object:
    try:
        return tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        return raw


def _apply_overrides(data: dict[str, Any], overrides: tuple[str, ...]) -> dict[str, Any]:
    result = copy.deepcopy(data)
    for override in overrides:
        key, separator, raw_value = override.partition("=")
        if not separator or "." not in key:
            raise ValueError(f"Invalid override {override!r}; expected section.key=value")
        section, setting = key.split(".", 1)
        if section not in _SECTIONS or "." in setting:
            raise ValueError(f"Invalid override path {key!r}")
        section_data = result.setdefault(section, {})
        if not isinstance(section_data, dict):
            raise ValueError(f"Cannot override {key!r}; [{section}] is not a table")
        section_data[setting] = _parse_toml_value(raw_value)
    return result


def load_avatar_config(path: str | Path, overrides: tuple[str, ...] = ()) -> AvatarConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as config_file:
        raw = _apply_overrides(tomllib.load(config_file), overrides)
    unknown_sections = sorted(set(raw) - _SECTIONS)
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {', '.join(unknown_sections)}")

    model = _expect_table(raw, "model")
    inputs = _expect_table(raw, "input")
    generation = _expect_table(raw, "generation", required=False)
    output = _expect_table(raw, "output")
    diagnostics = _expect_table(raw, "diagnostics", required=False)

    _reject_unknown(
        model,
        {"checkpoint_path", "gemma_root", "loras", "quantization", "offload", "compile", "warm_transformer"},
        "model",
    )
    _reject_unknown(inputs, {"image_path", "audio_path", "prompt", "enhance_prompt"}, "input")
    _reject_unknown(
        generation,
        {
            "width",
            "height",
            "frame_rate",
            "generation_frames",
            "overlap_frames",
            "continuation_mode",
            "reference_strength",
            "overlap_strength",
            "identity_anchor_strength",
            "seed",
            "seed_stride",
            "sigmas",
            "max_chunks",
        },
        "generation",
    )
    _reject_unknown(output, {"directory", "crf", "preset", "save_conditioning_frames", "allow_existing"}, "output")
    _reject_unknown(
        diagnostics,
        {"log_level", "jsonl_metrics", "synchronize_cuda", "log_denoising_steps", "tensor_statistics"},
        "diagnostics",
    )

    base_dir = config_path.parent
    model_config = ModelConfig(
        checkpoint_path=_resolve_path(_required_string(model, "checkpoint_path", "model"), base_dir),
        gemma_root=_resolve_path(_required_string(model, "gemma_root", "model"), base_dir),
        loras=_parse_loras(model.get("loras"), base_dir),
        quantization=model.get("quantization"),
        offload=str(model.get("offload", "none")),
        compile=_parse_compile(model.get("compile")),
        warm_transformer=bool(model.get("warm_transformer", True)),
    )
    input_config = InputConfig(
        image_path=_resolve_path(_required_string(inputs, "image_path", "input"), base_dir),
        audio_path=_resolve_path(_required_string(inputs, "audio_path", "input"), base_dir),
        prompt=_required_string(inputs, "prompt", "input"),
        enhance_prompt=bool(inputs.get("enhance_prompt", False)),
    )
    generation_values = dict(generation)
    if "sigmas" in generation_values:
        sigma_values = generation_values["sigmas"]
        if not isinstance(sigma_values, list):
            raise ValueError("generation.sigmas must be an array")
        generation_values["sigmas"] = tuple(float(value) for value in sigma_values)
    generation_config = GenerationConfig(**generation_values)
    output_config = OutputConfig(
        directory=_resolve_path(_required_string(output, "directory", "output"), base_dir),
        crf=int(output.get("crf", 19)),
        preset=str(output.get("preset", "veryfast")),
        save_conditioning_frames=bool(output.get("save_conditioning_frames", True)),
        allow_existing=bool(output.get("allow_existing", False)),
    )
    diagnostics_config = DiagnosticsConfig(**diagnostics)
    config = AvatarConfig(
        model=model_config,
        input=input_config,
        generation=generation_config,
        output=output_config,
        diagnostics=diagnostics_config,
    )
    config.validate()
    required_paths = {
        "model.checkpoint_path": config.model.checkpoint_path,
        "model.gemma_root": config.model.gemma_root,
        "input.image_path": config.input.image_path,
        "input.audio_path": config.input.audio_path,
        **{f"model.loras[{index}].path": lora.path for index, lora in enumerate(config.model.loras)},
    }
    missing = [f"{key}: {value}" for key, value in required_paths.items() if not Path(value).exists()]
    if missing:
        raise ValueError(f"Configured path(s) do not exist: {'; '.join(missing)}")
    return config

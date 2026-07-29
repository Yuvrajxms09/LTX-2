# Colab Setup for the Chunked Avatar Prototype

This notebook flow expects a high-memory CUDA runtime and a repository branch
that already contains `ltx_pipelines.avatar`. Cloning the upstream Lightricks
repository alone will not include this experimental package until it is merged.

Recommended capacity:

- An A100 80 GB or a GPU with more VRAM for the BF16 checkpoint and warm
  transformer.
- 80 GB or more system RAM for model loading.
- At least 120 GB free disk space.

Standard free Colab T4/L4 runtimes are not suitable for this 22B workflow.

An A100 supports BF16 well but does not have Hopper's native FP8 tensor cores.
It should run this prototype, but it should not be expected to reproduce
Daydream's H100 timing.

## 1. Inspect the runtime

```bash
!nvidia-smi
!free -h
!df -h /content
```

## 2. Clone the branch containing the avatar runner

Replace the URL and branch with the fork/branch containing this implementation.

```bash
!git clone --branch avatar-prototype --single-branch \
    https://github.com/YOUR_ACCOUNT/LTX-2.git /content/LTX-2
%cd /content/LTX-2
!test -f packages/ltx-pipelines/src/ltx_pipelines/avatar/runner.py
```

## 3. Install the complete repository environment

Run this before importing PyTorch in the notebook kernel. This installs all
workspace packages, including the trainer, plus every root and package-level
development dependency. Dependencies come from the committed `uv.lock` rather
than being resolved to untested newer versions:

```bash
!python -m pip install -q --upgrade pip uv

!uv export \
    --directory /content/LTX-2 \
    --frozen \
    --all-packages \
    --all-groups \
    --no-hashes \
    --no-emit-workspace \
    --no-emit-package ltx-kernels \
    --output-file /content/LTX-2/requirements-colab-all.txt

!uv pip install --system \
    --requirement /content/LTX-2/requirements-colab-all.txt

!uv pip install --system --no-deps \
    -e /content/LTX-2/packages/ltx-core \
    -e /content/LTX-2/packages/ltx-pipelines \
    -e /content/LTX-2/packages/ltx-trainer
```

Install the repository's optional compiled CUDA kernels separately so a build
failure is visible. An A100 has compute capability 8.0:

```bash
%env TORCH_CUDA_ARCH_LIST=8.0
!uv pip install --system \
    -e /content/LTX-2/packages/ltx-kernels \
    --no-deps \
    --no-build-isolation
```

Verify the subprocess environment:

```bash
!python - <<'PY'
import torch
import ltx_core
import ltx_kernels
import ltx_pipelines.avatar.runner
import ltx_trainer

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name())
print("vram_gib:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
print("all LTX workspace packages imported")
PY
```

The kernel build requires Linux, `nvcc`, and a CUDA toolkit compatible with the
installed PyTorch CUDA build. Check `!nvcc --version` if that cell fails. The
avatar baseline does not call the custom multi-GPU/blockwise kernels directly,
but this setup installs them because it intentionally covers the complete
repository. The blockwise FP8 GEMM path is unsupported on Ampere/A100; installing
the package does not make that path usable on this GPU.

## 4. Authenticate with Hugging Face

Accept the Gemma license on its Hugging Face model page first. Store a read
token as a Colab secret named `HF_TOKEN`; do not paste it into a notebook cell.

```python
from google.colab import userdata
from huggingface_hub import login

login(token=userdata.get("HF_TOKEN"), add_to_git_credential=False)
```

## 5. Download only the required weights

The avatar runner is one-stage, so it does not need a spatial upscaler or the
separate distilled LoRA.

```python
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

model_dir = Path("/content/LTX-2/models/ltx-2.3")
gemma_dir = Path("/content/LTX-2/models/gemma-3-12b")
model_dir.mkdir(parents=True, exist_ok=True)

checkpoint = hf_hub_download(
    repo_id="Lightricks/LTX-2.3",
    filename="ltx-2.3-22b-distilled-1.1.safetensors",
    local_dir=model_dir,
)
gemma = snapshot_download(
    repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
    local_dir=gemma_dir,
)

print("checkpoint:", checkpoint)
print("gemma:", gemma)
```

The checkpoint is approximately 46.1 GB and Gemma is approximately 24.4 GB.

## 6. Upload the avatar image and TTS waveform

```python
from pathlib import Path
from google.colab import files

input_dir = Path("/content/LTX-2/inputs")
input_dir.mkdir(parents=True, exist_ok=True)

print("Upload the reference image")
image_upload = files.upload()
image_name, image_bytes = next(iter(image_upload.items()))
(input_dir / "avatar.png").write_bytes(image_bytes)

print("Upload the TTS audio")
audio_upload = files.upload()
audio_name, audio_bytes = next(iter(audio_upload.items()))
(input_dir / "tts.wav").write_bytes(audio_bytes)

print("image source:", image_name)
print("audio source:", audio_name)
```

Use a WAV input for the first test to avoid codec-seeking ambiguity. Mono TTS
audio is accepted and duplicated to stereo automatically for the LTX audio VAE.

## 7. Create a two-chunk correctness configuration

No LoRA is required for the external driving-audio baseline. The full BF16
transformer can exceed the A100's usable 79.25 GiB once runtime tensors are
included. For a quality-first run, keep BF16 and stream model layers from
system RAM with `offload = "cpu"`. Gemma is freed after prompt encoding before
the diffusion transformer is built.

```toml
%%writefile /content/LTX-2/avatar-smoke.toml
[model]
checkpoint_path = "/content/LTX-2/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
gemma_root = "/content/LTX-2/models/gemma-3-12b"
offload = "cpu"
warm_transformer = true
compile = false

[input]
image_path = "/content/LTX-2/inputs/avatar.png"
audio_path = "/content/LTX-2/inputs/tts.wav"
prompt = "A stable front-facing conversational avatar speaks naturally to the camera. The camera and background remain static, with natural blinking and subtle head motion."
enhance_prompt = false

[generation]
width = 512
height = 512
frame_rate = 25.0
generation_frames = 33
overlap_frames = 8
reference_strength = 1.0
overlap_strength = 1.0
seed = 10
seed_stride = 1
sigmas = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
max_chunks = 2

[output]
directory = "/content/LTX-2/outputs/avatar-a100-bf16-offload"
crf = 19
preset = "veryfast"
save_conditioning_frames = true
allow_existing = false

[diagnostics]
log_level = "INFO"
jsonl_metrics = true
synchronize_cuda = true
log_denoising_steps = true
tensor_statistics = false
```

## 8. Validate and run

```bash
%env PYTORCH_ALLOC_CONF=expandable_segments:True
%env TOKENIZERS_PARALLELISM=false

!python -m ltx_pipelines.avatar.runner \
    --config /content/LTX-2/avatar-smoke.toml \
    --dry-run

!python -m ltx_pipelines.avatar.runner \
    --config /content/LTX-2/avatar-smoke.toml
```

Every output directory must be new unless `allow_existing=true`.

## 9. Inspect output and metrics

```python
import json
from pathlib import Path
from IPython.display import Video, display

run_dir = Path("/content/LTX-2/outputs/avatar-a100-bf16-offload")
manifest = json.loads((run_dir / "manifest.json").read_text())

print(json.dumps(manifest, indent=2))
for chunk in sorted(run_dir.glob("chunk_*.mp4")):
    print(chunk.name)
    display(Video(str(chunk), embed=True))
```

For a failed run, print the execution boundary, completed/failed phases, first
denoising preflight, and complete traceback:

```python
records = [
    json.loads(line)
    for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    if line.strip()
]
important_events = {"run_started", "phase", "denoising_preflight", "run_failed"}
for record in records:
    if record["event"] in important_events:
        print(json.dumps(record, indent=2))
```

Check `time_to_first_chunk_seconds`, following-chunk `denoising_fps`,
`real_time_factor`, `meets_realtime_deadline`, `deadline_margin_seconds`, and
the overlap/boundary continuity metrics.

## 10. Run the optimized experiment

After the smoke output is correct, create a second config with:

```toml
[model]
checkpoint_path = "/content/LTX-2/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
gemma_root = "/content/LTX-2/models/gemma-3-12b"
offload = "cpu"
warm_transformer = true
compile = { mode = "reduce-overhead", fullgraph = false, dynamic = true }
```

Use `generation_frames=49`, `overlap_frames=9`, `max_chunks=3`, and a new
output directory such as `outputs/avatar-optimized`. The first chunk includes
compilation cost; judge steady-state speed from chunks 1 and 2.

CPU offload is the recommended A100 quality baseline because it keeps the
official BF16 weights unchanged. It requires substantial system RAM and is
much slower because transformer layers are streamed to the GPU.

After validating quality, a separate memory-versus-quality experiment can
disable offload and enable weight-only FP8 cast:

```toml
offload = "none"
quantization = "fp8-cast"
```

On an A100, `fp8-cast` stores selected transformer weights in FP8 and upcasts
them for BF16 linear operations. It can reduce VRAM, but it is not native FP8
matrix-multiplication acceleration and may reduce throughput. Compare its
output directly with the CPU-offloaded BF16 baseline before adopting it.

Do not use `fp8-scaled-mm` with the official 46.1 GB monolithic BF16
checkpoint. That mode expects native FP8 scale tensors and appropriate FP8
hardware. Scope's separated transformer-only FP8 checkpoint is not a substitute
because this pipeline also loads its text projection and audio/video VAEs from
the monolithic checkpoint.

Daydream's `ltx-2.3-id-lora-talkvid-3k.safetensors` is an ID-LoRA for
reference-speaker audio conditioning. It is not required for external TTS
driving mode and should not be added to this baseline.

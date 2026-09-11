# LTX-2.5 A2V lip-sync LoRA readiness audit

Audit date: 2026-09-11

Scope: the local `LTX-2` repository, `packages/ltx-trainer`, the current
`a2v_lipsync_lora.yaml`, and the prepared `training_clips_612` dataset.

## Executive verdict

The proposed mode is the correct native LTX formulation for the requested
contract:

```text
first-frame image + frozen speech audio -> generated video
```

The current `FlexibleStrategy` expresses that contract as:

```text
video.is_generated = true
audio.is_generated = false
video condition = first_frame, probability 1.0
```

This is technically supported by the trainer. The transformer has an explicit
video-query/audio-key-value path (`audio_to_video_attn`), and the video-only
loss does not require an audio target. The local dataset is also mechanically
consistent for a first smoke test.

However, I would not launch the expensive run until the remaining runtime
inputs are supplied. There are three operational blockers and three quality
risks; the configuration/documentation defects found during this audit have
been corrected in the fork:

1. The model, VAE, text-encoder, precomputed-data, and validation paths are
   still placeholders.
2. There is no held-out validation identity/audio pair yet, so a successful
   training loss would not prove general image+speech lip-sync.
3. The objective is a full-video latent MSE with only the first frame excluded;
   it has no mouth-region weighting, lip-landmark loss, or explicit sync loss.

High VRAM solves capacity/OOM risk only. It does not solve wrong checkpoint
pairing, stale latents, target-module mistakes, train/validation leakage, or a
loss that underweights the mouth.

## What is already correct

### Dataset media invariants

The current folder contains 19 MP4s. The media audit shows:

- video: 1280x720, 25 fps, 153 decoded frames, 6.12 seconds;
- audio: embedded AAC, 48 kHz, stereo, starts at 0;
- audio container duration is about 6.13–6.20 seconds because of AAC frame
  padding, but the preprocessing path trims audio to the processed video
  duration (`153 / 25 = 6.12 s`);
- the exact Tom Cruise replacement is present and is the 6.12-second file;
- there are no `.pt` precomputed tensors yet, so preprocessing has not been
  accidentally mixed with an older model/bucket.

The AAC padding is not itself a training defect. `MediaDataset` computes the
target duration from the processed video frames, trims the extracted track,
and pads only if the track is shorter. This is the right pairing behavior for
the current embedded-audio manifest.

The only harmless directory noise is `.DS_Store`; it is not referenced by the
manifest and will not be loaded as a sample.

### Manifest and source-of-truth pairing

`dataset_manifest.jsonl` has 19 rows and only `video` plus neutral `caption`.
That is the right shape for this experiment. The official trainer convention
recognizes `video` as the target video, `caption` as text conditioning, and
auto-extracts the target audio when there is no explicit `audio` column.
See the [official dataset-preparation guide](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/dataset-preparation.md).

Using the same MP4 as the source for both video and audio avoids a common
failure mode: a separately trimmed WAV that is a few frames early or late.
Do not add an `audio` column for these files unless its offset is independently
verified.

The neutral captions are a deliberate and correct specialization for this
goal. They describe the visible talking-head setup without putting the exact
dialogue in text. If the transcript is put into the caption, the model can
partly learn to use text as a shortcut instead of learning to use the supplied
audio. This differs from character-specific LTX-2.3 recipes, which often put
the transcript in the prompt; that recipe is not the same objective as a
general image + arbitrary speech lip-sync model.

### First-frame condition

The training strategy correctly derives the first-frame condition from each
target video's first latent frame; the PNG review exports are not a second
training modality. `probability: 1.0` is correct for the desired inference
contract. With `probability: 0.5`, half the batches would remain T2V-like and
the model would receive a weaker signal for arbitrary input-image identity.

The existing first-frame QA is still the right acceptance test: inspect frame 0
and frames around 0.2, 0.4, and 1.0 seconds, and reject a clip if the intended
speaker is not stable, the mouth is obscured/soft, a cut/fade/subtitle/watermark
dominates the opening, or speech is already clipped. :codex-annotation{index="1"}

The current first-frame review is a pass for all 19 clips as a hard gate, but
not a clean-dataset pass. Known quality flags remain:

- Ronaldo m1: lens flare/darker face;
- Goggins: more 3/4 profile and microphone near the mouth;
- Emma, Nicole, and Veronika: visible overlays/subtitles/watermarks;
- Tom Cruise: begins essentially at speech onset, with little/no neutral lead-in;
- Tom Holland host: wider shot, so the mouth has fewer pixels.

Keep these for the first smoke test if desired, but do not call this the final
quality dataset. Text overlays and watermarks can be learned as visual
correlations, and the wide/profile samples reduce the useful mouth signal.

## Code-path assessment

### Pristine upstream comparison

For this audit, the official repository was cloned as:

```text
/Users/yuvraj/Desktop/ltx/ltx2.5/ltx2_org_code
```

Both repositories resolve to commit `a95ab856bf29407b6b066ede0abe1846050db56c`
(`Automated PR - 2026-08-25`). This makes the comparison a clean audit of the
local modifications, not a comparison against a different upstream revision.

The following files are byte-identical between the pristine clone and the
working repository:

- `packages/ltx-trainer/scripts/train.py`;
- `packages/ltx-trainer/scripts/process_dataset.py`;
- `packages/ltx-trainer/scripts/process_videos.py`;
- `packages/ltx-trainer/src/ltx_trainer/datasets.py`;
- `packages/ltx-trainer/src/ltx_trainer/training_strategies/flexible.py`;
- `packages/ltx-trainer/src/ltx_trainer/validation_runner.py`;
- `packages/ltx-trainer/src/ltx_trainer/model_loader.py`;
- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`.

The modified repository changes `config.py`, `config_display.py`, and
`trainer.py`, adds the specialized YAML, and updates documentation. The
trainer changes add runtime logging, first-batch metadata checks, LoRA target
audits, gradient/memory traces, and non-finite failure checks. They do not
change noise construction, modality freezing, first-frame masking, transformer
forwarding, loss computation, dataset preprocessing, or validation sampling.

Therefore the direct answer is: **we did not use the wrong training script**.
The official `scripts/train.py` plus `FlexibleStrategy` is the correct native
entry point for this experiment. Writing a separate `train_lipsync.py` would
duplicate existing infrastructure without changing the learning signal. The
custom trainer is safe to use as a diagnostic fork, subject to the preflight
checks below.

The pristine upstream `a2v_lora.yaml` is a generic audio-to-video profile: it
freezes audio and generates video, but it has no `first_frame` condition and
uses broad `to_k`/`to_q`/`to_v` patterns. It is not the exact image-conditioned
talking-head profile. The local `a2v_lipsync_lora.yaml` correctly adds the
always-on first-frame condition and narrows the adapters to video attention plus
the video-query/audio-key-value `audio_to_video_attn` path.

The remaining concern is not that the script is wrong; it is that the native
loss is still generic full-video flow-matching MSE. It is a valid baseline for
testing whether LTX-2.5's existing A2V representation can be specialized, but
it is not a dedicated phoneme/lip-sync objective. A failure to achieve reliable
mouth timing after this run would require a loss/data design change, not merely
more logging or a larger GPU.

### Training objective and modality flow

`FlexibleStrategy._process_modality()` keeps a conditioning-only modality clean:
sigma is zero, no noise is added, and no loss target is produced. The video
modality receives flow-matching noise and targets. `compute_loss()` therefore
reduces to video loss only. This correctly implements frozen-audio A2V.

The first-frame condition replaces the first spatial latent frame with the
clean target latent, sets its timestep to zero, and removes those tokens from
the loss. The remaining video tokens are trained. That is aligned with the
native LTX image-to-video conditioning mechanism.

The target is still the whole video latent sequence, not only the mouth. This
is the largest modeling limitation in the current approach. The trainer has no
face detector, mouth mask, landmark extractor, phoneme alignment loss, or
audio-visual sync discriminator. The LoRA can learn to use the existing A2V
representation and improve a talking-head prior, but this code cannot promise
perfect phoneme-level sync from the loss alone.

### Transformer attention direction

In [the local transformer block](../../../../Desktop/ltx/ltx2.5/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/transformer.py:102):

- `attn1` is video self-attention;
- `attn2` is video-to-text cross-attention;
- `audio_attn1` is audio self-attention;
- `audio_attn2` is audio-to-text cross-attention;
- `audio_to_video_attn` is Q=video, K/V=audio;
- `video_to_audio_attn` is Q=audio, K/V=video.

The direct path needed by this experiment is `audio_to_video_attn`. The native
block executes A2V and then V2A whenever both modalities are present. Because
the resulting audio hidden state is passed into the next block, the V2A path can
affect later video states indirectly even when audio has no supervised loss.
Therefore the statement that `video_to_audio_attn` is simply unused by the
video loss would be incorrect.

Omitting LoRA from `video_to_audio_attn` is still the right first choice for this
experiment, but for a different reason: it keeps adaptation causally focused on
audio-to-video and avoids giving the model an easy training shortcut in which
noisy/target-containing video features are adapted into the audio stream and
then fed back into later video blocks. The frozen pretrained V2A path may still
run as part of the base model; only its weights remain fixed. If the focused run
underperforms, compare an explicit broad-attention ablation on held-out audio
before adopting it.

The current target list therefore has a reasonable focused interpretation:

```text
video self-attention + video text cross-attention + audio-to-video attention
```

Before this audit, the comment at lines 50–52 of
[`a2v_lipsync_lora.yaml`](../../../../Desktop/ltx/ltx2.5/LTX-2/packages/ltx-trainer/configs/a2v_lipsync_lora.yaml:50)
was false: `attn1` and `attn2` do not match `audio_attn1` or `audio_attn2`.
The same ambiguity was repeated in the modified training-mode documentation.
Those comments and the reverse-path rationale have now been corrected in the
fork. Do not add audio self/text targets just because an old comment said they
were present. Decide explicitly whether the experiment is focused or broad.

My senior recommendation for the first native experiment is the focused set
already represented by the actual YAML: video attention plus
`audio_to_video_attn`, with audio self-attention and reverse V2A omitted. This
is a causal/regularized baseline, not a claim that reverse V2A is mathematically
disconnected from the video output. If that run underperforms, compare an
explicit broad-attention ablation rather than silently expanding every
`to_k`/`to_q`/`to_v` target.

### LoRA application audit

The trainer now audits requested weighted module names before calling PEFT and
audits actual tuner layers afterward. That is valuable. It should be treated
as a hard preflight condition:

- every requested target must either match intentionally or be intentionally
  omitted;
- the log must show the actual number of matched modules;
- the log must show the intended `video_to_audio_attn` policy (zero for the
  focused causal experiment);
- trainable parameter count must be nonzero and consistent with the expected
  rank and number of transformer blocks.

The fork now fails on unmatched targets instead of continuing with a smaller
adapter. For this exact LTX-2.5 source architecture, the current names should
match, but the actual count must still be checked against the loaded checkpoint
in Colab. The reverse-V2A diagnostic now states the accurate behavior: reverse
V2A can influence later video blocks indirectly, so its exclusion is a deliberate
causal/shortcut-control choice rather than an assertion that the path is unused.

### Gradient and trace changes

`py_compile`, `ruff check`, and `git diff --check` pass for the modified trainer
files. The periodic diagnostics are useful for a one-GPU Colab run:

- first-batch keys, shapes, dtypes, and duration metadata;
- loss, learning rate, sigma range, gradient norm, and GPU memory;
- immediate failure on non-finite loss/gradient norm.

The gradient clipping return type is documented by Accelerate as a Tensor, so
the current conversion is compatible with the supported API; see the
[Accelerate reference](https://huggingface.co/docs/accelerate/main/package_reference/accelerator#clip_grad_norm_).

One limitation: the code logs only a duration comparison, not a sample ID in
that alignment warning. If a batch has a mismatch, the operator must inspect
the first-batch debug dump to identify the file. This is acceptable for the
smoke test but is the first diagnostic improvement I would make before scaling
the dataset.

## Hard blockers before training

### 1. Replace every placeholder

The YAML still contains:

```text
path/to/ltx-2.5-transformer.safetensors
path/to/gemma-text-encoder
path/to/ltx-2.5-video-vae.safetensors
path/to/ltx-2.5-audio-vae.safetensors
/path/to/preprocessed/data
/path/to/validation/identity.png
/path/to/validation/speech.wav
```

These must be real local Colab paths. The repository explicitly requires local
model paths and requires the LTX-specific fine-tuned Gemma 4 paired with the
LTX-2.5 checkpoint, not a vanilla Gemma 4 or an LTX-2.3/Gemma-3 cache. See the
[official configuration reference](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/configuration-reference.md)
and [troubleshooting guidance](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/troubleshooting.md).

For a split LTX-2.5 pack, use the exact transformer, packed text-encoder,
video-VAE, and audio-VAE files belonging to the same release. Do not mix a
2.5 transformer with 2.3 VAEs or cached 2.3 text features.

### 2. Precompute fresh latents

The dataset currently has no precomputed tensors. Run the official
`process_dataset.py` against the 19-row manifest with:

```text
1280x704x153
```

The bucket is VAE-aligned: width/height are multiples of 32 and frames satisfy
`frames % 8 == 1`. The processor will center-crop 1280x720 to 1280x704; do not
manually pad the source. The official docs state that the output directory
must contain `latents/`, `conditions/`, and `audio_latents/`, and that changing
the checkpoint, Gemma model, trigger, or bucket requires a fresh output or
`--overwrite`.

Point `data.preprocessed_data_root` at the directory that directly contains
those three subdirectories, normally:

```text
training_clips_612/.precomputed
```

There is a subtle local-code trap: `PrecomputedDataset` accepts either the
dataset root or `.precomputed`, but `LtxTrainerConfig._validate_data_dirs_exist`
checks the configured root directly. In the training YAML, use `.precomputed`,
not its parent, unless you deliberately wrote the outputs directly there.

### 3. Add real validation inputs

The current validation sample is a placeholder. It must use an image and WAV
that are not in the training set. For the first meaningful run, prepare at
least three validation cases:

- an unseen identity image with ordinary speech;
- a second unseen identity or substantially different appearance;
- one held-out real talking-head clip's first frame and its matching audio,
  used only for evaluation, not training.

The validation audio must be the intended user audio and the validation image
must be a still that the model did not see as a training first frame. Use the
same 25 fps and `1280x704x153` validation shape. The `audio_to_video` condition
and `generate_video: true`, `generate_audio: false` are correct.

### 4. Prove that preprocessing and inference use the same contract

Before training, decode at least one precomputed video latent and one audio
latent using the same LTX-2.5 component files. Check:

- decoded video starts at the same first frame used for the first-frame
  condition;
- decoded audio duration is 6.12 seconds, not an accidental longer source;
- latent metadata has the expected latent frame count, spatial shape, FPS, and
  audio duration;
- all 19 samples have one matching file in each required directory;
- no conditions file came from an older LTX version.

The [official trainer guide](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/training-guide.md)
also recommends starting with a small run, saving checkpoints, and monitoring
validation samples rather than relying on training loss alone.

## Quality risks that remain after the blockers

### Dataset scale and identity leakage

The 19 clips provide approximately 116 seconds of material. This is enough to
test the pipeline and determine whether the native LTX-2.5 path can learn a
useful talking-head bias. It is not enough to claim robust universal
phoneme-to-mouth generalization or “perfect” sync.

Most speakers have only one clip. That is acceptable for a generic motion prior
smoke test, but the model sees speech, face, shot, and recording conditions
entangled at the clip level. More reliable training needs multiple clips per
speaker plus many speakers, with phonetic coverage (plosives, open vowels,
fricatives, rounded vowels, pauses, fast speech, and expressive speech).

Do not split adjacent clips from the same source across train and validation;
hold out complete identities or complete source videos.

### Six seconds is acceptable, but this set is overfit-prone

The community LTX-2.3 talking-head LoRA reports 25–30 clips of 6–10 seconds,
rank 32, learning rate 1e-4, batch size 1, and 2,000 steps; its reported final
dataset was 26 clips and the author reported identity locking around step 1250.
Those values are useful evidence that 6–10 seconds is a viable clip range, but
that release is character-specific and internalizes voice/identity. It is not
evidence that 19 heterogeneous clips can produce a universal lip-sync model.
See the [published LTX-2.3 model card](https://huggingface.co/elix3r/LTX-2.3-22b-AV-LoRA-talking-head).

With 19 clips and batch size 1, 1,000 steps is about 52.6 passes through the
dataset. That is a reasonable smoke-test budget but is overfit-prone. Save
every 100 steps and compare 300, 600, and 1,000-step outputs on held-out
identity/audio. Select by validation sync and identity, not the lowest train
loss. Do not jump straight to 2,000–3,000 steps just because the community
character LoRA did.

### Full-frame loss dilutes mouth supervision

The current target video loss includes background, hair, clothing, head motion,
and facial regions. The mouth is a small fraction of the pixels/tokens. A model
can reduce total MSE while still producing weak or inaccurate lip timing.

For this first native experiment, keep the code path simple and use the full
video loss to establish a baseline. If the output is not synchronized, the
next correct improvement is a deliberately designed mouth/face weighting or a
separate sync-aware objective—not a larger rank or more steps by default.

### Captions and trigger words

Do not add a trigger word unless you intentionally want a style adapter that is
activated by text. The current neutral captions are already repeated semantic
descriptions. A trigger token would not fix audio alignment and could become a
shortcut. If you later add one, prepend it consistently during preprocessing
using `--lora-trigger` and regenerate conditions; do not edit only the YAML.

### Watermarks, subtitles, microphones, and shot variation

The current flagged samples should not be the majority of a future dataset.
For the first run, keep them labeled in the audit and review their validation
outputs. For the next dataset version, prioritize:

- unobstructed lips without burned-in subtitles or watermarks;
- face occupying a consistent, large portion of the frame;
- mostly frontal views, with a controlled amount of 3/4 motion;
- no cuts, camera switches, or speaker entrances inside a clip;
- clean speech with natural pauses at the clip start;
- multiple speakers and multiple speech styles.

## Colab-specific assessment

Single-GPU execution is supported by the official training guide; use the
normal `scripts/train.py` invocation, not a multi-GPU Accelerate config. High
VRAM lets you keep bf16, no quantization, batch size 1, gradient checkpointing,
and the native 1280x704x153 bucket if it fits.

Do not infer that “high VRAM” means the largest bucket is automatically safe.
At 1280x704x153, the approximate transformer token count is:

```text
(704/32) * (1280/32) * (((153-1)/8) + 1) = 17,600 video tokens
```

That is substantially larger than the official 4,032-token example. First run
the preprocessing and one validation sample, record peak memory, then launch
training. If the 22B checkpoint cannot sustain that sequence on the allocated
Colab GPU, reduce the bucket consistently in both preprocessing and validation;
do not mix shapes or manually pad.

Use the repository's supported environment/dependency resolution. In
particular, keep Torch, TorchAudio, TorchCodec, CUDA, and any NATTEN/attention
backend compatible. A locally working Mac environment is not evidence that the
Linux CUDA Colab environment is correct.

## Final go/no-go table

| Area | Verdict | Action before expensive run |
|---|---|---|
| MP4 dimensions/FPS/frame count | Pass | None for smoke test |
| Embedded audio pairing | Pass with expected AAC padding | Let processor trim to video duration |
| Manifest schema | Pass | Keep `video` + `caption`; no separate WAV column |
| First-frame conditioning | Pass | Keep probability 1.0 and review PNGs |
| Neutral captions | Pass for this objective | Do not add transcripts as captions |
| Current strategy/loss | Supported but not sync-specialized | Treat as baseline, not perfection guarantee |
| Actual LoRA target set | Reasonable focused set | Confirm matched names on loaded 2.5 checkpoint |
| LoRA comments/docs | Corrected in fork | Keep the focused causal rationale |
| Model/VAE/Gemma paths | Blocked | Replace placeholders with matched 2.5 files |
| Precomputed tensors | Blocked | Fresh preprocess with `1280x704x153` |
| Validation | Blocked | Add held-out image/audio cases |
| Dataset diversity | Weak but usable for smoke test | Expand after baseline |
| Colab/high VRAM | Fine | Still measure sequence-memory peak |

Bottom line: the input preparation is good enough to test the hypothesis, but
the current repository/configuration is not yet “run it blindly” ready. The
right first experiment is a focused video+A2V LoRA with frozen audio, fresh
LTX-2.5 preprocessing, real held-out validation, 100-step checkpoints, and
selection by actual lip-sync/identity behavior. It should be treated as a
baseline experiment, not as a path that can guarantee perfect lipsync from 19
clips.

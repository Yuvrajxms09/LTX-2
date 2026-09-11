# LTX-2.5 native A2V lipsync LoRA: toolkit comparison and training decision

**Audit date:** 2026-09-11
**Scope:** LTX-2.5, image/first-frame + externally supplied speech audio -> generated talking-head video.
**Repositories inspected:** official LTX-2 trainer, the current modified fork, Ostris AI Toolkit, the Ostris-derived BIG DADDY fork, SimpleTuner, Musubi LTX-2 development fork, A4ax's ComfyUI LTX-2.5 trainer, `rs-nodes`, RelaxIS's RAM-offload fork, and a second ComfyUI LTX trainer.

## Executive decision

Keep the modified official LTX-2 trainer as the primary experiment. Do not switch the project to AI Toolkit, SimpleTuner, or the A4ax trainer for the first run.

Musubi is the only inspected external implementation that exposes the exact training semantics we need as a first-class option:

```text
AV mode + clean/frozen audio + video denoising/loss only + first-frame conditioning
```

Use Musubi as a reference implementation or a controlled second implementation if the official trainer cannot fit, preprocess, or sample correctly in Colab. Its `--ltx2_train_direction a2v` path is materially closer to the desired objective than the default behavior of the other alternatives.

The blunt status of the current project is:

- The source dataset is prepared for a smoke test: 19 paired MP4 clips, 1280x720, 25 FPS, 153 frames, approximately 6.12 seconds each, with embedded audio and a manifest.
- The training objective and custom configuration are conceptually correct for native A2V.
- The experiment is not end-to-end ready yet. The LTX-2.5 model assets, a clean Colab environment, and the precomputed video/audio/text caches still have to be produced and verified.
- Nineteen clips can validate the pipeline. They cannot justify a claim of robust arbitrary-image + arbitrary-speech lipsync or “perfect” quality.

## Objective contract

The desired inference contract is not voice cloning and not voice personalization:

```text
input image / first frame + exact target speech waveform
-> video is generated
-> supplied audio is preserved and drives mouth timing
```

During training, the correct directional contract is:

```text
video: noisy/generated, contributes video loss
audio: clean/frozen, contributes no audio loss
first frame: clean identity condition, paired with that same target video
```

This matters because “train with audio present” is not enough. A trainer that noises the audio stream and trains an audio loss is learning joint audio-video generation. A trainer that sets audio loss to zero but still feeds noisy audio is not equivalent to clean external-audio conditioning.

LTX-2 is a dual-stream audio/video transformer with bidirectional cross-modal attention. The A2V cross-attention path lets video queries read audio keys/values. The reverse V2A path also executes in the native block and can affect later video blocks indirectly through the evolving audio stream. Therefore, excluding trainable V2A adapters in the first causal experiment is a deliberate shortcut-control decision, not a claim that the frozen reverse path is mathematically unused. The LTX-2 paper describes the dual-stream/bidirectional design: [LTX-2 paper](https://arxiv.org/abs/2601.03233).

## Toolkit comparison

| Toolkit | Native LTX-2.5 | Clean frozen-audio A2V mode | First-frame/I2V support | Assessment for this project |
|---|---:|---:|---:|---|
| Official LTX-2 trainer, current fork | Yes | Yes, through `flexible` strategy | Yes | Primary choice; closest official code and simplest causal baseline |
| Musubi LTX fork | Yes | Yes, explicit `--ltx2_train_direction a2v` | Yes | Best external reference and strongest fallback |
| SimpleTuner | Yes | No exact directional mode found | Yes | Good native AV trainer, but not a drop-in replacement for clean external audio |
| Ostris AI Toolkit | Yes in inspected code | No dedicated clean-audio A2V mode found | Yes | Capable but less explicit for this objective; issues show audio, NaN, and performance risk |
| Ostris BIG DADDY fork | LTX-2.3-focused | No; joint/noisy audio path | Yes | Better joint voice-training diagnostics and fixes, but still the wrong objective and no verified native 2.5 path |
| A4ax ComfyUI LTX-2.5 trainer | Yes | No; trains video and audio jointly | Face+voice dataset path | Useful low-VRAM engineering reference, wrong objective for the first experiment |
| `rs-nodes` | Official LTX submodule | No dedicated directional mode | Via official trainer wrapper | Better ComfyUI UX/monitoring, not a better objective |
| RelaxIS LTX-2 fork | LTX-2 19B | Uses official flexible path, no 2.5 | Yes | Memory/offload fork for older LTX-2; not suitable as a 2.5 baseline |
| Jaimitoes ComfyUI trainer | Official LTX code | No dedicated directional mode | Toggles/standard strategies | UI wrapper around the same trainer semantics |
| LTX-Video-Trainer | LTX-Video, not LTX-2 | No | Older LTXV paths | Irrelevant to the native LTX-2.5 AV experiment |

### 1. Official LTX-2 trainer and current fork

The official trainer documents A2V as video generated from frozen audio. Its official `a2v_lora.yaml` uses the flexible strategy, `video.is_generated: true`, and `audio.is_generated: false`: [official A2V configuration](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/configs/a2v_lora.yaml), [official training modes](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/training-modes.md).

The current custom profile is a sensible specialization:

- rank 32 / alpha 32;
- video self-attention and video-text attention;
- explicit `audio_to_video_attn` projections;
- audio frozen and video generated;
- first-frame probability 1.0;
- batch size 1, BF16, gradient checkpointing;
- strict runtime logging and fail-fast non-finite checks.

The profile is at [a2v_lipsync_lora.yaml](/Users/yuvraj/Desktop/ltx/ltx2.5/LTX-2/packages/ltx-trainer/configs/a2v_lipsync_lora.yaml). The pristine official clone is at [ltx2_org_code](/Users/yuvraj/Desktop/ltx/ltx2.5/ltx2_org_code); the working fork is [LTX-2](/Users/yuvraj/Desktop/ltx/ltx2.5/LTX-2). Both were inspected at commit `a95ab856` for the upstream code.

The important difference from the official generic A2V example is target scope. The official example's short `to_k`, `to_q`, `to_v`, and `to_out.0` patterns attach to every matching attention branch, including audio self/text and reverse V2A attention. The custom profile explicitly targets the video branches plus A2V. That is the correct first ablation for isolating audio-to-mouth adaptation. A later broad-attention run can be evaluated, but it should not be the first run because it gives the LoRA more ways to solve the training set without proving that the external audio path is being learned.

The trainer now audits requested and actually applied LoRA modules and raises on an unmatched target. This is important because a typo in a target path can otherwise produce a successful-looking run with an incomplete adapter.

### 2. Musubi

The inspected LTX development fork has a dedicated directional feature:

```text
--ltx2_mode av
--ltx2_train_direction a2v
```

Its documentation states that A2V freezes audio at timestep/sigma zero, generates video, and excludes the frozen modality from loss. It also supports first-frame conditioning and LTX-2.5 split checkpoints/VAEs: [Musubi LTX-2 documentation](https://github.com/AkaneTendo25/musubi-tuner/blob/ltx-2-dev/docs/ltx_2.md), [Musubi LTX direction implementation](https://github.com/AkaneTendo25/musubi-tuner/blob/ltx-2-dev/src/musubi_tuner/ltx2_train_direction.py).

This is the cleanest external implementation of the intended causal experiment. It also has explicit valid-frame handling (`8*k+1`), separate audio buckets, cache invalidation warnings, and manual LoRA target presets. Its advanced features—IC-LoRA, DCR/TARP, cross-task synergy, CREPA, HFATO, and target estimation—should remain disabled for the first experiment. They change the experiment and make a failure harder to diagnose.

Community reports consistently describe Musubi as faster or lower-VRAM than AI Toolkit, but those reports are uncontrolled and hardware/version dependent. One Hivemind report claimed about 2.2 seconds/iteration for Musubi versus a multi-day AI Toolkit run on a 5090; another claimed roughly 2 seconds/iteration. These are useful operational signals, not quality evidence: [report 1](https://discord.com/channels/1076117621407223829/1457981700817817620/1539245877988958248), [report 2](https://discord.com/channels/1076117621407223829/1457981700817817620/1539392381235757139).

### 3. SimpleTuner

SimpleTuner has real LTX-2.5 support, Gemma 4 loading, video and audio VAE handling, first-frame conditioning, audio-aware datasets, and automatic audio-side LoRA targets: [LTX Video 2.5 documentation](https://docs.simpletuner.io/quickstart/LTXVIDEO2/).

The inspected implementation, however, samples a noisy audio timestep and computes joint video/audio training. Setting `audio_loss_weight` to zero only removes the audio loss; it does not make the audio conditioning clean at sigma zero. That is a fundamental mismatch with the first experiment, not a cosmetic configuration difference. SimpleTuner would become a candidate only after adding and validating an explicit directional A2V path equivalent to the official trainer or Musubi.

### 4. Ostris AI Toolkit

The inspected code has an LTX-2.5 model class, split transformer/Gemma 4/video-VAE/audio-VAE loading, `do_i2v`, `do_audio`, audio normalization, and native AV forward logic. It is a serious toolkit, not a fake integration. The repository is cloned at [ostris_ai_toolkit](/Users/yuvraj/Desktop/ltx/ltx2.5/ostris_ai_toolkit).

It was not selected because its configuration surface exposes I2V and audio loading, but no dedicated clean-audio A2V directional mode was found in the inspected code. Its default path is joint AV training. The project also has unresolved or historically reported LTX-specific operational risks:

- [Issue #975](https://github.com/ostris/ai-toolkit/issues/975) reports LTX-2.3 LoRAs with NaNs in `to_gate_logits.lora_B.weight` and black outputs.
- [Issue #701](https://github.com/ostris/ai-toolkit/issues/701) reports voice training not being learned and cache-latent crashes.
- [Issue #758](https://github.com/ostris/ai-toolkit/issues/758) reports approximately 5x slower LTX LoRA training than the community LTX trainer in one Windows/5090 comparison.

These issues do not prove that the current commit fails, but they make AI Toolkit a poor choice for the first correctness-critical experiment when the official trainer already expresses the needed objective.

### 5. A4ax ComfyUI LTX-2.5 trainer

The A4ax project is useful because it reports a real face+voice run and addresses low-VRAM execution with NF4, block sharding, spatial tiling, and checkpointing: [A4ax trainer](https://github.com/A4ax/comfyui-LTX-2.5-Tile-train-LoRa--On-multi-GPUs-low-VRAM-18-gb-Beta).

Its inspected training loop noises both video and audio and computes both losses. That makes it a joint face+voice adaptation trainer. It is not the correct first implementation for “use this arbitrary external speech waveform to drive only the mouth.” It is also solving a hardware constraint the project does not have because the planned run is on a high-VRAM single GPU. Avoid NF4/model sharding until a BF16 baseline is proven.

### 6. Ostris-derived BIG DADDY fork

The fork at [ai_toolkit_big_daddy_reference](/Users/yuvraj/Desktop/ltx/ltx2.5/ai_toolkit_big_daddy_reference) is more substantial than a superficial UI fork. Its inspected main branch contains independent audio timestep sampling, explicit audio latent cache validation, multi-fallback audio extraction, strict audio-supervision checks, dynamic audio-loss balancing, gradient-checkpointing wiring, and quantization/offload fixes. Those are meaningful improvements for joint character+voice LoRAs.

However, the fork's own training fix describes the target as joint LTX character voice training. Its code samples an audio sigma/timestep and builds an audio target/loss; it does not expose a clean-audio `a2v` directional mode equivalent to the official flexible strategy or Musubi. Its README and LTX-specific logic are also centered on LTX-2.3, and no verified LTX-2.5 training example was present in the inspected checkout. Therefore:

- use it as a source of ideas for audio extraction/cache assertions and logging;
- do not copy its dynamic audio-loss balancing into the native external-audio experiment;
- do not treat its “voice fix” as a lipsync fix—the fix is for learning a character's generated voice;
- do not select it over the official LTX-2.5 trainer for this project.

The community index identifies this fork as an audio/voice-training repair, which agrees with the code inspection but is not independent quality validation: [awesome-ltx2 toolkit index](https://github.com/wildminder/awesome-ltx2).

### 7. `rs-nodes` ComfyUI trainer

The [rs-nodes reference](/Users/yuvraj/Desktop/ltx/ltx2.5/rs_nodes_reference) provides a polished ComfyUI workflow around the official LTX trainer: dataset preparation, Ollama captioning, module toggles, a training monitor, divergence detection, and in-process reuse of the loaded 22B transformer. It is useful operationally.

The training node builds the official `TextToVideoStrategy` or `VideoToVideoStrategy`. Its `with_audio` switch enables the standard audio path, not a clean/frozen-audio A2V direction. It also defaults to user-selectable quantization and has UI defaults that are not the same as the controlled BF16 experiment. It is therefore a front end and convenience layer, not a better causal training implementation.

### 8. RelaxIS RAM-offload fork

The [RelaxIS reference](/Users/yuvraj/Desktop/ltx/ltx2.5/relaxis_ltx2_reference) is a practical fork of the older official trainer with `ramtorch` CPU offloading for LTX-2 19B on 24–32 GB cards. Its own guide disables validation under RAM offload because the validation sampler does not support that execution path. The checkout is pinned to an older LTX-2-era commit and has no verified LTX-2.5 training path. Since the planned machine has high VRAM, this adds an old execution path and removes useful validation; it is not an improvement for this experiment.

### 9. Jaimitoes ComfyUI LTX2 trainer

The [ComfyUI LTX2 trainer reference](/Users/yuvraj/Desktop/ltx/ltx2.5/comfyui_ltx2_trainer_reference) exposes useful module toggles, including A2V and V2A attention groups, and wraps the official trainer inside ComfyUI. Its actual configuration still selects the standard text-to-video/video-to-video strategies and uses `with_audio` for AV. There is no first-class clean-audio directional switch. It is another UI wrapper, not a different answer to the objective.

## Training-data assessment

The current source folder is documented in [DATASET_README.md](/Users/yuvraj/Downloads/ltx_lipsync_sources/native_720p_all/training_clips_612/DATASET_README.md). The current state is:

- 19 clips and 19 manifest rows;
- all source videos are 1280x720, 25 FPS, 153 frames, about 6.12 seconds;
- the manifest contains `video` and neutral visual `caption` only;
- audio is embedded in the same MP4, so preprocessing can extract the paired waveform from the same source;
- first-frame review passed the hard checks for all 19 clips, but several are conditional because of watermarks, subtitles, lens flare, a microphone, or wide framing: [first-frame QA report](/Users/yuvraj/Downloads/ltx_lipsync_sources/native_720p_all/training_clips_612/first_frame_qa_report.md);
- no `latents/`, `audio_latents/`, or `conditions/` cache is present yet.

### What is correct

Using real videos of real people speaking, with their own audio, is the correct primary supervision. Each sample contains the exact relationship the model must learn: visual mouth motion aligned with speech acoustics. The first frame should be taken from the same target video, not generated separately and not supplied as an unrelated image column.

Keeping the audio embedded in the paired MP4 is also correct for this dataset. It prevents accidental video/audio pairing errors. If audio is split into separate files later, the split must preserve the exact start time and duration.

The current 153-frame length is valid for LTX temporal alignment (`8*k+1`) and is a reasonable smoke-test bucket. Clips do not have to be the same duration in principle; the trainer can bucket them. With 19 samples and batch size 1, one fixed bucket is simpler and makes alignment failures obvious. Do not add synthetic silence to force durations.

The 6.12-second clips are not inherently too short. They are long enough to contain multiple phonemes, co-articulation, blinks, and expression transitions. They are too few in aggregate for robust generalization. A 6-second clip is acceptable for the first pipeline test; repeating the same tiny set until the loss looks good would only demonstrate memorization.

### What is not yet sufficient

The dataset is not sufficient for a credible “99/100” arbitrary audio/image result. The main deficiencies are:

1. **Too few independent speakers and clips.** Most identities appear once. The model cannot separate identity, camera, voice, background, and speech content reliably from one example each.
2. **Weak held-out evaluation.** A real test requires speech not present in training and images/identities not used for that training pair. The current validation placeholders are not a valid evaluation set until replaced.
3. **Conditional overlays.** Burned-in subtitles and watermarks add stable visual patterns that the LoRA can memorize. Remove or replace those clips for the quality run.
4. **Limited phonetic coverage.** The current clips are useful interview speech, but the manifest has no phoneme/word coverage audit. Add varied speech rates, plosives, fricatives, rounded vowels, pauses, and sentence endings across speakers.
5. **Legal provenance.** Celebrity/interview sources may be fine for private research, but their licensing and consent are not automatically suitable for a distributable training adapter. Preserve source URLs, license status, speaker identity, and consent/provenance metadata.

Generated talking-head clips from another model are not a better primary target. They teach that model's artifacts, mouth biases, and audio-generation behavior. They can be a deliberately labeled secondary augmentation experiment, not a replacement for real paired speech video.

## Recommended first experiment

### Phase 0: make the run reproducible

1. Use a fresh Colab environment with the exact LTX-2.5 package dependencies. The local macOS environment cannot validate the trainer import because its installed Pydantic is older than the package requirement; this is an environment issue, not evidence that the trainer is broken.
2. Download matching LTX-2.5 transformer, Gemma 4 text encoder, video VAE, and audio VAE assets. Do not mix LTX-2.3, LTX-2.5, vanilla Gemma 4, or stale caches.
3. Preprocess the 19 MP4s from the manifest into a new `1280x704x153` output directory. Let the official dataset processor extract and trim audio to the processed video duration.
4. Decode a few cached video/audio latents and verify shape, dtype, frame count, FPS, and duration before training. Do not start a 22B run before this check.
5. Replace every placeholder path in [a2v_lipsync_lora.yaml](/Users/yuvraj/Desktop/ltx/ltx2.5/LTX-2/packages/ltx-trainer/configs/a2v_lipsync_lora.yaml), especially model assets, preprocessed data, validation image, and validation audio.

### Phase 1: causal baseline

Run the current official-fork profile with:

```text
video generated = true
audio generated = false
first-frame probability = 1.0
rank/alpha = 32/32
BF16, no quantization
batch size = 1
gradient checkpointing = true
learning rate = 1e-4
1000 steps, checkpoint every 100
```

Keep the explicit current target list for this run. Do not add audio self-attention, audio-text attention, reverse V2A adapters, FFN adapters, quantization, or advanced regularizers simultaneously. The run must answer one question: did the LoRA improve video mouth timing when given clean external speech?

Train on the BF16 development/base checkpoint used by the intended inference path. Do not train against a distilled checkpoint and evaluate against a different development checkpoint without a deliberate compatibility plan.

### Phase 2: controlled ablations

Only after the baseline is sampled and inspected:

1. Compare current explicit video+A2V targets against the official broad attention target list.
2. If the explicit run follows audio but identity/motion is weak, add video FFN adapters as a separate run.
3. If the explicit run fails to respond to audio, inspect audio latent alignment and A2V target application before adding more modules. Then use Musubi's explicit A2V direction as an implementation cross-check.
4. Do not interpret lower training loss alone as improved lipsync.

### Evaluation that can falsify the result

Use a held-out set with:

- an identity image not used as the paired training frame;
- speech content and speaker/voice not used in the training clip;
- exact audio duration matching the requested video duration;
- the same frame rate and bucket contract.

Record the supplied audio, generated video, checkpoint, seed, prompt, dimensions, and adapter scale for every sample. Evaluate word intelligibility separately from mouth timing. Sync metrics can be useful, but they are not sufficient: a model can have a good sync distance while producing wrong words or poor identity. Human review of mouth closures, plosives, vowels, consonant transitions, blinks, and frame-to-audio delay remains mandatory.

## Alignment warning from the community

An LTX-2.5 community report found that custom audio longer than the generated clip caused lipsync to disappear, while exact-duration audio restored it. This is anecdotal, but it agrees with the trainer's need to align the audio latent duration to the video bucket. Treat exact duration/frame alignment as a hard input invariant: [Hivemind report](https://discord.com/channels/1076117621407223829/1309520535012638740/1539564978988912680).

Separate Reddit reports describe the same practical split between the toolkits: AI Toolkit runs that preserved visual character quality but failed to transfer voice, followed by Musubi runs that transferred voice and visuals more reliably. Other users explicitly describe fighting the AI Toolkit audio path and finding Musubi effective. These reports are not controlled comparisons, and they mostly concern character/voice LoRAs rather than clean external-audio A2V, but they are consistent with the code-level finding that Musubi exposes the more explicit AV training controls: [voice-transfer report](https://www.reddit.com/r/StableDiffusion/comments/1qy75cd/ltx2_lora_training/), [tooling comparison](https://www.reddit.com/r/StableDiffusion/comments/1rjn36f/is_there_someone_out_there_making_ltx2_finetunes/).

## Final recommendation

The current modified official trainer is the right primary codebase. The dataset is a valid smoke-test source set, not a finished quality dataset. The one external toolkit worth keeping in the workspace is Musubi, specifically as a clean A2V implementation cross-check. SimpleTuner and AI Toolkit are capable native AV trainers but do not currently give a cleaner fit for this exact objective. A4ax solves low-VRAM joint face+voice training and should not replace the causal baseline.

Do not train yet merely because the MP4s and YAML exist. First produce and verify the LTX-2.5 latent caches, replace the placeholder paths, and run a one-batch contract check. After that, a 1000-step BF16 baseline is a technically defensible first experiment; the quality conclusion must wait for held-out image/audio evaluation.

## Evidence and source quality

Highest confidence: official LTX-2 code/docs and direct inspection of the cloned implementations.
Medium confidence: SimpleTuner/Musubi/AI Toolkit repository code and documentation.
Lower confidence: Hugging Face model cards, Reddit, and Hivemind/Discord reports, which are self-reported or anecdotal.

Additional references:

- [Official LTX-2 repository](https://github.com/Lightricks/LTX-2)
- [Official LTX-2 quick start](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/quick-start.md)
- [SimpleTuner LTX-2.5 docs](https://docs.simpletuner.io/quickstart/LTXVIDEO2/)
- [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit/)
- [Ostris BIG DADDY fork](https://github.com/ArtDesignAwesome/ai-toolkit_BIG-DADDY-VERSION)
- [`rs-nodes` ComfyUI toolkit](https://github.com/richservo/rs-nodes)
- [RelaxIS LTX-2 RAM-offload fork](https://github.com/relaxis/LTX-2)
- [Jaimitoes ComfyUI LTX2 trainer](https://github.com/jaimitoes/ComfyUI-LTX2-TRAINER)
- [A4ax low-VRAM LTX-2.5 trainer](https://github.com/A4ax/comfyui-LTX-2.5-Tile-train-LoRa--On-multi-GPUs-low-VRAM-18-gb-Beta)
- [Official LTX-Video trainer, older LTX generation](https://github.com/Lightricks/LTX-Video-Trainer)
- [LTX-2.3 talking-head LoRA model card](https://huggingface.co/elix3r/LTX-2.3-22b-AV-LoRA-talking-head)
- [LTX-2.3 ID-LoRA TalkVid model card](https://huggingface.co/AviadDahan/LTX-2.3-ID-LoRA-TalkVid-3K)
- [ID-LoRA paper](https://arxiv.org/abs/2603.10256)
- [Community LipDub discussion](https://www.reddit.com/r/StableDiffusion/comments/1ta66f1/lipdub_beta_new_opensource_lipsync_iclora/)
- [Community report on inconsistent LTX-2.3 lipsync](https://www.reddit.com/r/StableDiffusion/comments/1u4ozdn/fixing_inconsistent_lip_sync_in_ltx_23/)

The Hivemind `unified_feed` view was unavailable during this audit (`relation public.unified_feed does not exist`), so the community evidence above was retrieved from the read-only raw `message_feed` endpoint after the required unified-view attempt.

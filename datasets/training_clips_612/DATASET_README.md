# Native LTX-2.5 A2V lip-sync dataset

This folder is the source dataset for the first native LTX-2.5 image + speech → video LoRA experiment.

## Dataset contract

- 19 MP4 clips
- 1280×720 source video
- 25 FPS
- 153 decoded frames per clip
- 6.12 seconds of video per clip
- 48 kHz, stereo, normalized embedded audio
- one visible talking speaker per clip
- `dataset_manifest.jsonl` contains only `video` and neutral visual `caption` fields

Use the video column only. `process_dataset.py` extracts the audio from the same MP4, which preserves the video/audio pairing. Do not add a separately extracted audio column unless its timing is independently verified.

## First-frame condition

The first frame is the visual identity condition. The trainer configuration should use:

```yaml
video:
  conditions:
    - type: first_frame
      probability: 1.0
```

The PNGs under `first_frames/` are review/reference exports of decoded frame 0. They are not a second training modality and should not be added as an `image` column for this experiment.

## Preprocessing

Use the LTX-2.5 transformer, matching Gemma text encoder, video VAE, and audio VAE. Preprocess into a fresh output directory with the native bucket:

```text
1280x704x153
```

The source is 1280×720, but the VAE-aligned training bucket is 1280×704. Let the trainer perform its normal crop/resize; do not pad the source videos manually.

Keep the precomputed output separate from the source clips and regenerate it when changing the checkpoint, text encoder, VAE, or bucket.

## QA references

- `FIRST_FRAME_QA.md` — acceptance checklist
- `first_frame_qa_report.md` — review of all 19 clips
- `first_frame_qa/contact_sheets/` — frame 0/0.20/0.40/1.00 s visual sheets
- `audio_normalization_report.tsv` — source loudness and final audio stream audit

# Native LTX-2.5 A2V lip-sync dataset

This folder is the source dataset for the first native LTX-2.5 image + speech → video LoRA experiment.

## Dataset contract

- 35 MP4 clips: the original 19 plus 16 newly added candidates
- 1280×720 source video
- 25 FPS
- 153 decoded frames per clip
- 6.12 seconds of video per clip
- 48 kHz, stereo, normalized embedded audio
- one visible talking speaker per clip
- `dataset_manifest.jsonl` is the authoritative all-35 manifest and contains only
  `video` and neutral visual `caption` fields

The split manifests are retained for auditability:

- `dataset_manifest_existing_19.jsonl` — the original notebook dataset
- `dataset_manifest_new_16.jsonl` — the newly added candidate clips
- `dataset_manifest_all_35.jsonl` — identical in content to the authoritative manifest

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

## Dataset selection and QA

- `FIRST_FRAME_QA.md` — acceptance checklist
- `first_frame_qa_report.md` — review of the original 19 clips
- `first_frame_qa_report_new_16.md` — review of the newly added 16 clips
- `first_frame_qa/contact_sheets/` — frame 0/0.20/0.40/1.00 s visual sheets
- `audio_normalization_report.tsv` — source loudness and final audio stream audit for
  the original 19 clips

The new clips are included in the training manifest as requested, but several are
marked conditional in the new-16 QA report. The notebook's first-frame QA cell is
the final gate before preprocessing; reject a clip there if the speaker, mouth, or
speech onset is not clean enough for the experiment.

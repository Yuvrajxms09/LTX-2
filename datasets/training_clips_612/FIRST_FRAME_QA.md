# First-frame QA for native LTX-2.5 A2V lip-sync

The trainer uses the first frame of each training video as the `first_frame` visual condition. Review frame 0 of every clip before preprocessing; do not judge only the middle of the clip.

## Accept

- Exactly one intended speaker is visible.
- The face is sharp enough to identify the eyes, lips, and jaw.
- The camera is already on the intended framing; there is no transition, cut, fade, or motion blur.
- The mouth is in a natural starting pose. A brief neutral lead-in before speech is preferable.
- No hand, microphone, subtitle, logo, watermark, or foreground object covers the mouth.
- Lighting and exposure are stable for the opening frames.
- The subject is looking at or naturally toward the camera, consistent with the target talking-head use case.

## Reject or re-trim

- Frame 0 is a cut, freeze frame, blurred transition, or a different speaker.
- The subject enters the frame after frame 0.
- The first frame catches an extreme mouth shape from a word that began before the clip.
- The crop cuts off the lips or chin, or the face is too small for reliable mouth motion.
- A subtitle or watermark overlaps the lower face.
- The speaker changes, the camera cuts, or the framing changes during the clip.

## Recommended review procedure

1. Extract frame 0 and frames at 0.20 s, 0.40 s, and 1.00 s.
2. Confirm the same speaker and stable framing across those frames.
3. Listen from the beginning and mark the first complete spoken word.
4. If speech begins immediately with a clipped phoneme, re-trim slightly earlier when possible; otherwise reject the clip.
5. Keep the same first-frame rule for validation and inference images: clear identity, stable lighting, no overlays, and a natural mouth pose.

The six-second duration is acceptable. A clean first frame and correct audio/video alignment matter more than adding padding or making the clips longer.

# First-frame QA report

Reviewed the 19 clips in this directory at frame 0, 0.20 s, 0.40 s, and 1.00 s. Audio onset was screened against the existing transcript selections and a `silencedetect` pass at `-35 dB`; silence detection is only an onset screen, not a phoneme-level lip-sync metric.

## Result

**19/19 pass the hard first-frame acceptance check:** the intended speaker remains visible, the shot is continuous at all four checkpoints, and the mouth is not obstructed. No video was changed during this review.

The `quality flag` column identifies clips that are valid for a smoke test but should not be treated as clean reference exemplars.

| Clip | First-frame result | Quality flag | Decision |
|---|---|---|---|
| `athena_isabella__y9aqMt5mOhI__candidate.mp4` | Clear face; stable car shot; short natural lead-in | none | preferred |
| `charlie_white__LagqBYdGX6I__candidate.mp4` | Clear face; stable single shot | speech begins immediately; no clipped word observed | usable |
| `cristiano_ronaldo__IGsE3eXby1g__candidate.mp4` | Clear face; stable interview shot | sponsor-board background | usable |
| `cristiano_ronaldo__m1xLo6hbYxU__candidate.mp4` | Face and mouth clear; shot stable within selected range | strong lens flare and very dark background | conditional |
| `david_goggins__-nMxwlqdl7I__candidate.mp4` | Face and mouth clear; stable shot | three-quarter profile and microphone close to the mouth | conditional |
| `emily_blunt__IZuyBeCeybk__candidate.mp4` | Clear face; stable shot; hand does not cover face | none | preferred |
| `emma_mmmenglish__-P-5RC17BHw__candidate.mp4` | Clear frontal face; stable shot | persistent lower-left watermark | conditional |
| `juli_ram__wTQ1zxnriXo__candidate.mp4` | Clear face; stable shot | speech begins immediately; audio was very quiet before normalization | usable |
| `nicole_van_groningen__4e5_aL07MCc__candidate.mp4` | Clear face; stable shot; mouth unobstructed | burned-in subtitles appear after the opening; cup remains in hand | conditional |
| `sydney_sweeney__sU75YyIQF2s__candidate.mp4` | Clear face; stable frontal shot | face is smaller than a close-up; laptop is visible | usable |
| `tara_marino__mNrJO88W_U8__candidate.mp4` | Clear face; stable outdoor shot | outdoor background and speech begins immediately | usable |
| `tom_cruise__bm1gvNQtYJQ__answer_that_question__6p12s__153f__1280x720.mp4` | Clear close-up; stable shot; no clipped word | begins about 40 ms before “answer,” effectively at speech onset; lower-left network logo | conditional |
| `tom_hiddleston__c6YN5tDAVW4__candidate.mp4` | Clear close-up; stable shot | gaze is somewhat off-camera | usable |
| `tom_holland_host__1z4Haf0fjDc__candidate.mp4` | Correct host remains visible; stable shot | wider framing makes the mouth relatively small | conditional |
| `unknown_man_british_education__M7RWz0xOJAg__candidate.mp4` | Clear frontal face; stable shot; short lead-in | microphone is near the left side of the frame, not over the mouth | usable |
| `unknown_woman_mental_health_speak__KTRwwcd-4BA__candidate.mp4` | Clear frontal face; stable shot; useful lead-in | none | preferred |
| `veronika__f96t2rsb4Lw__candidate.mp4` | Clear face; stable shot | burned-in subtitle is present from the opening | conditional |
| `will_creator__V80RXZ2N9o0__candidate.mp4` | Clear face; stable shot; useful lead-in | none | preferred |
| `yulisa__31Urh0XD9Mo__candidate.mp4` | Clear expressive face; stable shot | upward camera angle and hand approaches the upper frame later, not the mouth | usable |

## Training decision

Keep the 19-file manifest unchanged for the first smoke test: there are no hard failures, and the conditional examples add useful pose, lighting, and interview variation. Do not interpret the conditional clips as clean quality references.

For the next higher-quality run, replace or exclude the conditional clips with burned-in text/watermarks first (`emma_mmmenglish`, `nicole_van_groningen`, `veronika`, and Tom Cruise). Treat the Ronaldo lens-flare clip and the wide Tom Holland host clip as lower-priority replacements. Keep Goggins only if three-quarter talking-head motion is intentionally part of the target distribution.

## First-frame operating rule

Do not extract a separate arbitrary identity image for training. The native `first_frame` condition should come from the same target video so the image and target motion are paired. For each future clip, trim to a stable frame before speech, preferably with roughly 0.1–0.4 s of natural lead-in. Do not add synthetic silence or pad the video. At inference, the supplied identity image should meet the same visual standard: sharp eyes/lips/jaw, no overlays, no transition, stable exposure, and a natural mouth pose.

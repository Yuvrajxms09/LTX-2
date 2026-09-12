# First-frame QA report — new 16 clips

Reviewed the 16 newly added clips at frame 0 and around 0.20 s, 0.40 s, and
1.00 s. All have a continuous opening shot with the intended speaker visible;
there are no black opening frames or cuts in the inspected checkpoints. This is
an engineering QA report, not a phoneme-level lip-sync score.

| Clip | Frame result | Audio/word-boundary result | Decision |
|---|---|---|---|
| `camille__5TAV-tAf7To__candidate__6p12s__153f__1280x720.mp4` | Face visible, but sunglasses cover the eyes | Starts on “Cody”; long pause later | conditional |
| `camille__B3MRkuiUrx0__candidate__6p12s__153f__1280x720.mp4` | Face visible; hand gesture approaches the lower face | Starts on “Okay” at a clean phrase boundary | conditional |
| `matthew_moore__CwDX2vqrSpY__candidate__6p12s__153f__1280x720.mp4` | Stable outdoor close-up; mouth clear | Starts on “Look” at a clean phrase boundary | core |
| `amanda_pritchett__NH4ImrQqkjg__candidate__6p12s__153f__1280x720.mp4` | Stable indoor talking head; mouth clear | Starts on “Hi” | core |
| `maylene_espinoza__YBeKj4qYpcM__candidate__6p12s__153f__1280x720.mp4` | Stable indoor medium close-up | Starts with the continuation “from Aransas Pass High School” | conditional |
| `unknown_woman_success_subtext__YhmQwDAQP68__candidate__6p12s__153f__1280x720.mp4` | Stable talking head; face and mouth clear | Starts on “There are” | core |
| `joanna_calinski__bGEZWdGieC4__candidate__6p12s__153f__1280x720.mp4` | Stable indoor talking head; mouth clear | Starts on “Have you ever” | core |
| `evie__cY_Eyw1br8I__candidate__6p12s__153f__1280x720.mp4` | Face clear; handheld microphone and outdoor background | Starts on “My name is Evie” | conditional |
| `kaitlyn__fZtxwOrJtaE__candidate__6p12s__153f__1280x720.mp4` | Face clear; hand movement stays away from mouth | Starts with the continuation “my AP Lit class” | conditional |
| `savannah_rodriguez__l7azc-byXYE__candidate__6p12s__153f__1280x720.mp4` | Stable indoor close-up; mouth clear | Starts on “Um, my name is” | core |
| `ryan_robinson__nxgsqh1WL6s__candidate__6p12s__153f__1280x720.mp4` | Face clear; outdoor exposure/background variation | Starts on “Every second of our lives” | conditional |
| `danielle_reynolds__oXqXG1DYXQU__candidate__6p12s__153f__1280x720.mp4` | Stable indoor close-up; mouth clear | Starts on “I hope to go” | core |
| `polina_kravchenko__rbPTrA6aehI__candidate__6p12s__153f__1280x720.mp4` | Face clear; stable indoor shot | Starts on “Welcome back” | conditional |
| `ryan_robinson__rmTNB-OgQWM__candidate__6p12s__153f__1280x720.mp4` | Usable face, but low-light vehicle shot | Starts on “If you have to spend” | conditional |
| `tyler_owens__sU-8UUoWHXU__candidate__6p12s__153f__1280x720.mp4` | Face clear; outdoor/forest background | Starts on “We live in a world” | conditional |
| `citlalli_garcia_rodriguez__yAQkV_agoOE__candidate__6p12s__153f__1280x720.mp4` | Stable indoor close-up; mouth clear | Starts on “Um, some hobbies” | core |

The two explicit phrase-boundary exceptions are Maylene and Kaitlyn. They are
included in the all-35 manifest because the user requested the new clips be
combined with the existing set, but they should be replaced or trimmed at a
later quality pass if clean sentence starts are required.

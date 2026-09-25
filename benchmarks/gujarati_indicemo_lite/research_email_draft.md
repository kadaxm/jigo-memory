# DRAFT — email to research@rumik.ai (DO NOT SEND until you approve)

**To:** research@rumik.ai
**Subject:** Gujarati IndicEmo extension + speaker-drift quantification — findings from building on rumik-oss 1

---

Hi Rumik team,

Congratulations on rumik-oss 1 — the IndicEmo/NoVA/WER-CER release and the
"fewer than 70,000 hours" training story are exactly the kind of open work the
Indic TTS space needed.

We're reaching out because we've been building on it — and measuring it. Three
findings, then the application.

## 1. Gujarati IndicEmo-lite extension (preliminary)

Gujarati is in your claimed 22 languages but absent from the evaluated 7. We
extended the IndicEmo methodology to 24 code-switched Gujarati+English prompts
(5 deliveries, Gujarati anchor accent, your description format, Ira voice).

**Results (preliminary — substituted free-tier judge panel, 2 judges, 24 prompts;
not comparable in absolute terms to your published 3-judge scores):**

| delivery | score /5 |
|---|---|
| excited | 4.94 |
| happy | 4.75 |
| professional | 4.31 |
| sad | 3.31 |
| **angry** | **2.80** |
| **overall** | **4.00** |
| emotions-only | 3.95 |

Directionally: Gujarati delivery adherence looks *strong* — excited and happy
near-ceiling. Angry is the weak spot in our data (interestingly, the inverse of
your published pattern where angry was rumik-oss 1's best category). We'd love
to see whether your paid judge panel confirms this.

## 2. Speaker-drift: quantified (your admitted limitation)

Your post flags that timbre/speaker identity are "free to move" under the RL
rewards. We measured it: same Gujarati sentence, Ira voice, 6 conditions
(neutral + happy/sad/angry/excited/professional), speaker similarity via timbre
vectors (MFCC mean+std, cosine — approximate, disclosed).

**Result: mean similarity vs the neutral anchor = 0.996 (min 0.992, max 0.998).**

Good news: at least on short utterances, your reward routing (semantic →
codebook 0, acoustic → 1-7) appears to preserve speaker identity well in
practice. The limitation you disclosed is real on paper but small in
measurement — at least at our scale.

## 3. Speaker identity is pinned against descriptions

We probed whether `<description>` conditioning can cross speaker identity
(e.g., "a deep male voice, middle-aged Gujarati man") — it cannot. All four
trained voices are female-presenting, and male descriptions leave the voice
unchanged. Combined with finding 2, the picture is: **speaker identity is
pinned against descriptions but stable across emotions** — which your tech
report's speaker-consistency roadmap may find useful.

## 4. On-device feasibility

rumik-oss 1 runs on a 4GB consumer laptop GPU (RTX 3050) in 4-bit (bitsandbytes):
3.35GB VRAM peak, ~0.09-0.11× realtime (offline only), quality indistinguishable
from the official bf16 samples to our ears. We also ran XTTS-v2 on the same
Gujarati text as an exploratory comparison — XTTS does not support Gujarati
script; rumik-oss does natively. (XTTS exploratory attempts included for
completeness, labeled as unsupported-language attempts.)

## The application

All of this lives inside **Jigo** — a voice-native memory assistant we built
solo (open-source: github.com/kadaxm/jigo-memory): speak → it stores with
salience scoring, decays by memory type, resolves conflicts; ask → it answers,
grounded only in stored memories, with a refusal gate (adversarially evaluated:
100% top-1/top-3 retrieval, P50 67ms, 2/2 refusals). We're wiring rumik-oss 1
in as the Indic-expressive voice tier — including a mood-aware mode where
detected user sadness flips the reply into a warm Gujarati-optimist register.

Since you invited builders to reach out: we'd love your input on two things —

1. Would your team consider running our Gujarati prompts through your paid
   3-judge panel, so the preliminary numbers get your calibration?
2. Any guidance on the minimum data/compute for speaker-adaptation SFT on
   rumik-oss 1-base (we're attempting a regional persona voice)?

All prompts, methodology, WAVs, judge responses, and scoring code are in our
repo under `benchmarks/gujarati_indicemo_lite/`.

Kadam Vyas
[phone] · [email] · github.com/kadaxm

---
*DRAFT — awaiting Kadam's review before sending. Numbers: see
benchmarks/gujarati_indicemo_lite/judgments/findings.json + drift_findings.json*

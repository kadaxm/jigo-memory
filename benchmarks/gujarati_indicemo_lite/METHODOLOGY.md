# Gujarati IndicEmo-Lite — Methodology

A preliminary extension of [IndicEmo](https://github.com/ira-rumik/IndicEmo)
(Rumik AI, Apache-2.0) to **Gujarati** — a language rumik-oss 1 claims to support
(22 languages) but which is absent from the published IndicEmo evaluation set
(7 languages: EN, HI, TE, TA, KN, BN, PA).

**Status: PRELIMINARY. Substituted judge panel. Not comparable to published scores.**

## Scope

- 24 code-switched prompts (Gujarati native script + English), culturally grounded
- 5 deliveries, matching IndicEmo categories: happy 5, sad 5, angry 5, excited 5, professional 4
- Anchor accent: Gujarati (free-text description — the model's accent list documents 7 accents and does not include Gujarati; testing whether the description format generalizes)
- Paces: slow/steady/fast mixed, per IndicEmo conventions

## Systems under comparison (pairwise, self-adherence design)

1. **rumik-oss 1 (delivery-conditioned)** — Ira voice, `<description="{delivery}, Gujarati accent, {pace} pace">`
2. **rumik-oss 1 (neutral baseline)** — Ira voice, `<description="neutral, steady pace">`, same text

Rationale: rumik-oss 1 ships 4 fixed female voices with no zero-shot cloning, and
XTTS-v2 does not support Gujarati script — so a cross-system comparison would
compare an unsupported configuration. The delivery-vs-neutral pair measures
**delivery adherence on Gujarati** using the model's own controls.

Third exploratory arm (optional, labeled): XTTS-v2 clone on romanized Gujarati —
unsupported language, recorded as an exploratory attempt only.

## Judge protocol (substituted panel — disclosed limitation)

IndicEmo's published protocol uses three paid audio-capable judges (Gemini 3.1
Pro, Qwen 3.5 Omni, GPT Realtime). This extension substitutes **one free-tier Gemini
model** (`gemini-3.5-flash-lite`, called through two API keys — so the two "judges" are
the same model with sampling noise, mean |difference| 0.30), which introduces:

- a panel of one model instead of three
- same-family / single-judge bias (their own METHODOLOGY documents this: "judge
  dependence and possible same-family bias")
- leniency: the judge is told which delivery to look for, and rates neutral speech
  4.7–4.8/5 when asked for "happy" or "excited" — so a delivery score alone is
  uninformative; only the **paired** difference against the neutral twin is reported

Judges receive the same anchored 1–5 Expression Quality rubric axes
(tone, dynamics, phrasing, sustained expression). Median of available judges per
recording; delivery-category means with equal category weights.

## Measurements

1. **Delivery adherence**: judge score of delivery-conditioned clip vs neutral
   baseline clip, per category
2. **Pronunciation/intelligibility**: spot-listen + (optional) ASR spot-check
3. **Speaker drift (separate audit)**: same neutral sentence spoken across 5
   emotion conditions; speaker-embedding similarity matrix (resemblyzer ECAPA,
   fallback: librosa MFCC cosine — labeled approximate)

## Limitations of this extension

- 24 prompts vs IndicEmo's 100 — preliminary, not a headline benchmark
- Gujarati prompts drafted by a non-native model, reviewed by a native speaker
  (author) before judging
- Free-tier judge models, not the published paid panel
- rumik-oss 1 runs 4-bit quantized (bitsandbytes) on a 4GB RTX 3050 at ~0.1×
  realtime — generation is offline; quality impact of quantization is unmeasured
  upstream (user listening gate passed vs official bf16 samples)

## Results (corrected)

An earlier version of this file's companion draft reported an overall 4.00/5 and a
0.996 speaker similarity. Neither survives a control:

| Measurement | Result | Script |
|---|---|---|
| Delivery − neutral twin, paired (judge score, 1–5) | happy +0.05, excited +0.15, **sad +1.00**, angry +0.15, professional 0.00; all 24: **+0.28** (95% bootstrap CI −0.07…+0.68). Only sad's interval excludes 0 (+0.20…+1.80). | `judge_neutral_control.py`, `analyze_judgments.py` |
| Intelligibility, CER vs prompt text (Gemini ASR, English words kept in Latin script) | delivery 0.516, neutral 0.528 | `intelligibility_cer.py` |
| Speaker identity, resemblyzer 256-d cosine | emotion vs neutral twin 0.904 ± 0.036 (n=24); same voice, different sentences, both neutral 0.899 ± 0.024 (n=276); Ira vs other voices 0.49–0.62 | `drift_speaker_embedding.py` |

Reading: emotion conditioning leaves speaker identity and intelligibility intact, but
in Gujarati it changes delivery measurably only for **sad**; happy/excited are
indistinguishable from neutral to this judge, and the requested **angry** did not move.
`excited` has the lowest speaker similarity (0.882, min 0.789 — below the neutral-pair
minimum 0.805) but n=5 cannot support a claim.

Limits that still apply: one judge model, 4–5 clips per category, other-voice baselines
use a human male voice and XTTS clones (not rumik-oss's other three voices), and
Gujarati prompts were drafted by a model and reviewed by one native speaker.

Prosody check (same clips): `emotion.py` labelled 21 of 24 delivery-conditioned clips
"neutral" and 3/24 differently from their neutral twin — voice prosody is not a usable
emotion signal for this model's output.

## Reproduction

All prompts, generation configs, judge raw responses and scoring scripts are committed in
this repository under `benchmarks/gujarati_indicemo_lite/` (WAV outputs are gitignored).

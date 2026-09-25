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
Pro, Qwen 3.5 Omni, GPT Realtime). This extension substitutes **free-tier Gemini**
judges (1–2 models via the rumik-oss author's own API keys), which introduces:

- panel-size reduction (2 vs 3)
- same-family judge risk (their own METHODOLOGY documents this: "judge
  dependence and possible same-family bias")

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

## Reproduction

All prompts, generation configs, WAV outputs, judge raw responses, and scoring
scripts are committed in this repository under `benchmarks/gujarati_indicemo_lite/`.

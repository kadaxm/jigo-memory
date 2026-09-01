# Jigo — a voice-native AI memory assistant

Speak to it; it remembers. Ask it later; it answers out loud — in a synthesized voice,
grounded in what you told it and in your own Obsidian notes.

Jigo is an end-to-end demonstration of a production-shaped memory stack for voice
assistants: browser mic capture → dedicated STT → intent routing → embedding,
salience-scoring, conflict-aware storage in ChromaDB → hybrid-ranked retrieval →
LLM answer synthesis → streaming TTS. It runs as a premium web dashboard, a CLI,
and an HTTP service — all on the same memory core — and publishes honest eval
numbers for every claim.

> 🎬 **Demo:** *screen recording of a live voice turn (spoken question → grounded
> answer in a cloned voice) coming soon — this slot gets the GIF.*

## Why not just similarity search?

A raw "embed everything, cosine the top-k" memory is a toy. Real recall needs three
things pure similarity cannot express:

- **Not every memory matters equally.** Every memory gets an importance
  (**salience**) score at write time, blended into ranking. The score comes from a
  small model trained by **distilling an LLM judge** (teacher labels → Ridge on
  frozen MPNet embeddings), so it's instant and free — with the LLM judge kept as
  fallback.
- **Memories go stale at different rates.** Each memory gets a **type** — episodic,
  semantic, procedural, or knowledge — with decay half-lives of **7 / 30 / 90 / 365
  days**. Retrieval applies exponential recency decay accordingly.
- **People update themselves.** "I live in Delhi" then "I moved to Bangalore" should
  *replace*, not duplicate. A conflict check against the nearest existing memory
  overwrites the old row in place.

Retrieval ranks with `0.6·similarity + 0.25·salience + 0.15·recency`, then adds a
bounded **associative hop**: the best match's own neighbors join at half weight.

**Deliberate deviation:** memory-type classification uses keyword rules rather than
an LLM call — free, instant, deterministic. The LLM budget goes where it matters:
answering.

## Architecture

```
            ┌── STORE: embed → salience (LLM) → type (rules) → conflict check → ChromaDB
BROWSER MIC ┤
            └── RECALL: embed → hybrid rank → associative hop (0.5×)
                        → answer gate (raw ≥ 0.45)
                        → LLM answer synthesis, grounded ONLY in retrieved memories
                        → STREAMING TTS (audio plays as it generates)

STT: ElevenLabs Scribe (primary, ~2s) → Gemini audio (key-rotating fallback)
TEXT: OpenRouter / Ox Alpha (primary) → Gemini rotation (fallback)
TTS: local XTTS-v2 clone server (GPU-accelerated, sentence-streaming) → ElevenLabs fallback
STORAGE: ChromaDB, local, zero-network
```

Every layer has a fallback, and every provider call rotates keys automatically —
the pipeline degrades gracefully instead of dying.

## The web dashboard

`python web_ui.py` → **http://127.0.0.1:8000**

- **The orb** — a shader-displaced dark glass sphere that breathes when idle and
  ignites with ripple rings while you speak (Three.js, reduced-motion safe)
- **Hold-to-talk, or hands-free**: after each reply the mic auto-arms with
  client-side voice-activity detection — chain turns without touching anything
- **Confirm-before-store**: every memory shows "I HEARD — CHECK BEFORE I REMEMBER"
  with an editable transcript. Misheard audio can never silently corrupt memory
- **Memory ledger** (slide-over): every memory with salience and a live *retention*
  meter computed from its type's real half-life; search and delete
- **LAB mode**: one toggle runs each utterance through every transcription engine
  side-by-side — calibration as a feature, not a hidden script
- Per-turn telemetry: transcribe / retrieved / answered / time-to-first-audio ms

## Measured results

| Metric | Result | Source |
|---|---|---|
| Retrieval accuracy (16 fixed pairs) | **100% top-1 / top-3** | `eval.py` |
| Retrieval latency | **P50 76 ms / P95 86 ms** | `benchmark.py` |
| STT latency (Scribe, 5s clip) | **~2.1 s** | LAB comparison |
| Cross-language retrieval (EN↔HI) | **67% top-3**, failures analyzed | `bilingual_test.py` |
| Streaming TTS time-to-first-audio | shown live per turn | dashboard |
| Confidence gate on never-stored topics | **2/2 refusals** | `eval.py` |
| TTS synthesis (8s of speech, same text) | **6.9 s on GPU vs 64.7 s on CPU — 9.4×** | `voice_server.py` benchmark |
| Streaming reply first-audio | **~3.9 s** (sentence-chunked clone stream) | `/voice_stream` |
| Salience model (Ridge, 5-seed CV, n=838) | **Pearson 0.55 / Spearman 0.55 / MAE 0.23** | `compare_salience.py`, `salience_eval.json` |

**Honest failure analysis:** the multilingual encoder (`paraphrase-multilingual-
mpnet-base-v2`) has weak true-Hindi coverage; both bilingual misses are Roman-Hindi ↔
English semantic gaps ("uthta hoon" ↔ "rise"). LaBSE is the known upgrade path,
deliberately deferred.

**A bug the eval caught:** the original confidence gate compared the *hybrid* score —
but salience+recency put a ~0.28 floor under every score, so questions about things
never stored ("What is my cat's name?") scored 0.61+ and got confident wrong answers.
The gate now uses raw embedding similarity. This is why eval sets need adversarial cases.

## Obsidian integration — your second brain, speaking

`python obsidian_sync.py` imports an Obsidian vault (path in `.env` as
`OBSIDIAN_VAULT`): Markdown notes are split by headings, embedded as `knowledge`
memories with a 365-day half-life, hash-deduped, and synced incrementally by file
mtime. Read-only — the vault is never modified.

Result: *"What do my notes say about Gujarat?"* retrieves Gujarati-language notes
from the vault and answers in English, grounded in the actual text.

## Files

| File | Purpose |
|---|---|
| `memory.py` | Core: ChromaDB, salience, typing, conflict resolution, hybrid ranking, associative hop |
| `jigo_voice.py` | Voice pipeline: Scribe-primary STT, intent routing, answer synthesis, streaming TTS (CLI loop) |
| `voice_server.py` | Local XTTS-v2 **voice-clone server** (FastAPI): GPU/CPU auto, OOM fallback, warm-up, sentence-streamed PCM |
| `web_ui.py` + `static/index.html` | Dashboard: orb UI, confirm-before-store, hands-free VAD, LAB, drawer, `/voice_stream` (clone-first, ElevenLabs fallback) |
| `emotion.py` | Speech emotion analysis (per-turn label + intensity) |
| `llm.py` | Provider layer: OpenRouter/Ox Alpha primary → Gemini-rotating fallback for every text call |
| `api.py` | FastAPI service: `/add`, `/search`, `/health` |
| `obsidian_sync.py` | Read-only Obsidian vault importer (incremental, hash-deduped) |
| `benchmark.py` | P50/P95 retrieval latency over 20 queries (self-seeding) |
| `eval.py` | Top-1/top-3 accuracy + adversarial gate tests |
| `bilingual_test.py` | EN↔HI cross-language retrieval test |
| `export_salience_data.py` | Build the salience candidate pool from imported notes |
| `judge_distill.py` | LLM teacher: label vault chunks for salience (checkpointed) |
| `train_salience.py` | Train the MLP salience head on distilled labels |
| `compare_salience.py` | 5-seed MLP-vs-Ridge model selection (CV) |
| `finalize_salience.py` | Fit + save the deployed Ridge (`salience_ridge.npz`) |
| `salience_model.py` | Local salience inference: Ridge → MLP → LLM fallback ladder |

## Setup

Python 3.10+ (tested on 3.14, Windows).

```bash
pip install sentence-transformers chromadb google-genai elevenlabs sounddevice fastapi uvicorn python-dotenv numpy httpx playwright
python -m playwright install chromium   # optional: only for UI development
```

### Voice-clone server (optional — local TTS in a cloned voice)

A side venv (Python 3.11) keeps the heavy TTS stack isolated from the main env:

```bash
python -m venv xtts_env
xtts_env\Scripts\pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126   # or .../whl/cpu without an NVIDIA GPU
xtts_env\Scripts\pip install TTS fastapi uvicorn numpy
xtts_env\Scripts\python voice_server.py   # -> http://127.0.0.1:8100 (GPU if available, CPU fallback, ~85s warm-up)
```

Drop a `voice_sample.wav` (~20 seconds) in the project root and replies are spoken
in that cloned voice — zero-shot, no training data. Without it, a built-in studio
voice is used. The dashboard streams from this server automatically
(`/voice_stream`) and falls back to ElevenLabs whenever it's down.

`.env` in the project root:

```
GEMINI_API_KEY=...            # fallback provider (add _2, _3... for rotation)
ELEVENLABS_API_KEY=...        # STT + TTS (needs speech_to_text scope)
OPENROUTER_API_KEY=...        # primary text provider
OPENROUTER_MODEL=stealth/ox-alpha
OBSIDIAN_VAULT=C:\path\to\your\vault   # optional, for obsidian_sync.py
```

Run:

```bash
python web_ui.py            # dashboard -> http://127.0.0.1:8000
python jigo_voice.py        # terminal voice loop
python memory.py            # text CLI
python obsidian_sync.py     # import your vault
python benchmark.py         # latency P50/P95
python eval.py              # accuracy + adversarial tests
python bilingual_test.py    # EN<->HI retrieval
uvicorn api:app --port 8000 # headless HTTP service
```

First run downloads the embedding model (~420 MB). The vector store persists to
`./chroma_memory/`.

## Notes & trade-offs

- **Local-first storage.** ChromaDB keeps retrieval sub-100ms with zero network;
  only STT/TTS/LLM touch the cloud, and each has a fallback.
- **Dedicated STT beats multimodal LLMs.** Scribe transcribes ~2× faster than
  Gemini-audio with equal accuracy — so Gemini audio survives only as fallback.
- **Human checkpoint on writes.** Voice recognition is good, not perfect; the
  confirm card makes mishearing a one-tap fix instead of silent corruption.
- **Answer synthesis is grounded, not open-ended.** The recall LLM may only use
  retrieved memories and must refuse otherwise — auditable via the UI's
  retrieved-memories list.
- Replies speak in a **cloned voice** via the local XTTS-v2 server — zero-shot
  cloning from a ~20s reference sample, no training run, GPU-accelerated with
  automatic CPU/OOM fallback. ElevenLabs IVC was skipped deliberately: it needs a
  paid tier and uploads your voice to the cloud; the local server keeps the voice
  on your machine.

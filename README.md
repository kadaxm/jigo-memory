# Jigo — a voice-native AI memory assistant

Speak to it; it remembers. Ask it later; it recalls and speaks back.

Jigo is an end-to-end demonstration of a production-shaped memory layer for voice
assistants: mic capture → Gemini transcription + intent classification → embedding,
scoring, conflict-aware storage in ChromaDB → hybrid-ranked retrieval → ElevenLabs
streaming TTS. It also runs as an HTTP service (FastAPI) and is evaluated against
fixed test sets with published numbers.

## Why not just similarity search?

A raw "embed everything, cosine the top-k" memory is a toy. Real recall needs three
things that pure similarity cannot express:

- **Not every memory matters equally.** "I prefer window seats" and "I saw a dog on
  my street today" are both worth storing but not worth ranking equally forever.
  Jigo scores every memory with an LLM-judged **salience** (0–1) at write time, and
  retrieval blends it into ranking.
- **Memories go stale at different rates.** An event from three weeks ago is mostly
  noise; your job title is not. Every memory gets a **type** — episodic, semantic,
  or procedural — with type-specific decay half-lives of **7 / 30 / 90 days**.
  Retrieval applies exponential recency decay accordingly.
- **People update themselves.** "I live in Delhi" followed by "I moved to Bangalore"
  should *replace*, not duplicate. Before writing, Jigo checks the nearest existing
  memory above a similarity threshold and asks an LLM whether the new statement is a
  contradiction/update; if so, the old row is overwritten in place.

The result is a ranking function `0.6·similarity + 0.25·salience + 0.15·recency`
that behaves like a memory instead of a search engine. On top of that, retrieval
does a bounded **associative hop**: the best match's own nearest neighbors are added
at half weight, so related context surfaces without crowding out direct hits.

**Deliberate deviation:** memory-type classification uses keyword rules rather than
an LLM call. It is free, instant, deterministic, and runs offline; salience judgment
is where the LLM actually adds value, so it stays LLM-powered.

## Measured results

All numbers reproducible via the scripts below on any machine.

| Metric | Result |
|---|---|
| Retrieval accuracy (16 fixed query→memory pairs) | **100% top-1 / 100% top-3** (`eval.py`) |
| Retrieval latency, full call incl. embedding | **P50 76 ms / P95 86 ms** (`benchmark.py`) |
| Cross-language retrieval (EN↔HI, 6 pairs) | **67% top-3**, failures analyzed below |
| Streaming TTS time-to-first-audio | **~3.5 s**, vs ~9.3 s to finish generating (`jigo_voice.py`) |
| Confidence gate on never-stored topics | **2/2 refusals** after gate redesign |

**Honest failure analysis — cross-language:** the multilingual encoder
(`paraphrase-multilingual-mpnet-base-v2`) has weak true Hindi coverage despite its
name. Both persistent misses are Roman-Hindi ↔ English semantic gaps ("uthta hoon" ↔
"rise", "kab tak complete" ↔ "deadline"). A stronger bilingual embedder (e.g. LaBSE)
is the known upgrade path; it was tested against spec constraints and deliberately
deferred.

**A bug the eval caught:** the original confidence gate compared the *hybrid* score
against a threshold — but salience + recency put a ~0.28 floor under every hybrid
score, so questions about things never stored ("What is my cat's name?") scored 0.61+
and would have been answered confidently with wrong content. The gate now uses raw
embedding similarity (legit ≥ 0.72, off-topic ≤ 0.64 on the eval set → threshold 0.68)
and refuses cleanly. This is exactly why eval sets include adversarial cases.

## Architecture

```
VOICE IN (mic, sounddevice)
   -> TRANSCRIBE + CLASSIFY INTENT (Gemini, single call, bilingual EN/HI)
       -> STORE path: embed -> salience score (LLM) -> type classify (rules)
                      -> conflict check vs nearest neighbor -> write (ChromaDB)
       -> RECALL path: embed query -> vector search
                       -> hybrid rank (similarity 0.6 + salience 0.25 + recency 0.15)
                       -> optional associative hop (neighbors @ 0.5x)
                       -> raw-similarity confidence gate
   -> COMPOSE reply
   -> SPEAK (ElevenLabs streaming TTS, playback starts before generation completes)
-> VOICE OUT
```

Cross-cutting: latency measured at retrieval; the same memory core exposed as a
FastAPI service; fixed-seed eval/benchmark suites for reproducible numbers.

## Files

| File | Purpose |
|---|---|
| `memory.py` | Core: ChromaDB storage, salience, typing, conflict resolution, hybrid ranking, associative hop |
| `jigo_voice.py` | Full voice loop: record → transcribe/classify → route → stream-speak |
| `web_ui.py` + `static/index.html` | Browser dashboard: hold-to-talk voice console, memory ledger with salience/decay meters, delete |
| `api.py` | FastAPI service exposing `/add`, `/search`, `/health` |
| `clone_voice.py` | One-time ElevenLabs Instant Voice Cloning setup (requires paid plan; stock voice used by default) |
| `benchmark.py` | P50/P95 retrieval latency over 20 queries against a seeded store |
| `eval.py` | Top-1/top-3 accuracy over 16 pairs + adversarial gate tests |
| `bilingual_test.py` | Cross-language (English ↔ Hindi-Roman) retrieval stress test |

## Setup

Requires Python 3.10+. Tested on Python 3.14, Windows.

```bash
pip install sentence-transformers chromadb google-genai elevenlabs sounddevice fastapi uvicorn python-dotenv numpy
```

Create `.env` in the project root:

```
GEMINI_API_KEY=your_google_ai_studio_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

Run:

```bash
python jigo_voice.py        # full spoken loop (Enter to talk)
python web_ui.py            # browser dashboard -> http://127.0.0.1:8000
python memory.py            # text CLI: 'add: <text>' to store, type to search
python benchmark.py         # latency P50/P95
python eval.py              # accuracy + adversarial gate tests
python bilingual_test.py    # EN<->HI retrieval test
uvicorn api:app --port 8000 # HTTP service
python clone_voice.py       # optional: clone your own voice (paid plan required)
```

First run downloads the embedding model (~420 MB). The vector store persists locally
to `./chroma_memory/`.

The dashboard records through the browser's own mic (Web Audio → 16 kHz WAV), so it
works on any device that can reach the server; browsers only grant mic access on
`localhost` or HTTPS. Each turn shows its telemetry — transcribe / memory / TTS ms,
and the raw-similarity gate score for recalls — and the ledger renders every memory's
salience plus a live retention meter computed from its type's decay half-life.

## Notes & trade-offs

- **Local-first storage.** ChromaDB keeps the whole pipeline zero-network except the
  two model APIs; retrieval stays sub-100 ms because nothing leaves the machine.
- **Single LLM call per turn.** Transcription and intent share one Gemini call;
  salience rides the storage path only. Latency budget goes to audio, not JSON.
- **Streaming speech.** Playback begins on the first audio chunk (~3.5 s), not after
  the full response generates (~9 s for long sentences).
- Voice cloning code ships complete (`clone_voice.py`) but ElevenLabs IVC requires a
  paid tier; the loop falls back to a stock voice until `JIGO_VOICE_ID` exists in `.env`.

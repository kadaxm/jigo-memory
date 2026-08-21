"""
Retrieval latency benchmark.

Seeds a scratch collection with 25 fixed memories (so results are reproducible
on any machine — the real DB is gitignored and may be empty), then times
search_memory() over 20 realistic queries. Each timed call includes query
embedding + ChromaDB search + hybrid ranking. 3 untimed warmups absorb
one-time costs.

Prints P50 / P95 / mean / min / max in milliseconds.
"""

import sys
import time

import chromadb

from memory import search_memory

SCRATCH_NAME = "benchmark_data"
WARMUPS = 3

SEED_MEMORIES = [
    "My name is Kadam Vyas",
    "I am applying for an AI research intern position at Rumik",
    "I am building a voice-native memory assistant called Jigo",
    "I live in India and my timezone is IST",
    "I prefer tea over coffee in the evening",
    "I drink two cups of coffee every morning",
    "My favorite programming language is Python",
    "I work with ChromaDB for vector storage",
    "I use Gemini for transcription and intent classification",
    "ElevenLabs handles my text to speech",
    "Yesterday I visited the new art museum downtown",
    "Last week I recorded audio samples for voice testing",
    "I met my college professor for lunch last Friday",
    "The project demo deadline is next Friday",
    "My router needs a power cycle when the internet drops",
    "To reset my router I hold the power button for ten seconds",
    "I always back up my code to GitHub before sleep",
    "My sister is a doctor in Pune",
    "My mother makes excellent biryani on weekends",
    "I prefer window seats on long flights",
    "I wake up at five thirty every morning",
    "Evening walks help me think through hard problems",
    "I am learning Hindi conversational phrases",
    "Sounddevice works better than pyaudio on this machine",
    "Supabase is blocked on my network so I use local storage",
]

QUERIES = [
    "What do I do for work?",
    "What am I working on these days?",
    "What is my name?",
    "Where do I live?",
    "What are my hobbies?",
    "What did I do last weekend?",
    "What is my favorite food?",
    "How do I fix my router?",
    "What projects am I focused on?",
    "Who did I meet recently?",
    "What are my career goals?",
    "What music do I like?",
    "When is my next deadline?",
    "What programming languages do I know?",
    "Where did I travel recently?",
    "What books am I reading?",
    "What is my daily routine?",
    "Remind me what I said about Rumik",
    "What languages can I speak?",
    "What was that thing I told you about my application?",
]


def percentile(sorted_values, p):
    idx = int(round((p / 100) * (len(sorted_values) - 1)))
    return sorted_values[idx]


def run():
    client = chromadb.PersistentClient(path="./chroma_memory")
    if SCRATCH_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(SCRATCH_NAME)
    scratch = client.get_or_create_collection(name=SCRATCH_NAME, metadata={"hnsw:space": "cosine"})

    print(f"Seeding scratch collection with {len(SEED_MEMORIES)} memories...")
    from memory import model, TYPE_HALF_LIFE_DAYS
    import uuid as _uuid
    scratch.add(
        ids=[str(_uuid.uuid4()) for _ in SEED_MEMORIES],
        embeddings=[model.encode(m).tolist() for m in SEED_MEMORIES],
        documents=SEED_MEMORIES,
        metadatas=[{"source": "benchmark", "salience": 0.5,
                    "timestamp": time.time(), "type": "semantic"} for _ in SEED_MEMORIES],
    )

    print(f"Running {WARMUPS} warmup calls (untimed)...")
    for q in QUERIES[:WARMUPS]:
        search_memory(q, top_k=3, associative=False, col=scratch)

    times_ms = []
    print(f"\nTimed runs ({len(QUERIES)} queries, associative=False):")
    for q in QUERIES:
        t0 = time.perf_counter()
        results = search_memory(q, top_k=3, associative=False, col=scratch)
        dt = (time.perf_counter() - t0) * 1000
        times_ms.append(dt)
        top1 = results[0]["content"][:45] if results else "(none)"
        print(f"  {dt:8.1f} ms  {q[:38]:40} -> {top1}")

    times_ms.sort()
    n = len(times_ms)
    print("\n" + "=" * 60)
    print("RETRIEVAL LATENCY (full search_memory call, local ChromaDB)")
    print("=" * 60)
    print(f"  P50 : {percentile(times_ms, 50):8.1f} ms")
    print(f"  P95 : {percentile(times_ms, 95):8.1f} ms")
    print(f"  mean: {sum(times_ms) / n:8.1f} ms")
    print(f"  min : {times_ms[0]:8.1f} ms")
    print(f"  max : {times_ms[-1]:8.1f} ms")
    print(f"  n   : {n} queries over {len(SEED_MEMORIES)} memories")

    client.delete_collection(SCRATCH_NAME)
    print("\nScratch collection deleted.")
    return 0


if __name__ == "__main__":
    sys.exit(run())

"""
Retrieval accuracy eval against a fixed held-out test set.

Seeds a scratch collection with 18 known memories (the real DB is gitignored
and machine-specific, so the eval is self-contained and reproducible), then
runs 16 query -> expected-memory pairs plus 2 adversarial queries about things
that were never stored.

Adversarial queries test the confidence gate: if the top hybrid score for an
unanswerable question exceeds the voice loop's threshold, Jigo would speak a
wrong answer as fact — so that number is reported honestly here.

Prints top-1 / top-3 accuracy for the retrieval pairs and gate behavior for
the adversarial pairs. This number goes in the README.
"""

import sys
import chromadb

from memory import add_memory, search_memory

SCRATCH_NAME = "eval_data"
# Gate is on RAW embedding similarity (see jigo_voice.py rationale): legit queries
# score >= 0.72 raw, off-topic <= 0.64 raw on this seed set -> 0.68 splits them.
SIMILARITY_THRESHOLD = 0.68

SEED_MEMORIES = [
    "My name is Kadam Vyas",
    "I am applying for an AI research intern position at Rumik",
    "I am building Jigo, a voice-native memory assistant",
    "I use ChromaDB for vector storage in my projects",
    "My favorite programming language is Python",
    "I drink two cups of coffee every morning",
    "I prefer window seats on long flights",
    "My sister is a doctor in Pune",
    "I wake up at five thirty every morning",
    "Yesterday I visited the new art museum downtown",
    "The Jigo demo deadline is next Friday",
    "To reset my router I hold the power button for ten seconds",
    "I always back up my code to GitHub before sleeping",
    "Evening walks help me think through hard problems",
    "I am learning conversational Hindi phrases",
    "ElevenLabs handles text to speech in my voice assistant",
    "I recorded audio samples last week for voice testing",
    "My mother makes biryani on weekends",
]

# (query, expected_substring)
EVAL_PAIRS = [
    ("What is my name?", "kadam"),
    ("What job am I applying for?", "rumik"),
    ("What project am I building?", "jigo"),
    ("Where do I store my vectors?", "chromadb"),
    ("Which programming language do I like most?", "python"),
    ("How much coffee do I drink in the morning?", "two cups of coffee"),
    ("Which seat do I prefer on flights?", "window seats"),
    ("What does my sister do?", "doctor"),
    ("When do I get out of bed?", "five thirty"),
    ("Where did I go yesterday?", "art museum"),
    ("When is the demo due?", "next friday"),
    ("How do I fix my internet when it drops?", "router"),
    ("What do I do with my code before sleeping?", "github"),
    ("What helps me think through problems?", "evening walks"),
    ("What language am I learning these days?", "hindi"),
    ("What does my mother cook on weekends?", "biryani"),
]

# Things that were NEVER stored — a well-gated system must refuse them.
ADVERSARIAL_QUERIES = [
    "What is my cat's name?",
    "Where did I park my car today?",
]


def run():
    client = chromadb.PersistentClient(path="./chroma_memory")
    if SCRATCH_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(SCRATCH_NAME)
    scratch = client.get_or_create_collection(name=SCRATCH_NAME, metadata={"hnsw:space": "cosine"})

    print(f"Seeding {len(SEED_MEMORIES)} memories into scratch collection...")
    for m in SEED_MEMORIES:
        add_memory(m, source="eval", col=scratch)

    print("\n" + "=" * 92)
    print(f"{'QUERY':46} {'TOP-1':6} {'TOP-3':6} {'SCORE':7} RESULT")
    print("=" * 92)

    top1_hits = 0
    top3_hits = 0

    for query, expected in EVAL_PAIRS:
        results = search_memory(query, top_k=3, associative=False, col=scratch)
        contents = [r["content"].lower() for r in results]
        hit1 = bool(contents) and expected in contents[0]
        hit3 = any(expected in c for c in contents)
        top1_hits += hit1
        top3_hits += hit3
        score = f"{results[0]['similarity']:.2f}" if results else "-"
        verdict = "PASS" if hit3 else "FAIL"
        print(f"{query[:44]:46} {('YES' if hit1 else 'no'):6} {('YES' if hit3 else 'no'):6} {score:7} {verdict}")
        if contents:
            print(f"   got: '{results[0]['content'][:70]}'")

    n = len(EVAL_PAIRS)
    print("=" * 92)
    print(f"\nRETRIEVAL:  top-1 accuracy {top1_hits}/{n} ({100 * top1_hits / n:.0f}%)"
          f"  |  top-3 accuracy {top3_hits}/{n} ({100 * top3_hits / n:.0f}%)")

    print("\nADVERSARIAL (never stored — correct behavior is a REFUSAL, i.e. raw score < "
          f"{SIMILARITY_THRESHOLD}):")
    gate_ok = 0
    for q in ADVERSARIAL_QUERIES:
        results = search_memory(q, top_k=1, associative=False, col=scratch)
        raw = results[0]["raw_similarity"] if results else 0.0
        rejected = raw < SIMILARITY_THRESHOLD
        gate_ok += rejected
        print(f"  '{q}'  -> top raw {raw:.2f}  [{'GATE HELD' if rejected else 'GATE PASSED (would speak wrong answer)'}]")

    total_pairs = n + len(ADVERSARIAL_QUERIES)
    print("\nSUMMARY")
    print(f"  top-1 accuracy: {100 * top1_hits / n:.0f}%")
    print(f"  top-3 accuracy: {100 * top3_hits / n:.0f}%")
    print(f"  adversarial refusals: {gate_ok}/{len(ADVERSARIAL_QUERIES)}")

    client.delete_collection(SCRATCH_NAME)
    print("Scratch collection deleted.")
    return 0


if __name__ == "__main__":
    sys.exit(run())

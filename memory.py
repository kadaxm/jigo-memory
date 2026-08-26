from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import json
import math
import time
import re
import os

import llm
from dotenv import load_dotenv

# --- CONFIG ---
load_dotenv()
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_memory")
COLLECTION_NAME = "memories"
CONFLICT_SIMILARITY_THRESHOLD = 0.75
SIMILARITY_WEIGHT = 0.6
SALIENCE_WEIGHT = 0.25
RECENCY_WEIGHT = 0.15
ASSOCIATIVE_NEIGHBORS = 2   # how many extra "related" memories to pull per top match
ASSOCIATIVE_WEIGHT = 0.5    # neighbors compete at half strength so they only surface when nothing strong exists
TYPE_HALF_LIFE_DAYS = {
    "episodic": 7,      # events go stale fast
    "semantic": 30,     # facts about identity/goals stay relevant longer
    "procedural": 90,   # how-to knowledge barely decays
    "knowledge": 365,   # imported second-brain notes (Obsidian) — reference knowledge
}
# -------------------------------------------

print("Loading embedding model... (first run may take a minute)")
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

client = chromadb.PersistentClient(path=DB_PATH)
DISTANCE_SPACE = "cosine"  # cosine distance [0,2]; 1/(1+dist) then maps near-duplicates ~1.0, unrelated ~0.4
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": DISTANCE_SPACE},
)


def _ensure_cosine_space():
    """Rebuild the collection as cosine if it was created with a different space."""
    global collection
    if collection.configuration["hnsw"]["space"] == DISTANCE_SPACE:
        return
    print(f"Migrating '{COLLECTION_NAME}' from {collection.configuration['hnsw']['space']} to {DISTANCE_SPACE}...")
    backup = collection.get(include=["documents", "metadatas"])
    client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": DISTANCE_SPACE},
    )
    docs = backup["documents"]
    metas = backup["metadatas"]
    if docs:
        embeddings = [model.encode(d).tolist() for d in docs]
        collection.add(ids=[str(uuid.uuid4()) for _ in docs], embeddings=embeddings,
                       documents=docs, metadatas=metas)
        print(f"Re-embedded {len(docs)} existing memories.")


_ensure_cosine_space()


# ---------- LLM helpers ----------

def _llm_judge(prompt):
    return llm.chat_json(prompt, max_tokens=200, temperature=0.2)


def _score_salience(content):
    prompt = (
        f'Rate how important this memory is to remember long-term, from 0.0 (trivial, forgettable) '
        f'to 1.0 (critical, identity-defining). Content: "{content}". '
        'Respond ONLY with JSON: {"salience": 0.0}'
    )
    try:
        result = _llm_judge(prompt)
        return float(result.get("salience", 0.5))
    except Exception:
        return 0.5


def _check_conflict(new_content, existing_content):
    prompt = (
        f'Existing memory: "{existing_content}"\n'
        f'New statement: "{new_content}"\n'
        'Does the new statement contradict, correct, or update the existing memory '
        '(e.g. a changed fact, a moved location, a new job replacing an old one)? '
        'Respond ONLY with JSON: {"conflict": true or false}'
    )
    try:
        result = _llm_judge(prompt)
        return bool(result.get("conflict", False))
    except Exception:
        return False


# ---------- Memory type classification (keyword rules — instant, no LLM call) ----------

_PROCEDURAL_PATTERNS = [
    r"\bhow to\b", r"\bhow do i\b", r"\bi always\b", r"\bsteps? to\b",
    r"\bthe way to\b", r"\bprocess for\b", r"\bmethod for\b",
]
_EPISODIC_PATTERNS = [
    r"\bremember when\b", r"\byesterday\b", r"\blast week\b", r"\blast night\b",
    r"\bwent to\b", r"\bvisited\b", r"\bhappened\b", r"\bi saw\b", r"\bi met\b",
    r"\btoday i\b", r"\bearlier i\b",
]
_SEMANTIC_PATTERNS = [
    r"\bi am\b", r"\bi work\b", r"\bi have\b", r"\bi live\b", r"\bmy name is\b",
    r"\bi like\b", r"\bi prefer\b", r"\bi own\b",
]


def _classify_memory_type(content):
    """Rule-based classification: procedural > episodic > semantic > default semantic.

    Deliberately not an LLM call: it is free, instant, and deterministic.
    """
    lowered = content.lower()

    for pattern in _PROCEDURAL_PATTERNS:
        if re.search(pattern, lowered):
            return "procedural"
    for pattern in _EPISODIC_PATTERNS:
        if re.search(pattern, lowered):
            return "episodic"
    for pattern in _SEMANTIC_PATTERNS:
        if re.search(pattern, lowered):
            return "semantic"

    return "semantic"  # default fallback — most general-purpose


# ---------- Recency ----------

def _recency_score(timestamp, memory_type="semantic"):
    age_days = (time.time() - timestamp) / 86400
    half_life = TYPE_HALF_LIFE_DAYS.get(memory_type, TYPE_HALF_LIFE_DAYS["semantic"])
    return math.exp(-age_days * (math.log(2) / half_life))


# ---------- Store ----------

def _salience_for(content, embedding):
    """Fine-tuned local head first (instant, reuses the stored embedding);
    LLM judge as fallback when the head is unavailable."""
    try:
        from salience_model import predict_salience
        return round(predict_salience(embedding), 3), "model"
    except Exception:
        return _score_salience(content), "llm"


def add_memory(content, source="manual", col=None):
    col = col or collection
    embedding = model.encode(content).tolist()
    memory_type = _classify_memory_type(content)
    salience, salience_src = _salience_for(content, embedding)

    existing = col.query(query_embeddings=[embedding], n_results=1)
    if existing["documents"] and existing["documents"][0]:
        old_content = existing["documents"][0][0]
        old_id = existing["ids"][0][0]
        old_distance = existing["distances"][0][0]
        old_similarity = 1 / (1 + old_distance)

        if old_similarity >= CONFLICT_SIMILARITY_THRESHOLD:
            if _check_conflict(content, old_content):
                col.update(
                    ids=[old_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[{
                        "source": source,
                        "salience": salience,
                        "timestamp": time.time(),
                        "type": memory_type,
                    }],
                )
                print(f"Updated (was: '{old_content[:40]}...') [{memory_type}]: {content[:50]}...")
                return old_id

    memory_id = str(uuid.uuid4())
    col.add(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{
            "source": source,
            "salience": salience,
            "timestamp": time.time(),
            "type": memory_type,
        }],
    )
    print(f"Stored (salience {salience:.2f} via {salience_src}, type={memory_type}): {content[:50]}...")
    return memory_id


# ---------- Search with associative retrieval ----------

def search_memory(query, top_k=3, associative=True, col=None):
    """
    Retrieve up to top_k direct hits ranked by similarity+salience+recency.
    If associative=True, also pulls ASSOCIATIVE_NEIGHBORS related memories for the single
    best match (multi-hop). They are scored with the same hybrid formula, multiplied by
    ASSOCIATIVE_WEIGHT, tagged is_associative, and appended to the returned list — so they
    rank below any strong direct hit but still surface as "related" extras (list length
    can exceed top_k; filter on is_associative if strict top_k semantics are needed).
    """
    col = col or collection
    query_embedding = model.encode(query).tolist()
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=max(top_k * 3, 10),
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    docs = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]

    def _score_row(doc, dist, meta, doc_id):
        raw_similarity = 1 / (1 + dist)
        salience = meta.get("salience", 0.5)
        recency = _recency_score(meta.get("timestamp", time.time()), meta.get("type", "semantic"))
        final_score = (
            SIMILARITY_WEIGHT * raw_similarity
            + SALIENCE_WEIGHT * salience
            + RECENCY_WEIGHT * recency
        )
        return {
            "id": doc_id,
            "content": doc,
            "similarity": final_score,
            "raw_similarity": raw_similarity,
            "salience": salience,
            "recency": recency,
            "type": meta.get("type", "semantic"),
            "source": meta.get("source"),
            "is_associative": False,
        }

    scored = [_score_row(d, dist, m, i) for d, dist, m, i in zip(docs, distances, metadatas, ids)]
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = scored[:top_k]

    if associative and top_results:
        best = top_results[0]
        best_embedding = model.encode(best["content"]).tolist()
        neighbor_results = col.query(
            query_embeddings=[best_embedding],
            n_results=ASSOCIATIVE_NEIGHBORS + top_k + 1,  # extra buffer to filter out dupes
        )

        seen_ids = {r["id"] for r in top_results}
        added = 0
        if neighbor_results["documents"] and neighbor_results["documents"][0]:
            for doc, dist, meta, doc_id in zip(
                neighbor_results["documents"][0],
                neighbor_results["distances"][0],
                neighbor_results["metadatas"][0],
                neighbor_results["ids"][0],
            ):
                if doc_id in seen_ids:
                    continue
                if added >= ASSOCIATIVE_NEIGHBORS:
                    break
                row = _score_row(doc, dist, meta, doc_id)
                row["similarity"] *= ASSOCIATIVE_WEIGHT
                row["is_associative"] = True
                top_results.append(row)
                seen_ids.add(doc_id)
                added += 1

        top_results.sort(key=lambda x: x["similarity"], reverse=True)

    return top_results


if __name__ == "__main__":
    print("Memory system ready.")
    print("Commands: type text to search, 'add: <text>' to store, 'quit' to exit.\n")

    while True:
        query = input("> ").strip()
        if query.lower() == "quit":
            break

        if query.lower().startswith("add:"):
            content = query[4:].strip()
            if content:
                add_memory(content)
            else:
                print("Nothing to add.")
            continue

        results = search_memory(query)
        if not results:
            print("(no memories stored yet — try 'add: <something>' first)")
            continue

        for r in results:
            tag = " [associative]" if r["is_associative"] else ""
            print(f"[{r['similarity']:.2f}] ({r['type']}){tag} {r['content']}  (sal={r['salience']:.2f}, rec={r['recency']:.2f})")

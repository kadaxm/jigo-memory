"""
Obsidian vault -> Jigo memory importer. READ-ONLY: never writes to the vault.

Usage:  python obsidian_sync.py [vault_path]
        (vault path defaults to OBSIDIAN_VAULT in .env)

- Parses .md notes only; skips .obsidian/ and attachments/ folders
- Splits each note by headings; long sections split into ~500-word windows
- Every chunk: type=knowledge (365-day half-life), salience 0.6,
  source=obsidian:<relative path> (shows as a distinct badge in the drawer)
- Content-hash dedup: re-running never creates duplicates
- Incremental: only notes whose file mtime changed get re-embedded
"""

import hashlib
import os
import re
import sys
import time
import uuid

import chromadb
from dotenv import load_dotenv

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv(os.path.join(PROJECT, ".env"))

from memory import model  # noqa: E402  (embedding model, loads once)

VAULT = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OBSIDIAN_VAULT")
SKIP_DIRS = {".obsidian", "attachments"}
HEADING = re.compile(r"^(#{2,6})\s+(.+)$", re.M)
BATCH = 64


def chunk_note(text):
    """Split note into (section_title, body) chunks; long bodies get word windows."""
    parts = []
    matches = list(HEADING.finditer(text))
    if not matches:
        parts.append(("note", text))
    else:
        intro = text[:matches[0].start()].strip()
        if intro:
            parts.append(("intro", intro))
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            title = m.group(2).strip()
            body = text[m.start():end].strip()
            if body:
                parts.append((title, body))
    out = []
    for title, body in parts:
        words = body.split()
        if len(words) > 650:
            for w0 in range(0, len(words), 500):
                seg = " ".join(words[w0:w0 + 500]).strip()
                if seg:
                    out.append((title, seg))
        else:
            out.append((title, body))
    return out


def main():
    if not VAULT or not os.path.isdir(VAULT):
        sys.exit(f"Vault not found: {VAULT!r}  (pass a path or set OBSIDIAN_VAULT in .env)")

    client = chromadb.PersistentClient(path=os.path.join(PROJECT, "chroma_memory"))
    col = client.get_or_create_collection("memories", metadata={"hnsw:space": "cosine"})

    print("Loading existing chunk index...")
    existing = col.get(include=["metadatas"])
    known_hashes = set()
    by_source = {}
    for cid, meta in zip(existing["ids"], existing["metadatas"]):
        meta = meta or {}
        h = meta.get("hash")
        if h:
            known_hashes.add(h)
        src = str(meta.get("source", ""))
        if src.startswith("obsidian:"):
            by_source.setdefault(src, []).append((cid, float(meta.get("mtime", 0))))

    md_files = []
    for root, dirs, files in os.walk(VAULT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if fn.endswith(".md"):
                md_files.append(os.path.join(root, fn))
    print(f"Found {len(md_files)} markdown notes in vault.")

    added = updated_notes = unchanged = empty = 0
    batch_ids, batch_docs, batch_metas = [], [], []

    def flush():
        nonlocal added
        if not batch_ids:
            return
        col.add(
            ids=batch_ids,
            embeddings=[model.encode(d).tolist() for d in batch_docs],
            documents=batch_docs,
            metadatas=batch_metas,
        )
        added += len(batch_ids)
        batch_ids.clear(); batch_docs.clear(); batch_metas.clear()

    for path in md_files:
        rel = os.path.relpath(path, VAULT).replace("\\", "/")
        src = "obsidian:" + rel
        mtime = os.path.getmtime(path)
        old = by_source.get(src)
        if old and all(m >= mtime for _, m in old):
            unchanged += 1
            continue
        if old:
            col.delete(ids=[cid for cid, _ in old])
            updated_notes += 1
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue
        chunks = chunk_note(text)
        if not chunks:
            empty += 1
            continue
        for title, seg in chunks:
            h = hashlib.sha1((rel + "|" + title + "|" + seg).encode()).hexdigest()
            if h in known_hashes:
                continue
            known_hashes.add(h)
            doc = f"{title}: {seg}" if title and title != "intro" else seg
            batch_ids.append(str(uuid.uuid4()))
            batch_docs.append(doc)
            batch_metas.append({
                "source": src, "salience": 0.6, "timestamp": time.time(),
                "type": "knowledge", "hash": h, "mtime": mtime,
            })
            if len(batch_ids) >= BATCH:
                flush()
        print(f"  {rel}: {len(chunks)} chunks")
    flush()

    print(f"\nDone. added={added} chunks | updated_notes={updated_notes} | "
          f"unchanged={unchanged} | empty={empty}")
    print(f"Collection now holds {col.count()} total chunks.")


if __name__ == "__main__":
    main()

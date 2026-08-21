"""
FastAPI wrapper around the Jigo memory system.

Proves the memory layer runs as a real service, independent of the CLI/voice
scripts. Logic is imported from memory.py — zero duplication.

Run:  uvicorn api:app --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel

from memory import add_memory, search_memory

app = FastAPI(title="Jigo Memory Service", version="1.0")


class AddRequest(BaseModel):
    content: str
    source: str = "api"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/add")
def add(req: AddRequest):
    memory_id = add_memory(req.content, req.source)
    return {"id": memory_id, "status": "stored"}


@app.post("/search")
def search(req: SearchRequest):
    results = search_memory(req.query, top_k=max(1, min(req.top_k, 10)), associative=False)
    return {
        "results": [
            {
                "content": r["content"],
                "score": round(r["similarity"], 4),
                "type": r["type"],
                "salience": r["salience"],
                "is_associative": r["is_associative"],
            }
            for r in results
        ]
    }

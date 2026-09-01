"""Local salience inference: frozen encoder embedding -> trained model (~instant).

Replaces the LLM salience call on the store path. Falls back gracefully:
Ridge (preferred, 5-seed CV winner) -> MLP head -> raise (caller uses LLM judge).
"""

import os

import numpy as np
import torch
import torch.nn as nn

PROJECT = os.path.dirname(os.path.abspath(__file__))
HEAD_PATH = os.path.join(PROJECT, "salience_head.pt")
RIDGE_PATH = os.path.join(PROJECT, "salience_ridge.npz")
_ridge = None
_head = None
_loaded = False


class Head(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _load():
    global _ridge, _head, _loaded
    if _loaded:
        return
    _loaded = True
    if os.path.exists(RIDGE_PATH):
        try:
            d = np.load(RIDGE_PATH)
            _ridge = (d["W"].astype(np.float64), float(d["b"]))
            return
        except Exception as e:
            print(f"[salience ridge load failed: {e}]")
    if not os.path.exists(HEAD_PATH):
        return
    try:
        ckpt = torch.load(HEAD_PATH, map_location="cpu", weights_only=True)
        _head = Head()
        _head.load_state_dict(ckpt["state"])
        _head.eval()
    except Exception as e:
        print(f"[salience head load failed: {e}]")
        _head = None


def predict_salience(embedding):
    """embedding: 768-dim list/array (the SAME vector stored in ChromaDB — zero extra encode cost)."""
    _load()
    x = np.asarray(embedding, dtype=np.float64)
    if _ridge is not None:
        W, b = _ridge
        return float(np.clip(W @ x + b, 0.0, 1.0))
    if _head is None:
        raise RuntimeError("salience model not available")
    with torch.no_grad():
        return float(_head(torch.tensor(np.array([x], dtype=np.float32))).item())

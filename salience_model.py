"""Local salience inference: frozen encoder embedding -> trained head (~instant).

Replaces the LLM salience call on the store path. Falls back gracefully if the
trained head is missing (add_memory then uses the LLM judge).
"""

import os

import numpy as np
import torch
import torch.nn as nn

PROJECT = os.path.dirname(os.path.abspath(__file__))
HEAD_PATH = os.path.join(PROJECT, "salience_head.pt")
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
    global _head, _loaded
    if _loaded:
        return _head
    _loaded = True
    if not os.path.exists(HEAD_PATH):
        return None
    try:
        ckpt = torch.load(HEAD_PATH, map_location="cpu", weights_only=True)
        _head = Head()
        _head.load_state_dict(ckpt["state"])
        _head.eval()
    except Exception as e:
        print(f"[salience head load failed: {e}]")
        _head = None
    return _head


def predict_salience(embedding):
    """embedding: 768-dim list/array (the SAME vector stored in ChromaDB — zero extra encode cost)."""
    head = _load()
    if head is None:
        raise RuntimeError("salience head not available")
    with torch.no_grad():
        x = torch.tensor(np.array([embedding], dtype=np.float32))
        return float(head(x).item())

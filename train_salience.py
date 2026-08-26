"""Train the salience head: frozen MPNet embeddings -> small MLP -> salience score.

Dataset: LLM-judged (text -> salience) pairs distilled from the vault
(salience_dataset.json). Reports Pearson/Spearman/MAE on a held-out test split
and saves the best head to salience_head.pt.
"""

import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 42
EPOCHS = 400
PATIENCE = 60
LR = 1e-3

random.seed(SEED)
torch.manual_seed(SEED)


def pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    def rank(x):
        order = np.argsort(x)
        ranks = np.empty(len(x))
        ranks[order] = np.arange(len(x))
        return ranks
    return pearson(rank(a), rank(b))


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


def main():
    from memory import model  # frozen MPNet encoder

    with open(os.path.join(PROJECT, "salience_dataset.json"), encoding="utf-8") as f:
        data = json.load(f)
    items = data["judged"]
    print(f"dataset: {len(items)} judged examples")
    texts = [x["text"] for x in items]
    ys = np.array([x["salience"] for x in items], dtype=np.float32)

    print("embedding (frozen encoder)...")
    E = np.array([model.encode(t) for t in texts], dtype=np.float32)

    idx = list(range(len(texts)))
    random.shuffle(idx)
    n_train = int(0.8 * len(idx))
    n_val = int(0.1 * len(idx))
    tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    E_tr, y_tr = E[tr], ys[tr]
    E_va, y_va = E[va], ys[va]
    E_te, y_te = E[te], ys[te]
    print(f"split: train={len(tr)} val={len(va)} test={len(te)}")

    head = Head()
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.MSELoss()
    E_tr_t = torch.tensor(E_tr); y_tr_t = torch.tensor(y_tr)
    E_va_t = torch.tensor(E_va); y_va_t = torch.tensor(y_va)

    best = {"val_loss": float("inf"), "state": None, "epoch": 0}
    for epoch in range(EPOCHS):
        head.train()
        opt.zero_grad()
        pred = head(E_tr_t)
        loss = lossf(pred, y_tr_t)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            vl = lossf(head(E_va_t), y_va_t).item()
        if vl < best["val_loss"]:
            best = {"val_loss": vl, "state": {k: v.clone() for k, v in head.state_dict().items()}, "epoch": epoch}
        if epoch % 50 == 0:
            print(f"  epoch {epoch}: train {loss.item():.4f} val {vl:.4f}")
        if epoch - best["epoch"] > PATIENCE:
            print(f"  early stop at {epoch} (best epoch {best['epoch']})")
            break

    head.load_state_dict(best["state"])
    head.eval()
    with torch.no_grad():
        p_te = head(torch.tensor(E_te)).numpy()
        p_all = head(torch.tensor(E)).numpy().tolist()

    print("\n=== TEST SET (held-out) ===")
    print(f"pearson : {pearson(p_te, y_te):.3f}")
    print(f"spearman: {spearman(p_te, y_te):.3f}")
    print(f"MAE     : {float(np.abs(p_te - y_te).mean()):.3f}")

    # sklearn Ridge baseline for comparison
    try:
        from sklearn.linear_model import Ridge
        rg = Ridge(alpha=1.0).fit(E_tr, y_tr)
        rp = rg.predict(E_te)
        print(f"[baseline ridge] pearson {pearson(rp, y_te):.3f} mae {float(np.abs(rp - y_te).mean()):.3f}")
    except ImportError:
        print("[sklearn not installed — ridge baseline skipped]")

    torch.save({"state": head.state_dict(), "val_loss": best["val_loss"],
                "test_pearson": pearson(p_te, y_te)}, os.path.join(PROJECT, "salience_head.pt"))
    with open(os.path.join(PROJECT, "salience_eval.json"), "w", encoding="utf-8") as f:
        json.dump({"test_pearson": pearson(p_te, y_te), "test_spearman": spearman(p_te, y_te),
                   "test_mae": float(np.abs(p_te - y_te).mean()), "n": len(items)}, f, indent=2)
    print("\nsaved salience_head.pt + salience_eval.json")


if __name__ == "__main__":
    main()

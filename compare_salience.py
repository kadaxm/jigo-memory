"""Multi-seed comparison: MLP head vs Ridge on the full judged dataset.

Decides which model ships as the salience scorer. Reports mean +/- std across
5 seeds for Pearson/Spearman/MAE on held-out test splits.
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

EPOCHS = 400
PATIENCE = 60
LR = 1e-3
SEEDS = [42, 7, 123, 2024, 99]


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
    texts = [x["text"] for x in items]
    ys = np.array([x["salience"] for x in items], dtype=np.float32)
    print(f"dataset: {len(items)}")

    E = np.array([model.encode(t) for t in texts], dtype=np.float32)

    idx_all = list(range(len(texts)))
    results = {"mlp": [], "ridge": []}

    try:
        from sklearn.linear_model import Ridge
        have_sklearn = True
    except ImportError:
        have_sklearn = False

    for seed in SEEDS:
        random.seed(seed)
        idx = idx_all[:]
        random.shuffle(idx)
        n_train = int(0.8 * len(idx))
        n_val = int(0.1 * len(idx))
        tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
        E_tr, y_tr = E[tr], ys[tr]
        E_va, y_va = E[va], ys[va]
        E_te, y_te = E[te], ys[te]

        torch.manual_seed(seed)
        head = Head()
        opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
        lossf = nn.MSELoss()
        E_tr_t = torch.tensor(E_tr); y_tr_t = torch.tensor(y_tr)
        E_va_t = torch.tensor(E_va); y_va_t = torch.tensor(y_va)

        best = {"val_loss": float("inf"), "state": None, "epoch": 0}
        for epoch in range(EPOCHS):
            head.train()
            opt.zero_grad()
            loss = lossf(head(E_tr_t), y_tr_t)
            loss.backward()
            opt.step()
            head.eval()
            with torch.no_grad():
                vl = lossf(head(E_va_t), y_va_t).item()
            if vl < best["val_loss"]:
                best = {"val_loss": vl, "state": {k: v.clone() for k, v in head.state_dict().items()}, "epoch": epoch}
            if epoch - best["epoch"] > PATIENCE:
                break
        head.load_state_dict(best["state"])
        head.eval()
        with torch.no_grad():
            p = head(torch.tensor(E_te)).numpy()
        results["mlp"].append((pearson(p, y_te), spearman(p, y_te), float(np.abs(p - y_te).mean())))

        if have_sklearn:
            rg = Ridge(alpha=1.0).fit(E_tr, y_tr)
            rp = rg.predict(E_te)
            results["ridge"].append((pearson(rp, y_te), spearman(rp, y_te), float(np.abs(rp - y_te).mean())))

    for name, rows in results.items():
        if not rows:
            continue
        arr = np.array(rows)
        print(f"\n=== {name.upper()} ({len(rows)} seeds) ===")
        print(f"pearson : {arr[:,0].mean():.3f} +/- {arr[:,0].std():.3f}")
        print(f"spearman: {arr[:,1].mean():.3f} +/- {arr[:,1].std():.3f}")
        print(f"MAE     : {arr[:,2].mean():.3f} +/- {arr[:,2].std():.3f}")


if __name__ == "__main__":
    main()

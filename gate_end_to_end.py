"""
End-to-end recall check: gate (jigo_voice.ANSWER_GATE) + grounded LLM (compose_reply).

For every query in the gate-calibration set, run exactly what web_ui.py runs on a
recall turn (top-8 retrieval -> gate -> compose_reply) and score against ground truth:
  answerable   -> answered AND contains the expected value
  unanswerable -> refused (by the gate or by the LLM saying the REFUSAL string)

Uses the free LLM chain via llm.py; replies are cached in gate_e2e_cache.json so a
quota-interrupted run can resume. Run: python gate_end_to_end.py
"""
import json
import os
import time

import numpy as np

import gate_calibration as G
from jigo_voice import ANSWER_GATE, REFUSAL, compose_reply

A, M = G.A, G.M

# The harness stores memories on a simulated clock; make the answer prompt's
# "[Nd ago]" tags use the same clock (otherwise every memory reads "[today]" and
# the LLM cannot resolve updated facts).
import time as _real_time
import types as _types
import jigo_voice as _jv
_jv.time = _types.SimpleNamespace(time=lambda: A._clock[0], sleep=_real_time.sleep,
                                  perf_counter=_real_time.perf_counter)

CACHE = "gate_e2e_cache.json"
cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}


def main():
    col = G.build_collection()
    rows = ([(q, "answerable", ans) for _, _, q, ans in A.FACTS]
            + [(u[5], "answerable", u[6]) for u in A.UPDATES]
            + [(q, "easy", None) for q in A.ABSTAIN + G.EASY_EXTRA]
            + [(q, "hard", None) for q in G.HARD])
    out = []
    for q, kind, ans in rows:
        res = M.search_memory(q, top_k=8, associative=False, col=col)
        gated = res[0]["raw_similarity"] < ANSWER_GATE
        if gated:
            reply, layer = REFUSAL, "gate"
        else:
            if q not in cache:
                try:
                    cache[q] = compose_reply(q, res) or ""
                except Exception as e:  # quota etc.
                    cache[q] = None
                    print("LLM failed:", str(e)[:60])
                json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                time.sleep(3)
            reply, layer = cache[q], "llm"
        if reply is None:
            continue  # LLM unavailable for this query; excluded, counted below
        refused = reply.strip().rstrip(".") == REFUSAL.rstrip(".")
        out.append((kind, ans, refused, layer, reply, q))

    n_skipped = len(rows) - len(out)
    print(f"scored {len(out)}/{len(rows)} queries ({n_skipped} skipped: LLM unavailable)")
    ans_rows = [r for r in out if r[0] == "answerable"]
    correct = [r for r in ans_rows if not r[2] and r[1].lower() in r[4].lower()]
    print(f"answerable: correct answer {len(correct)}/{len(ans_rows)} | "
          f"refused (wrong) {sum(r[2] for r in ans_rows)}/{len(ans_rows)}")
    for kind in ("easy", "hard"):
        rr = [r for r in out if r[0] == kind]
        by_gate = sum(1 for r in rr if r[2] and r[3] == "gate")
        by_llm = sum(1 for r in rr if r[2] and r[3] == "llm")
        print(f"{kind:5s}: refused {by_gate + by_llm}/{len(rr)} (gate {by_gate}, LLM {by_llm}) | "
              f"answered anyway {len(rr) - by_gate - by_llm}")
    bad = [r for r in out if r[0] != "answerable" and not r[2]]
    for r in bad[:8]:
        print(f"  HALLUCINATED? {r[5]!r} -> {r[4][:80]!r}")


if __name__ == "__main__":
    main()

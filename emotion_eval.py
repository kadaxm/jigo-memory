"""
Emotional-memory evaluation.

Part 1  detector      text_emotion() accuracy on 60 labelled statements (English,
                      Hinglish, Gujarati script; 10 per label incl. neutral).
                      Includes cases written WITHOUT lexicon cue words - misses are expected.
Part 2  query intent  query_feeling() on feeling questions, and false positives on the
                      92 factual/never-stored questions from ablation_memory.py.
Part 3  retrieval     "what has been stressing me out?" style questions share no words with
                      the memories they want. Compare feeling-aware retrieval vs plain
                      similarity+recency, with emotion tags set to the gold labels
                      (i.e. assuming the detector was right; Part 1 gives the detector's
                      own error rate).
Part 4  decay         recency of emotional vs neutral memories at several ages.

Run: python emotion_eval.py
"""
import chromadb
import numpy as np

import ablation_memory as A
import gate_calibration as G
from emotion_text import text_emotion, query_feeling

M = A.M

# ----------------------------------------------------------------- part 1
STATEMENTS = {
    "sad": [
        "Aaj maan kharab chhe, kaik saras nathi lagtu", "I keep thinking about my grandfather and it hurts",
        "I failed the entrance exam and I feel terrible", "Feeling low since the result came out",
        "My dog passed away last night", "I miss my hometown a lot", "Nothing feels worth it these days",
        "I cried after the call with my mom", "આજે મન ખરાબ છે", "I was so disappointed with my marks",
    ],
    "stressed": [
        "I am very worried about tomorrow's viva", "The deadline is killing me, I can't sleep",
        "My heart is racing before the interview", "I'm anxious about the results", "Too much pressure from the team lead",
        "I feel overwhelmed with all this work", "ઇન્ટરવ્યૂની ચિંતા છે", "Mane bahu tension chhe exam nu",
        "I keep panicking before every presentation", "Everything is piling up and I'm burnt out",
    ],
    "angry": [
        "I am so annoyed with my roommate", "That was unfair, they blamed me for their mistake",
        "Mane bahu gussa aave chhe", "I'm furious about the cancelled train", "Why do people have to be so rude",
        "Fed up of this constant lag", "I hate when meetings run over", "He took credit for my work again",
        "ખૂબ ગુસ્સો આવે છે", "This is so frustrating",
    ],
    "happy": [
        "I'm so happy with how the demo went", "Finally fixed the bug in the pipeline", "Had a wonderful evening with family",
        "Aaj bahu maja aavi", "I feel relieved the exams are over", "Proud of my sister for winning",
        "ખુશ છું આજે", "Deployed the dashboard successfully", "Enjoyed the movie night a lot",
        "Got a lovely message from an old friend",
    ],
    "excited": [
        "I'm so excited for the trip next week", "I can't wait for the fest", "Got selected for the internship!",
        "Super thrilled about the new project", "Looking forward to meeting everyone", "Yay, the offer letter arrived",
        "My team got placed, amazing news", "The concert tickets are booked, I'm pumped",
        "Tomorrow is going to be a big day", "આવતા અઠવાડિયે ટ્રિપ પર જવા ઉત્સાહિત છું",
    ],
    "neutral": [
        "My meeting is at three tomorrow", "I take the 7:40 train to college", "The wifi password is in the notes app",
        "I live in Jaipur", "Dr Mehta's clinic is near the temple", "My laptop has a 3050 GPU",
        "I need to buy milk on the way home", "The project report is due Friday", "My mother makes biryani on weekends",
        "I deployed my app on port eight thousand",
    ],
}


def part1():
    labels = list(STATEMENTS)
    conf = {t: {p: 0 for p in labels} for t in labels}
    wrong = []
    for t, items in STATEMENTS.items():
        for s in items:
            p, _ = text_emotion(s)
            conf[t][p] += 1
            if p != t:
                wrong.append((t, p, s))
    n = sum(len(v) for v in STATEMENTS.values())
    correct = sum(conf[l][l] for l in labels)
    print(f"PART 1  detector accuracy: {correct}/{n} = {correct / n:.0%}")
    print(f"{'true \\ pred':12s}" + "".join(f"{l:>9s}" for l in labels))
    for t in labels:
        print(f"{t:12s}" + "".join(f"{conf[t][p]:>9d}" for p in labels))
    emo = [l for l in labels if l != "neutral"]
    tp = sum(1 for t in emo for p in emo if conf[t][p])  # placeholder to keep structure simple
    detected = sum(10 - conf[t]["neutral"] for t in emo)
    print(f"emotional statements detected as emotional: {detected}/50 | neutral false positives: "
          f"{10 - conf['neutral']['neutral']}/10")
    for t, p, s in wrong:
        print(f"   miss: true={t:8s} pred={p:8s} | {s}")


# ----------------------------------------------------------------- part 2
FEELING_QUERIES = [
    ("What has been stressing me out lately?", "stressed"), ("What is worrying me the most?", "stressed"),
    ("What's been bothering me this week?", "stressed"), ("What made me sad recently?", "sad"),
    ("What has been upsetting me?", "sad"), ("What annoyed me this week?", "angry"),
    ("What made me angry lately?", "angry"), ("What made me happy this month?", "happy"),
    ("Any good news from the past weeks?", "happy"), ("What am I excited about?", "excited"),
    ("What am I looking forward to?", "excited"), ("How have I been feeling lately?", "any"),
]


def part2():
    ok = sum(query_feeling(q) == lab for q, lab in FEELING_QUERIES)
    others = [q for _, _, q, _ in A.FACTS] + [u[5] for u in A.UPDATES] + A.ABSTAIN + G.EASY_EXTRA + G.HARD
    fp = [q for q in others if query_feeling(q)]
    print(f"\nPART 2  query_feeling: {ok}/{len(FEELING_QUERIES)} feeling questions routed correctly | "
          f"false positives on {len(others)} factual/off-topic questions: {len(fp)}")
    for q in fp:
        print("   false positive:", q, "->", query_feeling(q))


# ----------------------------------------------------------------- part 3
# (text, age_days, gold_label) - none of these contain an emotion cue word
EMOTIONAL = [
    ("My interview is on Friday", 4, "stressed"), ("The viva is next week", 6, "stressed"),
    ("I have to present to the director on Monday", 3, "stressed"), ("My rent is due and I am short this month", 9, "stressed"),
    ("My grandfather's health report came back", 12, "sad"), ("Rohan moved to another city last week", 8, "sad"),
    ("I did not get the scholarship", 20, "sad"),
    ("My sister's wedding date is fixed", 15, "happy"), ("The demo went well in front of the class", 5, "happy"),
    ("I cleared the driving test", 30, "happy"),
    ("The Goa trip is booked for December", 10, "excited"), ("The fest line-up was announced", 7, "excited"),
    ("My cousin is visiting next month", 14, "excited"),
    ("The landlord kept my deposit", 18, "angry"), ("My project partner skipped the meeting again", 2, "angry"),
]


def build_collection():
    A.configure(A.VARIANTS["V8 prod, rec weight .05"])
    col = chromadb.EphemeralClient().create_collection("emo_eval", metadata={"hnsw:space": "cosine"})
    events = [(age, text, "base", None) for age, text, _, _ in A.build_events()]  # facts + distractors
    ev = [(age, text) for age, text, _, _ in A.build_events()] + [(age, t) for t, age, _ in EMOTIONAL]
    gold = {t: lab for t, _, lab in EMOTIONAL}
    for age, text in sorted(ev, key=lambda e: -e[0]):
        A._clock[0] = A.T0 - age * A.DAY
        mid = M.add_memory(text, source="emo_eval", col=col)
        if text in gold:  # tag as if a perfect detector (voice/LLM) had labelled it
            meta = col.get(ids=[mid])["metadatas"][0]
            meta.update({"emotion": gold[text], "emotion_intensity": 0.7})
            col.update(ids=[mid], metadatas=[meta])
    A._clock[0] = A.T0
    return col


def part3():
    col = build_collection()
    gold_by_text = {t: lab for t, _, lab in EMOTIONAL}
    rows = []
    for q, lab in FEELING_QUERIES:
        feeling = query_feeling(q)
        for name, f in (("plain", None), ("feeling-aware", feeling)):
            res = M.search_memory(q, top_k=3, associative=False, col=col, feeling=f)
            top = [gold_by_text.get(r["content"]) for r in res[:3]]
            want = (lambda g: g is not None) if lab == "any" else (lambda g, lab=lab: g == lab)
            rows.append((name, q, sum(map(want, top)) / 3, any(map(want, top)), res[0]["content"][:48]))
    print("\nPART 3  retrieval on feeling questions (top-3; gold emotion tags assumed)")
    for name in ("plain", "feeling-aware"):
        sel = [r for r in rows if r[0] == name]
        print(f"  {name:14s} precision@3 = {np.mean([r[2] for r in sel]):.2f} | "
              f"at least one right in top-3 = {np.mean([r[3] for r in sel]):.0%}")
    for r in rows:
        if r[0] == "feeling-aware":
            print(f"     {r[1][:42]:44s} p@3={r[2]:.2f}  top1: {r[4]!r}")


# ----------------------------------------------------------------- part 4
def part4():
    print("\nPART 4  recency of a semantic memory (30-day base half-life)")
    print(f"{'age':>6s} {'neutral':>9s} {'intensity .5':>13s} {'intensity 1.0':>14s}")
    now = A.T0
    A._clock[0] = now
    for d in (7, 30, 90, 180):
        ts = now - d * A.DAY
        print(f"{d:>5d}d {M._recency_score(ts, 'semantic', 0.0):>9.2f} "
              f"{M._recency_score(ts, 'semantic', 0.5):>13.2f} {M._recency_score(ts, 'semantic', 1.0):>14.2f}")


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()

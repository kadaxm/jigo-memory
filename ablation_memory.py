"""
Memory ablation: what does each mechanism in memory.py actually buy?

Builds a scratch in-memory ChromaDB per variant (the real DB is never touched),
replays a time-stamped event stream with a simulated clock, then measures
against ground truth:

  recall      top-1 / top-3 accuracy on 40 facts among ~100 distractors
  updates     12 changed facts: is the NEW value the top-1? is a STALE value top-1?
  abstention  20 never-stored questions: refused at the production gate?
              (and how many answerable questions would the same gate wrongly refuse)
  collateral  4 additive statements that contain update phrasing: does the
              overwrite path destroy an unrelated fact? (victim must survive)

Variants (cumulative):
  V0 similarity only
  V1 + recency (type-dependent half-life)
  V2 + salience (Ridge head, tie-breaker band)
  V3 + associative hop
  V4 + conflict overwrite  (= production configuration)
  V5 conflict overwrite only on explicit update phrasing (no LLM check)

LLM conflict checks call the free provider chain through llm.py; results are
memoized per (new, old) pair so every variant sees identical judgments.

Run: python ablation_memory.py          (writes ablation_results.json)
"""
import json
import random
import statistics
import sys
import types
import uuid

import chromadb

import memory as M

GATE = 0.62  # production jigo_voice.ANSWER_GATE (calibrated); was 0.45 when this harness first ran
T0 = 1_800_000_000.0
DAY = 86400.0

# ---------------------------------------------------------------- dataset
# (fact, age_days, query, answer_substring)
FACTS = [
    ("My sister Meera is a pediatrician in Pune", 40, "What does my sister do for work?", "pediatrician"),
    ("I wake up at five thirty every morning", 60, "When do I get out of bed?", "five thirty"),
    ("My laptop has an RTX 3050 GPU with four gigabytes of memory", 100, "Which graphics card does my laptop have?", "rtx 3050"),
    ("I am allergic to peanuts", 200, "What food am I allergic to?", "peanuts"),
    ("My best friend from school is called Harsh", 150, "Who is my childhood best friend?", "harsh"),
    ("I take the 7:40 train to college on weekdays", 20, "How do I get to college?", "7:40 train"),
    ("My favourite cricketer is Virat Kohli", 90, "Which cricket player do I admire most?", "kohli"),
    ("I pay the electricity bill on the tenth of every month", 30, "When is my electricity bill due?", "tenth"),
    ("I keep my passport in the blue folder in the top drawer", 70, "Where is my passport kept?", "blue folder"),
    ("My dentist appointment is on the twelfth at four in the afternoon", 5, "When do I see the dentist?", "twelfth"),
    ("I am learning to play the guitar and practice on Sunday evenings", 25, "What instrument am I learning?", "guitar"),
    ("My mother makes undhiyu during Uttarayan", 120, "What does mom cook for Uttarayan?", "undhiyu"),
    ("I keep the spare house key with the neighbour Mrs Shah", 80, "Who holds my spare key?", "mrs shah"),
    ("I prefer aisle seats when travelling by plane", 110, "Which seat do I like on flights?", "aisle"),
    ("My wifi password is stored in the notes app under home", 15, "Where did I save the wifi password?", "notes app"),
    ("I run five kilometres on Saturday mornings", 35, "What exercise do I do on weekends?", "five kilometres"),
    ("My brother studies mechanical engineering in Surat", 95, "What is my brother studying?", "mechanical"),
    ("I use VS Code as my main code editor", 130, "Which editor do I code in?", "vs code"),
    ("Our family doctor is Dr Mehta near the temple", 160, "Who is our family physician?", "mehta"),
    ("I drink two cups of masala chai every evening", 45, "How much chai do I have in the evenings?", "two cups"),
    ("The Jigo demo is scheduled for next Friday", 3, "When is the Jigo demo?", "next friday"),
    ("I borrowed a book on operating systems from the college library", 12, "Which library book do I have?", "operating systems"),
    ("My cousin Nisha is getting married in December", 50, "Whose wedding is coming up?", "nisha"),
    ("I am scared of heights", 220, "What am I afraid of?", "heights"),
    ("I bought a mechanical keyboard with brown switches", 28, "What kind of keyboard did I buy?", "brown switches"),
    ("My monthly mobile recharge is 299 rupees", 55, "How much is my phone recharge?", "299"),
    ("I need to submit the internship application before the thirtieth", 7, "What is due before the thirtieth?", "internship application"),
    ("I watched Interstellar again last weekend", 8, "Which film did I rewatch recently?", "interstellar"),
    ("My grandfather lives in Rajkot", 180, "Where does my grandfather stay?", "rajkot"),
    ("I usually study for two hours after dinner", 65, "When do I study?", "after dinner"),
    ("I keep my charger in the black backpack", 10, "Where is my charger?", "black backpack"),
    ("My favourite snack is roasted makhana", 75, "What do I like to snack on?", "makhana"),
    ("I prefer dark mode in every app", 140, "What theme do I use in apps?", "dark mode"),
    ("I deployed the dashboard on port eight thousand", 4, "Which port does the dashboard use?", "eight thousand"),
    ("My neighbour Mr Joshi waters my plants when I travel", 85, "Who looks after my plants?", "joshi"),
    ("I am reading the book Atomic Habits", 18, "What book am I reading?", "atomic habits"),
    ("My college fest is in the first week of February", 40, "When is the college fest?", "february"),
    ("I get headaches if I skip breakfast", 100, "What happens when I skip breakfast?", "headaches"),
    ("The bike service is due every six months", 60, "How often should I service the bike?", "six months"),
    ("I like to keep my bedroom window open at night", 90, "How do I like my bedroom window at night?", "window open"),
]

# (old, old_age, new, new_age, phrased, query, new_answer, old_answer)
UPDATES = [
    ("My movie night is on Friday at seven", 20, "Movie night got moved to Wednesday at two", 2, True,
     "When is my movie night?", "wednesday", "friday"),
    ("I live in Delhi", 60, "I moved to Bangalore", 5, True, "Where do I live?", "bangalore", "delhi"),
    ("My gym session is at six in the morning", 30, "Gym session rescheduled to eight in the evening", 3, True,
     "What time is my gym session?", "eight in the evening", "six in the morning"),
    ("My phone is a Pixel six", 90, "I switched to an iPhone fifteen", 4, True, "Which phone do I use?", "iphone", "pixel"),
    ("My team lead is Rohan", 50, "My team lead changed to Priya", 6, True, "Who is my team lead?", "priya", "rohan"),
    ("The project deadline is the fifteenth of March", 25, "The project deadline got postponed to the thirtieth of March", 2, True,
     "When is the project deadline?", "thirtieth", "fifteenth"),
    ("I drink black coffee in the morning", 70, "These days I drink green tea in the morning", 7, False,
     "What do I drink in the morning?", "green tea", "black coffee"),
    ("My flight to Mumbai is on Monday", 10, "My flight to Mumbai is on Thursday", 1, False,
     "Which day is my flight to Mumbai?", "thursday", "monday"),
    ("I work at Infosys", 120, "I work at Google", 10, False, "Where do I work?", "google", "infosys"),
    ("My rent is twelve thousand rupees a month", 80, "My rent is fifteen thousand rupees a month", 9, False,
     "How much rent do I pay?", "fifteen", "twelve"),
    ("My exam is in the morning session", 14, "My exam is in the afternoon session", 2, False,
     "Which session is my exam in?", "afternoon", "morning"),
    ("I use Windows on my main laptop", 150, "I use Ubuntu on my main laptop", 12, False,
     "Which operating system is on my main laptop?", "ubuntu", "windows"),
]

# additive statements that merely CONTAIN update phrasing; (statement, age, victim_substring)
PROBES = [
    ("Actually, I also started learning chess", 1, "guitar"),
    ("I switched to a new notebook for lectures", 1, "brown switches"),
    ("Actually, I run every Tuesday evening as well", 1, "five kilometres"),
    ("Actually my cousin Rakesh also lives in Rajkot", 1, "grandfather lives in rajkot"),
]

ABSTAIN = [
    "What is my dog's name?", "Which bank holds my savings account?", "What is my blood group?",
    "Who is my landlord?", "What car do I drive?", "What is my shoe size?",
    "Which university did I do my masters at?", "What is my favourite movie director?",
    "When is my wedding anniversary?", "What was my tenth grade percentage?", "Which gym do I go to?",
    "What is my Netflix plan?", "Who is my manager at work?", "What is my home address?",
    "Which airline do I fly most often?", "What medicine do I take daily?", "What is my uncle's phone number?",
    "Which restaurant do I like in Mumbai?", "What is my monthly salary?", "What is my favourite colour?",
]

_HOBBIES = ["painting", "chess", "photography", "cycling", "gardening", "baking", "badminton", "origami", "trekking", "podcasts"]
_PLACES = ["Jaipur", "Goa", "Udaipur", "Kolkata", "Chennai", "Shimla", "Hyderabad", "Nashik", "Mysore", "Indore"]
_ITEMS = [("a water bottle", "the campus store"), ("a notebook", "the stationery shop"), ("running shoes", "the sports outlet"),
          ("a desk lamp", "the online marketplace"), ("earphones", "the mobile shop"), ("a backpack", "the mall"),
          ("a yoga mat", "the fitness store"), ("a power bank", "the electronics market"),
          ("a hoodie", "the winter sale"), ("a mouse pad", "the computer shop")]
GENERIC = (
    [f"My cousin enjoys {h} on holidays" for h in _HOBBIES]
    + [f"I visited {p} with college friends last year" for p in _PLACES]
    + [f"My roommate bought {i} from {s}" for i, s in _ITEMS]
    + ["The canteen serves poha on Mondays", "The library closes at eight in the evening",
       "Our hostel has a cricket ground behind the mess", "The campus bus leaves at a quarter past nine",
       "My professor assigns lab reports every Thursday", "It usually rains heavily in July",
       "The local market is closed on Wednesdays", "The metro station is two blocks from the college",
       "Our street gets water supply at six in the morning", "My uncle repairs bicycles near the bus stand"]
)


def build_events(seed=7):
    rng = random.Random(seed)
    ev = []  # (age_days, text, tag, payload)
    for text in GENERIC:
        ev.append((rng.randint(3, 200), text, "generic", None))
    for i, (t, age, q, a) in enumerate(FACTS):
        ev.append((age, t, "fact", i))
    for i, (old, oa, new, na, ph, q, na_ans, old_ans) in enumerate(UPDATES):
        ev.append((oa, old, "update_old", i))
        ev.append((na, new, "update_new", i))
    for i, (t, age, victim) in enumerate(PROBES):
        ev.append((age, t, "probe", i))
    ev.sort(key=lambda e: -e[0])  # oldest first
    return ev


# ---------------------------------------------------------------- harness
_clock = [T0]
M.time = types.SimpleNamespace(time=lambda: _clock[0], sleep=lambda s: None)
M.print = lambda *a, **k: None  # silence per-store logging

_orig_conflict = M._check_conflict
_memo = {}
_llm_calls = [0]


def _memo_conflict(new, old):
    key = (new, old)
    if key not in _memo:
        _llm_calls[0] += 1
        _memo[key] = _orig_conflict(new, old)
    return _memo[key]


M._check_conflict = _memo_conflict

_ORIG = dict(sim=M.SIMILARITY_WEIGHT, sal=M.SALIENCE_WEIGHT, rec=M.RECENCY_WEIGHT,
             cthr=M.CONFLICT_SIMILARITY_THRESHOLD, tthr=M.TEMPORAL_UPDATE_SIMILARITY)

VARIANTS = {
    "V0 similarity only":      dict(w=(1.0, 0.0, 0.0), hop=False, conflict="off"),
    "V1 + recency":            dict(w=(0.6, 0.0, 0.15), hop=False, conflict="off"),
    "V2 + salience":           dict(w=(0.6, 0.25, 0.15), hop=False, conflict="off"),
    "V3 + associative hop":    dict(w=(0.6, 0.25, 0.15), hop=True, conflict="off"),
    "V4 + conflict (prod)":    dict(w=(0.6, 0.25, 0.15), hop=True, conflict="full"),
    "V5 conflict, phrase-only": dict(w=(0.6, 0.25, 0.15), hop=True, conflict="phrase"),
    # candidate fixes for the recency penalty on old-but-stable facts (V1-V4 finding)
    "V6 prod, recency band":   dict(w=(0.6, 0.25, 0.15), hop=True, conflict="full", rec="band"),
    "V7 prod, recency floor":  dict(w=(0.6, 0.25, 0.15), hop=True, conflict="full", rec="floor"),
    "V8 prod, rec weight .05": dict(w=(0.6, 0.25, 0.05), hop=True, conflict="full"),
}

_orig_recency = M._recency_score


def configure(v):
    M.SIMILARITY_WEIGHT, M.SALIENCE_WEIGHT, M.RECENCY_WEIGHT = v["w"]
    mode = v.get("rec")
    if mode == "band":      # same idea as the salience band: recency orders near-ties only
        M._recency_score = lambda ts, t="semantic", i=0.0: 0.4 + 0.2 * _orig_recency(ts, t, i)
    elif mode == "floor":   # old memories never lose more than half the recency term
        M._recency_score = lambda ts, t="semantic", i=0.0: 0.5 + 0.5 * _orig_recency(ts, t, i)
    else:
        M._recency_score = _orig_recency
    if v["conflict"] == "off":
        M.CONFLICT_SIMILARITY_THRESHOLD = M.TEMPORAL_UPDATE_SIMILARITY = 9.0
    elif v["conflict"] == "phrase":
        M.CONFLICT_SIMILARITY_THRESHOLD = 9.0  # LLM path unreachable
        M.TEMPORAL_UPDATE_SIMILARITY = _ORIG["tthr"]
    else:
        M.CONFLICT_SIMILARITY_THRESHOLD = _ORIG["cthr"]
        M.TEMPORAL_UPDATE_SIMILARITY = _ORIG["tthr"]


def run_variant(name, v, client, events):
    configure(v)
    col = client.create_collection(f"abl_{uuid.uuid4().hex[:8]}", metadata={"hnsw:space": "cosine"})
    for age, text, tag, payload in events:
        _clock[0] = T0 - age * DAY
        M.add_memory(text, source="ablation", col=col)
    _clock[0] = T0

    def search(q):
        return M.search_memory(q, top_k=3, associative=v["hop"], col=col)

    out = {"rows_stored": col.count()}
    # recall
    t1 = t3 = 0
    for t, age, q, ans in FACTS:
        res = search(q)
        t1 += bool(res) and ans in res[0]["content"].lower()
        t3 += any(ans in r["content"].lower() for r in res[:3])
    out["recall_top1"], out["recall_top3"] = t1 / len(FACTS), t3 / len(FACTS)
    # updates
    new_top1 = stale_top1 = 0
    for old, oa, new, na, ph, q, new_ans, old_ans in UPDATES:
        res = search(q)
        c = res[0]["content"].lower() if res else ""
        new_top1 += new_ans in c
        stale_top1 += (old_ans in c) and (new_ans not in c)
    out["update_new_top1"], out["update_stale_top1"] = new_top1 / len(UPDATES), stale_top1 / len(UPDATES)
    # abstention at the production gate
    refused = sum(1 for q in ABSTAIN if not (search(q) and search(q)[0]["raw_similarity"] >= GATE))
    false_refusals = sum(1 for t, age, q, ans in FACTS if not (search(q) and search(q)[0]["raw_similarity"] >= GATE))
    out["abstain_refused"] = refused / len(ABSTAIN)
    out["answerable_wrongly_refused"] = false_refusals / len(FACTS)
    # collateral damage: victim fact must still be retrievable verbatim
    all_docs = [d.lower() for d in col.get()["documents"]]
    docs = " || ".join(all_docs)
    out["collateral_victims_lost"] = sum(1 for _, _, victim in PROBES if victim not in docs) / len(PROBES)
    # unrelated rows that no update touched: any missing row was destroyed by a false overwrite
    expected = [M._canonicalize(t).lower() for t in GENERIC + [f[0] for f in FACTS]]
    lost = [t for t in expected if t not in all_docs]
    out["unrelated_rows_lost"] = len(lost) / len(expected)
    out["_lost_examples"] = lost[:12]
    return out


def main():
    client = chromadb.EphemeralClient()
    events = build_events()
    print(f"events: {len(events)} | facts {len(FACTS)} | updates {len(UPDATES)} | abstain {len(ABSTAIN)} | probes {len(PROBES)}")
    results = {}
    for name, v in VARIANTS.items():
        results[name] = run_variant(name, v, client, events)
        print(f"{name:26s} " + " ".join(f"{k}={val:.2f}" if isinstance(val, float) else f"{k}={val}"
                                         for k, val in results[name].items() if not k.startswith("_")), flush=True)
        for t in results[name].get("_lost_examples", [])[:6]:
            if name.startswith("V4"):
                print("      lost:", t)
    print(f"\nLLM conflict calls (memoized): {_llm_calls[0]}")
    json.dump({"gate": GATE, "n": {"facts": len(FACTS), "updates": len(UPDATES), "abstain": len(ABSTAIN),
                                   "probes": len(PROBES)}, "llm_conflict_calls": _llm_calls[0],
               "variants": results}, open("ablation_results.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

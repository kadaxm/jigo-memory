"""
Emotion from WORDS (English / Hinglish / Gujarati script) for emotional memory.

Why text, not voice: emotion.py's prosody heuristic labelled 21/24 clips of
deliberately expressive speech as "neutral" (see README, emotion check), and a
calm speaker saying "mann kharab chhe" is sad regardless of tone. Text is the
primary signal; voice prosody is only used as a weak tie-breaker.

Free and offline: a phrase lexicon with intensifiers and simple negation. It is
deliberately conservative - no match means "neutral". Accuracy on a labelled
statement set is measured in emotion_eval.py; it is the first thing to swap for a
distilled model (same teacher->student recipe as salience_model.py).

Public API
  text_emotion(text)            -> (label, intensity)
  merge_emotion(text, voice)    -> (label, intensity)
  query_feeling(query)          -> label | "any" | None   (is the user asking about feelings?)
"""
import re

LABELS = ("sad", "stressed", "angry", "happy", "excited", "neutral")

_LEX = {
    "stressed": [
        "stressed", "stressing", "stress", "anxious", "anxiety", "worried", "worry", "worrying", "nervous",
        "panic", "overwhelmed", "can't sleep", "cannot sleep", "pressure", "scared", "afraid", "tension",
        "restless", "burnt out", "burnout", "ghabrai", "ghabhray", "ghabrahat", "dar lag", "dar lage",
        "ચિંતા", "ટેન્શન", "ઘબરાહટ", "ડર",
    ],
    "sad": [
        "sad", "depressed", "feeling low", "low feel", "feeling down", "not feeling good", "not feeling well",
        "not feeling", "lonely", "cried", "crying", "cry", "heartbroken", "passed away", "died", "grief",
        "hopeless", "upset", "hurt", "disappointed", "let down", "miss my", "missing my", "maan kharab",
        "man kharab", "mann kharab", "dukh", "dukhi", "udas", "rou chhu", "roy chhu", "barbaad", "kharab chhe",
        "kharab chhu", "saras nathi", "nusasan", "nukshan", "થાકી ગયો", "મન ખરાબ", "ઉદાસ", "દુઃખ", "દુખી",
        "ખરાબ છે", "સરસ નથી", "નથી લાગતું",
    ],
    "angry": [
        "angry", "furious", "annoyed", "irritated", "irritating", "frustrated", "frustrating", "fed up", "hate",
        "pissed", "unfair", "blamed", "rude", "gussa", "gussey", "gusse", "ગુસ્સો", "ગુસ્સે",
    ],
    "excited": [
        "excited", "can't wait", "cannot wait", "thrilled", "pumped", "stoked", "hyped", "looking forward",
        "got selected", "got the job", "got an offer", "got placed", "got the internship", "offer letter",
        "yay", "amazing news", "aavi gayu", "ઉત્સાહિત",
    ],
    "happy": [
        "happy", "glad", "delighted", "so good", "great day", "proud", "relieved", "grateful", "thankful",
        "enjoyed", "loved", "wonderful", "khush", "maja", "mast", "finally fixed", "finally done",
        "finally working", "deployed", "shipped", "launched", "cracked the", "cleared the", "passed the",
        "won the", "it works now", "solved the", "ખુશ", "મજા",
    ],
}
_PRIORITY = ("stressed", "sad", "angry", "excited", "happy")
_INTENSIFIERS = re.compile(
    r"\b(very|so|really|extremely|totally|completely|super|too much|bahu|bahut|khub|ekdum|kaafi)\b|!", re.I)
_NEGATION = re.compile(r"(not|never|no longer|isn't|aren't|wasn't|don't|didn't|nahi|nathi)\s+(\w+\s+){0,2}$", re.I)


def _compile(phrases):
    parts = []
    for p in sorted(phrases, key=len, reverse=True):
        if re.search(r"[઀-૿]", p):
            parts.append(re.escape(p))
        else:
            parts.append(r"(?<!\w)" + re.escape(p) + r"(?!\w)")
    return re.compile("|".join(parts), re.I)


_RX = {k: _compile(v) for k, v in _LEX.items()}


def text_emotion(text):
    """(label, intensity in [0.3, 1.0]); ('neutral', 0.0) when nothing matches."""
    t = " ".join((text or "").split()).lower()
    if not t:
        return "neutral", 0.0
    scores = {}
    for label, rx in _RX.items():
        hits = 0
        for m in rx.finditer(t):
            if label in ("happy", "excited") and _NEGATION.search(t[: m.start()]):
                continue  # "not happy", "never excited"
            hits += 1
        if hits:
            scores[label] = hits
    if not scores:
        return "neutral", 0.0
    best = max(scores.values())
    label = next(l for l in _PRIORITY if scores.get(l) == best)
    inten = 0.45 + 0.15 * (best - 1) + (0.15 if _INTENSIFIERS.search(t) else 0.0)
    return label, round(min(1.0, max(0.3, inten)), 2)


def merge_emotion(text_result, voice_label=None, voice_intensity=None):
    """Text wins; a confident voice reading only fills in when the words are neutral.
    Voice 'soft' is ignored (not an emotion label we store)."""
    t_label, t_int = text_result
    if t_label != "neutral":
        return t_label, t_int
    if voice_label in ("excited", "stressed") and (voice_intensity or 0) >= 0.5:
        return voice_label, round(float(voice_intensity), 2)
    return "neutral", 0.0


# ---- is the user asking about feelings? (query-side) -------------------------
_FEELING_QUERIES = [
    ("stressed", r"stress(ing|ed)? me|worr(y|ying|ied) (me|about)|anxious about|nervous about|bothering me|"
                 r"on my mind|keeping me up|weighing on me"),
    ("sad", r"made me (sad|cry)|makes me (sad|cry)|upset(ting)? me|why was i (sad|upset|down)|"
            r"what (was|is) making me (sad|upset|down)"),
    ("angry", r"(angry|annoyed|irritated|frustrated) me|made me (angry|mad)|what (was|am i) (angry|annoyed|frustrated)"),
    ("happy", r"made me happy|makes me happy|good news|what (was|is) going well|made me proud"),
    ("excited", r"excited about|looking forward to|what am i excited|can'?t wait for"),
    ("any", r"how (have|am|was) i (been )?feeling|how('?s| is| has) my mood|what('?s| has) been going on with me|"
            r"what (have|has) i been (feeling|going through)|my emotional|how have i been"),
]
_FQ = [(l, re.compile(p, re.I)) for l, p in _FEELING_QUERIES]


def query_feeling(query):
    q = " ".join((query or "").split())
    for label, rx in _FQ:
        if rx.search(q):
            return label
    return None

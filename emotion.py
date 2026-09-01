"""Lightweight speech-emotion analysis from raw PCM — pure numpy, no new deps.

Computes prosodic features (energy, pitch level/variability via autocorrelation,
speech-rate proxy) and maps them to a coarse emotional state:
  neutral | excited | stressed | soft
plus a 0-1 intensity. Coarse by design: consistent signal for memory metadata
and reply styling, not a clinical classifier.
"""

import numpy as np

FRAME = 400          # 25ms @ 16kHz
PITCH_MIN, PITCH_MAX = 70, 400


def _frame_rms(x):
    n = len(x) // FRAME
    if n == 0:
        return np.array([0.0])
    return np.array([float(np.sqrt(np.mean(x[i * FRAME:(i + 1) * FRAME] ** 2))) for i in range(n)])


def _frame_pitch(x, rate=16000):
    """Median F0 via autocorrelation on voiced frames; None if too few voiced."""
    f0s = []
    n = len(x) // FRAME
    lag_min, lag_max = rate // PITCH_MAX, rate // PITCH_MIN
    for i in range(n):
        fr = x[i * FRAME:(i + 1) * FRAME].astype(np.float64)
        if np.sqrt(np.mean(fr ** 2)) < 0.02:
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, "full")[FRAME - 1:]
        if lag_max >= len(ac):
            continue
        seg = ac[lag_min:lag_max]
        if seg.max() <= 0 or ac[0] <= 0:
            continue
        f0 = rate / (lag_min + int(np.argmax(seg)))
        if PITCH_MIN <= f0 <= PITCH_MAX:
            f0s.append(f0)
    if len(f0s) < 3:
        return None, None
    return float(np.median(f0s)), float(np.std(f0s))


def _speech_rate(rms, rate=16000):
    """Peaks in the energy envelope per second — a syllable-rate proxy."""
    frame_sec = FRAME / rate
    thr = max(0.02, rms.mean() * 0.6)
    peaks, above = 0, False
    for v in rms:
        if v > thr and not above:
            peaks += 1
            above = True
        elif v <= thr:
            above = False
    dur = len(rms) * frame_sec
    return peaks / dur if dur > 0 else 0.0


def analyze_emotion(pcm_int16, rate=16000):
    x = np.asarray(pcm_int16, dtype=np.float32) / 32768.0
    rms = _frame_rms(x)
    energy = float(rms.mean())
    dyn = float(rms.std())
    f0, f0_std = _frame_pitch(x, rate)
    rate_proxy = _speech_rate(rms, rate)

    feats = {
        "energy": round(energy, 4),
        "dynamics": round(dyn, 4),
        "pitch": round(f0, 1) if f0 else None,
        "pitch_var": round(f0_std, 1) if f0_std else None,
        "speech_rate": round(rate_proxy, 2),
    }

    # normalized drivers (calibrated for typical laptop-mic speech)
    loud = min(1.0, energy / 0.12)
    pitchy = min(1.0, (f0_std or 0.0) / 55.0)
    high = min(1.0, ((f0 or 0.0) - 120.0) / 140.0) if f0 else 0.0
    fast = min(1.0, max(0.0, (rate_proxy - 2.2) / 2.6))
    quiet = min(1.0, max(0.0, (0.05 - energy) / 0.04))

    excitement = 0.45 * pitchy + 0.30 * fast + 0.25 * loud
    stress = 0.55 * pitchy + 0.25 * high + 0.20 * min(1.0, dyn / 0.08) - 0.15 * loud
    softness = quiet * (1.0 - 0.5 * pitchy)

    scores = {"excited": excitement, "stressed": max(0.0, stress), "soft": softness, "neutral": 0.35}
    label = max(scores, key=scores.get)
    intensity = round(min(1.0, max(0.15, scores[label])), 2)

    return {"label": label, "intensity": intensity, "features": feats}

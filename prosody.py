#!/usr/bin/env python3
"""
prosody.py - hearing HOW something was said, not just what.

Until now the face picked its expression from the words. That means it reads
your text, not you: "I'm fine" said flatly and "I'm fine!" said brightly
produce the same face, which is exactly the failure that makes an assistant
feel like a form with a mouth.

This measures the things a listener actually picks up on before parsing a
sentence - how loud, how fast, how high, how much the pitch moves, and which
way it ends - and turns them into a small set of numbers the face and the
prompt can both use.

Everything here is numpy on the 16 kHz mono audio we already captured. No
model, no download, no extra latency worth measuring: the utterance is sitting
in memory anyway, and this reads it once.

The output is deliberately NOT an emotion. Prosody carries arousal reliably
and valence badly - loud and fast is excitement or fury depending on words
that are not in the signal - so this reports energy and lets the words, which
the model already has, supply the sign.
"""
import numpy as np

SR = 16000
FRAME = 0.040            # 40 ms: long enough for two cycles of a low voice
HOP = 0.010              # 10 ms, the usual pitch-tracking hop
F0_MIN, F0_MAX = 70.0, 350.0     # human speech, low male to high female
VOICED_RATIO = 0.30      # autocorrelation peak height that counts as voiced

# Loudness has to be measured against how this person usually sounds, not
# against a constant. Fixed thresholds reported "possibly shouting" for six
# recordings out of six, because this microphone clips - every utterance
# arrives near the top of the scale, so an absolute reading carries no
# information at all. Relative to your own baseline, "loud" means louder than
# you normally are, which is the thing that actually signals feeling.
RMS_HISTORY = 24         # utterances remembered for the baseline
_recent = []

RATE_SLOW = 2.0          # syllables per second
RATE_FAST = 5.5

# Pitch tracking is noisy, and its favourite mistake is the octave error -
# reporting half or double the true pitch for one frame. Left in, those made
# the measured range 20 semitones, nearly two octaves, when ordinary speech
# covers four to eight.
F0_MEDIAN_WIN = 5
F0_JUMP = 7.0            # semitones from the running median before a frame is
                         # thrown away as an octave error


def _frames(x, n, hop):
    if len(x) < n:
        return np.empty((0, n), dtype=np.float32)
    count = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(count)[:, None]
    return x[idx]


def _f0_track(x, sr):
    """Pitch per frame by autocorrelation, 0 where unvoiced.

    Autocorrelation rather than anything cleverer because it needs no
    dependency and is accurate enough for "is this voice higher or lower and
    more or less varied than usual", which is all that is being asked.
    """
    n = int(FRAME * sr)
    hop = int(HOP * sr)
    fr = _frames(x, n, hop)
    if not len(fr):
        return np.zeros(0)
    fr = fr - fr.mean(axis=1, keepdims=True)
    fr = fr * np.hanning(n)[None, :]

    # autocorrelation via FFT, which is the same thing and much faster
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(fr, size, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), size, axis=1)[:, :n]

    lo = max(1, int(sr / F0_MAX))
    hi = min(n - 1, int(sr / F0_MIN))
    if hi <= lo:
        return np.zeros(len(fr))
    zero = ac[:, 0:1]
    zero[zero <= 0] = 1e-9
    band = ac[:, lo:hi] / zero
    peak = band.argmax(axis=1)
    strength = band[np.arange(len(band)), peak]
    f0 = sr / (peak + lo).astype(np.float64)
    f0[strength < VOICED_RATIO] = 0.0
    return _despike(f0)


def _despike(f0):
    """Median filter, then drop frames that leap away from their neighbours.

    Both steps are needed: the median smooths ordinary jitter, and the jump
    test removes the octave errors that survive it - a single frame at double
    pitch widens the measured range by an octave on its own.
    """
    v = f0.copy()
    n = len(v)
    if n < F0_MEDIAN_WIN:
        return v
    half = F0_MEDIAN_WIN // 2
    med = np.zeros(n)
    for i in range(n):
        w = v[max(0, i - half):i + half + 1]
        w = w[w > 0]
        med[i] = np.median(w) if len(w) else 0.0
    keep = (v > 0) & (med > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        semis = np.abs(12.0 * np.log2(np.where(keep, v, 1.0) /
                                      np.where(keep, med, 1.0)))
    v[keep & (semis > F0_JUMP)] = 0.0
    return v


def _syllable_rate(x, sr):
    """Syllables per second, from peaks in the loudness envelope.

    A syllable is a bump of energy, so counting bumps counts syllables closely
    enough to tell hurried from unhurried - which is the only distinction that
    matters here.
    """
    n = int(0.020 * sr)
    hop = int(0.010 * sr)
    fr = _frames(x, n, hop)
    if len(fr) < 5:
        return 0.0
    env = np.sqrt((fr ** 2).mean(axis=1))
    if env.max() <= 0:
        return 0.0
    env = env / env.max()
    # smooth, or every wobble in a vowel counts as its own syllable
    k = np.hanning(7)
    env = np.convolve(env, k / k.sum(), mode="same")
    thr = max(0.15, env.mean() * 0.6)
    peaks = ((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) &
             (env[1:-1] > thr)).sum()
    seconds = len(x) / float(sr)
    return peaks / seconds if seconds > 0.2 else 0.0


def _norm(v, lo, hi):
    if hi <= lo:
        return 0.0
    return float(min(1.0, max(0.0, (v - lo) / (hi - lo))))


def analyse(pcm, sr=SR):
    """Measurements of one utterance. Never raises - a bad reading must not
    be able to stop the conversation."""
    out = {"energy": 0.0, "rate": 0.0, "f0": 0.0, "f0_range": 0.0,
           "rise": 0.0, "arousal": 0.0, "voiced": 0.0, "seconds": 0.0}
    try:
        x = np.asarray(pcm, dtype=np.float32).ravel()
        if len(x) < int(0.15 * sr):
            return out
        out["seconds"] = len(x) / float(sr)
        rms = float(np.sqrt((x ** 2).mean()))
        out["rms"] = rms
        base = np.median(_recent) if len(_recent) >= 4 else None
        if base and base > 1e-6 and rms > 1e-6:
            # in semitone-ish terms: half your usual level reads 0, double
            # reads 1, and your ordinary speaking voice sits in the middle
            out["energy"] = _norm(np.log2(rms / base), -1.0, 1.0)
        else:
            out["energy"] = 0.5          # no baseline yet: assume ordinary
        _recent.append(rms)
        if len(_recent) > RMS_HISTORY:
            del _recent[0]

        rate = _syllable_rate(x, sr)
        out["rate"] = rate

        f0 = _f0_track(x, sr)
        v = f0[f0 > 0]
        # A second pass, against the WHOLE utterance rather than each frame's
        # neighbours. The local filter cannot see a run of octave errors,
        # because within the run every frame agrees with its neighbours - and
        # a run is exactly what a pitch tracker produces when it locks onto
        # the wrong harmonic for a while. One voice does not span two octaves
        # in a sentence, so anything that far from the speaker's own median is
        # the tracker being wrong.
        if len(v) >= 8:
            m = np.median(v)
            if m > 0:
                v = v[np.abs(12.0 * np.log2(v / m)) <= 8.0]
        out["voiced"] = float(len(v)) / max(1, len(f0))
        if len(v) >= 5:
            out["f0"] = float(np.median(v))
            # spread in semitones: an octave is 12, and a lively sentence
            # covers several while a flat one covers barely one
            lo, hi = np.percentile(v, 10), np.percentile(v, 90)
            if lo > 0:
                out["f0_range"] = float(12.0 * np.log2(hi / lo))
            # how the last third moves against the middle - rising endings are
            # questions and uncertainty, falling ones are statements
            tail = v[-max(3, len(v) // 3):]
            mid = v[:max(3, len(v) // 2)]
            if len(tail) and len(mid) and mid.mean() > 0:
                out["rise"] = float(12.0 * np.log2(tail.mean() / mid.mean()))

        # Arousal only. Prosody separates worked-up from calm reliably; it
        # does not separate delighted from furious, and pretending otherwise
        # would put a happy face on an angry sentence.
        out["arousal"] = float(np.clip(
            0.5 * out["energy"] +
            0.3 * _norm(rate, RATE_SLOW, RATE_FAST) +
            0.2 * _norm(out["f0_range"], 1.5, 8.0), 0.0, 1.0))
    except Exception:
        pass
    return out


def describe(p):
    """One short clause for the prompt, or "" when nothing stands out.

    Only strong readings are reported. Telling the model "they sound normal"
    on every turn would spend tokens to say nothing, and invite it to comment
    on your tone constantly, which is unbearable.
    """
    if p.get("seconds", 0) < 0.3 or p.get("voiced", 0) < 0.15:
        return ""
    bits, tail = [], []
    a, e, r = p["arousal"], p["energy"], p["rate"]
    if a > 0.72:
        bits.append("energised and animated")
    elif a < 0.22:
        bits.append("flat and subdued")
    if e > 0.88:
        bits.append("much louder than usual")
    elif e < 0.12:
        bits.append("much quieter than usual, almost a murmur")
    if r > 5.2:
        bits.append("hurried")
    elif 0 < r < 1.8:
        bits.append("unhurried")
    # A clause, not an adjective: appended to "they sound" it produced
    # "they sound ending on a rising note".
    if p["rise"] > 3.0:
        tail.append("Their voice rises at the end, so they may be unsure or "
                    "asking rather than telling.")
    if not bits and not tail:
        return ""
    lead = ("they sound %s. " % ", ".join(bits)) if bits else ""
    return ("\nTHEIR TONE OF VOICE: %s%s Let it colour your reply and your "
            "expression, but do NOT comment on how they sound unless they "
            "ask." % (lead, " ".join(tail)))

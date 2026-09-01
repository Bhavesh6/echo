#!/usr/bin/env python3
"""
sounds.py - the little non-verbal voice.

Cozmo barely speaks English. Almost all of its character arrives as chirps,
trills and warbles, and Anki's audio lead was explicit about why:

    "If you have voice-over for every single thing he does, it totally gets
     old. So the other system I wanted to build is subtlety."
                                    - Ben Gabaldon, Audio Lead, Anki

Their voice "combines human- and animal-inspired syllables with motor sounds
and more robot-like beeps". So these are not pure beeps: every one glides in
pitch, carries vibrato, and has a couple of harmonics on top, because a bare
sine wave reads as a microwave finishing rather than as a creature.

Everything is synthesised - no audio files - so it costs nothing to ship and
can be retuned by changing numbers.
"""
import math

import numpy as np

SAMPLE_RATE = 16000
VOLUME = 0.32           # these punctuate speech; they should never dominate it


def _tone(f0, f1, dur, vib_hz=0.0, vib=0.0, harm=(1.0, 0.30, 0.12),
          attack=0.012, release=None, curve=1.0):
    """One pitched syllable gliding f0 -> f1.

    The glide is exponential because pitch is perceived logarithmically - a
    linear sweep sounds like it slows down at the top. Harmonics and a little
    vibrato are what separate "a creature made a noise" from "a timer went off".
    """
    n = max(1, int(SAMPLE_RATE * dur))
    t = np.linspace(0.0, dur, n, endpoint=False)
    p = (t / dur) ** curve
    freq = f0 * (f1 / f0) ** p
    if vib > 0.0:
        freq = freq * (1.0 + vib * np.sin(2 * np.pi * vib_hz * t))
    phase = np.cumsum(2 * np.pi * freq / SAMPLE_RATE)
    y = np.zeros(n, dtype=np.float64)
    for i, amp in enumerate(harm, start=1):
        y += amp * np.sin(phase * i)
    y /= sum(harm)

    rel = release if release is not None else min(0.12, dur * 0.5)
    env = np.ones(n)
    a = max(1, int(SAMPLE_RATE * attack))
    r = max(1, int(SAMPLE_RATE * rel))
    env[:a] = np.linspace(0.0, 1.0, a)
    env[-r:] = np.linspace(1.0, 0.0, r) ** 1.6
    return (y * env).astype(np.float32)


def _silence(dur):
    return np.zeros(max(1, int(SAMPLE_RATE * dur)), dtype=np.float32)


def _seq(*parts):
    return np.concatenate([p for p in parts if len(p)]).astype(np.float32)


def _build():
    """Each sound is a short phrase, not a single beep - two or three syllables
    read as intent where one reads as a notification."""
    s = {}

    # heard you / woke up: a bright rising question-hook
    s["wake"] = _seq(_tone(430, 760, 0.085, 9, 0.02),
                     _silence(0.015),
                     _tone(700, 980, 0.075, 9, 0.02))

    # small acknowledgement - "mm-hm", used where a whole sentence is too much
    s["ack"] = _tone(520, 610, 0.075, 12, 0.03, attack=0.008)

    # did not understand: up, then a puzzled sag
    s["confused"] = _seq(_tone(520, 700, 0.10, 11, 0.03),
                         _tone(700, 480, 0.16, 7, 0.05, curve=0.8))

    # pleased: a quick three-note trill
    s["happy"] = _seq(_tone(600, 660, 0.06, 14, 0.02),
                      _tone(760, 820, 0.06, 14, 0.02),
                      _tone(930, 1010, 0.10, 12, 0.03))

    # delighted: same idea, further and brighter
    s["delight"] = _seq(_tone(560, 640, 0.055, 15, 0.02),
                        _tone(740, 820, 0.055, 15, 0.02),
                        _tone(960, 1060, 0.055, 15, 0.02),
                        _tone(1150, 1290, 0.13, 13, 0.03))

    # disappointed: a slow fall
    s["sad"] = _seq(_tone(520, 300, 0.28, 5, 0.04, curve=0.85, release=0.16))

    # startled: a hard blip, no glide to speak of
    s["startle"] = _seq(_tone(880, 1180, 0.05, 0, 0.0, attack=0.003),
                        _silence(0.02),
                        _tone(1080, 900, 0.07, 0, 0.0, attack=0.003))

    # annoyed: a low flat buzz, deliberately unmusical
    s["annoyed"] = _tone(240, 210, 0.18, 22, 0.06,
                         harm=(1.0, 0.55, 0.40, 0.25), attack=0.006)

    # going to sleep: a long yawn downward
    s["sleep"] = _tone(430, 190, 0.55, 3.5, 0.05, curve=0.7, release=0.30)

    # waking from sleep: the same, reversed and shorter
    s["waking"] = _tone(240, 520, 0.32, 5, 0.04, curve=1.3)

    # thinking: two quiet ticks, the sound of something working
    s["thinking"] = _seq(_tone(430, 450, 0.045, 0, 0.0, attack=0.004),
                         _silence(0.10),
                         _tone(470, 490, 0.045, 0, 0.0, attack=0.004))

    for k in s:
        peak = float(np.abs(s[k]).max()) or 1.0
        s[k] = (s[k] * (VOLUME / peak)).astype(np.float32)
    return s


_CACHE = None


def get(name):
    """The PCM for one sound, or None. Built once, on first use."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _build()
    return _CACHE.get(name)


def names():
    global _CACHE
    if _CACHE is None:
        _CACHE = _build()
    return sorted(_CACHE)

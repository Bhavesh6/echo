#!/usr/bin/env python3
"""
face.py - the companion's face. Two eyes, nothing else.

Rendering follows the reverse-engineered Anki Cozmo/Vector model (Vector
Technical Reference Manual, ch.106): each eye is a rounded rect with curved
eyelid/cheek carves, shaded by a radial "hotspot" gradient whose centre can
move - that hotspot is what sells "it is looking at something".

Behaviour follows the same manual's ch.119/121:

  MODE x EMOTION are ORTHOGONAL.
      mode    = what I am doing  (sleeping/idle/listening/thinking/speaking)
                -> drives behaviour: blink rate, gaze style, motion
      emotion = how I feel       (neutral/happy/sad/...)
                -> drives shape
  A `listening + happy` face differs from `listening + worried`. Vector kept
  these apart too: "Where the eyes are looking is controlled within the
  procedural face manager, rather than in the animation files."

  STIMULATION is a decaying scalar, not an event. Low stimulation = chill and
  observing; the wake word spikes it. It gates how busy idle behaviour is and
  how long before the face gets sleepy.

The face NEVER decides its expression from content. Drivers call set_mode /
set_emotion / event / set_amplitude. Keyboard drives it today; the voice loop
drives it later through the same calls.
"""
import math
import random

import pygame

# ----------------------------------------------------------------- look
BG_COLOR = (8, 10, 16)
EYE_COLOR = (30, 205, 255)
FPS = 60

BLINK_DURATION = 0.16
BLINK_MIN = 0.07          # eye height fraction at fully closed
BLINK_SQUASH = 0.18       # width gained while closing (squash & stretch).
                          # Kept modest: these eyes are already wide, and a
                          # big squash drove them into each other mid-blink.

# Radial hotspot (the "pupil"). Vector shades each eye from a movable centre;
# moving it reads as looking around WITHOUT moving the eye itself.
HOTSPOT_FALLOFF = 0.20    # 0 = flat, 1 = very dark at the rim. Kept subtle
                          # now that the vertical gradient does the shading.
HOTSPOT_RANGE = 0.22      # hotspot travel, as a fraction of eye size
# The gradient surface is much larger than the eye (so it still covers it when
# the hotspot is pushed to one side), which means the eye only samples the
# innermost slice of it. Complete the falloff within that slice, or the eye
# renders almost flat.
HOTSPOT_EDGE = 0.42

# Vector's display has a scanline opacity factor - a subtle CRT feel. Set
# SCANLINE_ALPHA = 0 for the clean, modern look (LOOI has no scanlines).
SCANLINE_ALPHA = 0       # 0 = off. LOOI has none; they read as cheap here.
SCANLINE_STEP = 3

# ------------------------------------------------------------ proportions
# Fractions of the window. These are the numbers that decide whether the face
# reads as cute or as blocky, so they are knobs rather than magic numbers.
EYE_W_FRAC = 0.28        # eye width  - wide and short, WALL-E proportions
EYE_H_FRAC = 0.34        # eye height
EYE_GAP_FRAC = 0.20      # half the distance between eye centres AT REST.
MIN_EYE_CLEARANCE = 0.022  # eyes push apart rather than overlap when they grow
                         # (surprised, mid-blink). Measured collision without
                         # this: the two eyes crossed the centre line.

# Eye outline is a SUPERELLIPSE, |x/a|^n + |y/b|^n = 1, not a rounded rect.
# A rounded rect caps its radius at half the shorter side, so a wide eye gets
# flat vertical walls and reads as a pill. A superellipse stays a soft blob at
# any aspect ratio. n=2 is a pure ellipse, n=4 is a squircle, n -> inf is a
# rectangle.
EYE_SUPER = 2.7
SHAPE_STEPS = 64         # points around the outline
SUPERSAMPLE = 2          # render each eye at this scale, then shrink: cheap
                         # antialiasing, and it softens the lid cuts too

# Vertical gradient down the eye, LOOI-style: a bright top edge falling to a
# deeper bottom. Multipliers on the eye colour. Set both to 1.0 to disable.
GRAD_TOP = 1.18
GRAD_BOTTOM = 0.62

# ------------------------------------------------------- emotions (SHAPE)
# ws/hs   width/height scale     oy      vertical look offset (+ = down)
# top_in  top lid coverage at the INNER corner (0..1 of height)
# top_out top lid coverage at the OUTER corner
# bottom  bottom lid/cheek coverage        bend  extra mid-cheek arc
#
# Direction rules, confirmed against pycozmo + RoboEyes + the Cozmo face sheet:
#   bottom carve, symmetric   -> happy dome sitting high
#   top_out > top_in          -> sad / tired  (OUTER corner droops)
#   top_in  > top_out         -> angry / annoyed (INNER corner droops)
#   large SYMMETRIC top carve -> bored, which is not an emotion signal; avoid
EMOTIONS = {
    # neutral idles at ~0.86 so surprise/listening have headroom to GROW.
    # Real Cozmo's neutral runs at 0.8 for exactly this reason.
    "neutral":   dict(ws=0.86, hs=0.86, oy=0.00, top_in=0.05, top_out=0.05, bottom=0.05, bend=0.00, tbend=0.00, ang=0.0),
    "happy":     dict(ws=1.02, hs=0.92, oy=-0.10, top_in=0.00, top_out=0.00, bottom=0.42, bend=0.22, tbend=0.00, ang=0.0),
    "sad":       dict(ws=0.94, hs=0.84, oy=0.12, top_in=0.12, top_out=0.12, bottom=0.04, bend=0.00, tbend=0.07, ang=13.0),
    "curious":   dict(ws=0.95, hs=1.00, oy=-0.07, top_in=0.03, top_out=0.03, bottom=0.03, bend=0.00, tbend=0.00, ang=0.0),
    "surprised": dict(ws=1.16, hs=1.22, oy=-0.05, top_in=0.00, top_out=0.00, bottom=0.00, bend=0.00, tbend=0.00, ang=0.0),
    "annoyed":   dict(ws=0.96, hs=0.78, oy=0.02, top_in=0.20, top_out=0.20, bottom=0.04, bend=0.00, tbend=0.05, ang=-15.0),
    "sleepy":    dict(ws=0.92, hs=0.72, oy=0.10, top_in=0.32, top_out=0.32, bottom=0.14, bend=0.00, tbend=0.08, ang=8.0),
    # "aww" / pleading. Shares sad's outer-corner droop, but the eyes stay BIG
    # and look UP instead of shrinking and looking down.
    "pleading":  dict(ws=1.05, hs=1.05, oy=-0.06, top_in=0.06, top_out=0.06, bottom=0.02, bend=0.00, tbend=0.06, ang=11.0),
    # scared. Shares sleepy's slight outer droop, but WIDE instead of shut.
    "scared":    dict(ws=1.10, hs=1.16, oy=-0.02, top_in=0.03, top_out=0.03, bottom=0.00, bend=0.00, tbend=0.04, ang=6.0),
}
SHAPE_KEYS = ("ws", "hs", "oy", "top_in", "top_out", "bottom", "bend", "tbend", "ang")

# --------------------------------------------------- emotions (MOTION)
# A shape alone is ambiguous. sad and pleading share a silhouette; sleepy and
# scared share a narrowed one. What separates them is how the eyes MOVE:
#
#   sad       slow, heavy, looking DOWN, long blinks, shallow breathing
#   pleading  same droop but looking UP, held wide, with a small fast quiver
#   sleepy    very slow drift, very long blinks, no tremor
#   scared    frozen wide stare, rare blinks, fast tremor, darting gaze
#
# Same reason RoboEyes signals confusion with a motion flick rather than a brow
# shape: motion disambiguates where a static pose cannot.
#
# blink_mul   scales the mode blink interval (>1 = blinks less often)
# gaze_x/y    constant pull on the gaze target (y+ = downcast, y- = upturned)
# tremor      jitter amplitude, as a fraction of eye size
# tremor_hz   jitter frequency
# drift       scales saccade rate and size (>1 = restless, <1 = still)
# breathe     scales idle breathing amplitude
# breathe_hz  scales idle breathing rate
#
# freq/zeta   the SPRING that morphs INTO this emotion. freq is how fast it
#             gets there; zeta is damping - below 1.0 it overshoots and settles
#             back. This is what makes surprise SNAP wide and rebound while
#             sadness sinks slowly with no bounce. A single global lerp rate
#             gives every emotion the same timing, which reads as lifeless.
# antic       anticipation: how far the eyes move the WRONG way first, as a
#             fraction of the move about to happen. Sells weight and intent.
# blink_dur   scales blink duration (>1 = long heavy lids, <1 = snappy)
MOTION_KEYS = ("blink_mul", "blink_dur", "gaze_x", "gaze_y", "tremor", "tremor_hz",
               "drift", "breathe", "breathe_hz", "freq", "zeta", "antic")
EMOTION_MOTION = {
    "neutral":   dict(blink_dur=1.00, antic=0.10, blink_mul=1.00, gaze_x=0.0, gaze_y=0.00, tremor=0.000, tremor_hz=0.0,
                      drift=1.00, breathe=1.00, breathe_hz=1.00, freq=2.20, zeta=1.00),
    "happy":     dict(blink_dur=0.95, antic=0.18, blink_mul=0.85, gaze_x=0.0, gaze_y=-0.10, tremor=0.000, tremor_hz=0.0,
                      drift=1.30, breathe=1.45, breathe_hz=1.30, freq=3.10, zeta=0.62),
    "sad":       dict(blink_dur=1.55, antic=0.00, blink_mul=1.90, gaze_x=0.0, gaze_y=0.45, tremor=0.000, tremor_hz=0.0,
                      drift=0.45, breathe=0.55, breathe_hz=0.60, freq=1.05, zeta=1.15),
    "pleading":  dict(blink_dur=1.25, antic=0.05, blink_mul=1.35, gaze_x=0.0, gaze_y=-0.38, tremor=0.010, tremor_hz=7.5,
                      drift=0.65, breathe=1.15, breathe_hz=1.10, freq=1.70, zeta=0.85),
    "curious":   dict(blink_dur=0.90, antic=0.25, blink_mul=0.90, gaze_x=0.0, gaze_y=-0.12, tremor=0.000, tremor_hz=0.0,
                      drift=1.45, breathe=1.10, breathe_hz=1.15, freq=3.30, zeta=0.70),
    # the one the whole spring model exists for: normal -> WIDE, fast, with a
    # visible rebound past the target before it settles.
    "surprised": dict(blink_dur=0.70, antic=0.38, blink_mul=1.60, gaze_x=0.0, gaze_y=-0.05, tremor=0.004, tremor_hz=11.0,
                      drift=0.55, breathe=1.30, breathe_hz=1.60, freq=5.60, zeta=0.40),
    "scared":    dict(blink_dur=0.60, antic=0.30, blink_mul=2.20, gaze_x=0.0, gaze_y=-0.04, tremor=0.026, tremor_hz=15.0,
                      drift=2.40, breathe=1.70, breathe_hz=2.10, freq=6.40, zeta=0.50),
    "annoyed":   dict(blink_dur=1.05, antic=0.12, blink_mul=1.10, gaze_x=0.0, gaze_y=0.05, tremor=0.000, tremor_hz=0.0,
                      drift=0.70, breathe=0.85, breathe_hz=0.90, freq=2.40, zeta=1.10),
    "sleepy":    dict(blink_dur=2.20, antic=0.00, blink_mul=2.60, gaze_x=0.0, gaze_y=0.30, tremor=0.000, tremor_hz=0.0,
                      drift=0.30, breathe=0.50, breathe_hz=0.45, freq=0.80, zeta=1.20),
}


def spring_step(cur, vel, target, freq, zeta, dt):
    """One step of a damped spring. zeta < 1 overshoots, zeta >= 1 does not.
    Semi-implicit Euler, clamped so a long frame cannot blow it up."""
    dt = min(dt, 1.0 / 30.0)
    w = 2.0 * math.pi * freq
    vel += (w * w * (target - cur) - 2.0 * zeta * w * vel) * dt
    return cur + vel * dt, vel

# ------------------------------------------------------- modes (BEHAVIOUR)
# The shape contribution is a MODIFIER on top of the emotion, never a
# replacement. top_add applies equally to top_in and top_out, so a mode can
# never flip an emotion's asymmetry direction - "listening" cannot accidentally
# turn a sad face angry. That invariant is the whole point of the split.
#
# blink    : (min, max) seconds between blinks, or None for no auto-blink
# gaze     : wander | lock | avert | still
# amp_gain : how strongly live audio amplitude modulates eye height
MODES = {
    "sleeping":  dict(ws_mul=1.00, hs_mul=0.07, oy_add=0.24, top_add=0.00, bottom_add=0.00,
                      blink=None, gaze="still", amp_gain=0.00),
    "idle":      dict(ws_mul=1.00, hs_mul=1.00, oy_add=0.00, top_add=0.00, bottom_add=0.00,
                      blink=(2.8, 6.5), gaze="wander", amp_gain=0.03),
    # attention: eyes GROW and blink LESS. Both are real cues - people blink
    # less while concentrating, and a slow blink on WIDE eyes reads as
    # attentive rather than sleepy.
    "listening": dict(ws_mul=1.16, hs_mul=1.20, oy_add=-0.03, top_add=-0.03, bottom_add=-0.02,
                      blink=(5.0, 9.5), gaze="lock", amp_gain=0.10),
    # thinking looks AWAY (gaze aversion): symmetric squint + averted glance.
    # Never asymmetric brows here - that reads as anger.
    "thinking":  dict(ws_mul=0.97, hs_mul=0.88, oy_add=-0.05, top_add=0.06, bottom_add=0.00,
                      blink=(3.5, 7.0), gaze="avert", amp_gain=0.00),
    "speaking":  dict(ws_mul=1.06, hs_mul=1.04, oy_add=-0.01, top_add=0.00, bottom_add=0.00,
                      blink=(3.0, 6.0), gaze="lock", amp_gain=0.20),
}
DEFAULT_MODE, DEFAULT_EMOTION = "idle", "neutral"

# --------------------------------------------------- emotion events
# Vector's mood manager maps an event NAME (just a string) to an "affector":
# which dimensions move, by how much, and how it decays. Events are data, not
# code - same idea here. `emotion` is a suggested expression; `gesture` is a
# one-shot "get in" motion (every Vector behaviour had a get-in animation).
EVENTS = {
    # --- voice loop. Vector's equivalents are ReactToTriggerWord /
    #     NoValidVoiceIntent / RespondToShortVoiceCommand.
    "wake_word":      dict(stim=+0.55, social=+0.35, happy=+0.03, emotion="surprised", gesture="perk"),
    "speech_start":   dict(stim=+0.15, social=+0.10),
    "speech_end":     dict(stim=+0.05),
    "understood":     dict(stim=+0.10, confident=+0.12),
    # "I heard you but did not get it" deserves its own reaction. Most hobby
    # assistants have nothing here, and it is the moment they feel most robotic.
    "not_understood": dict(stim=+0.05, confident=-0.18, emotion="curious", gesture="tilt", symbol="question"),
    # --- social. Vector had a separately authored behaviour for each of these.
    "greeting":       dict(stim=+0.40, social=+0.30, happy=+0.20, emotion="happy", gesture="perk"),
    "compliment":     dict(stim=+0.20, social=+0.25, happy=+0.45, emotion="happy", gesture="bounce", symbol="sparkle"),
    "affection":      dict(stim=+0.25, social=+0.35, happy=+0.55, emotion="happy", gesture="bounce", symbol="heart"),
    "apology":        dict(social=+0.15, happy=+0.20, emotion="happy"),
    "abuse":          dict(stim=+0.20, happy=-0.45, confident=-0.20, emotion="sad", symbol="tear"),
    "startled":       dict(stim=+0.75, confident=-0.25, emotion="scared", gesture="perk", symbol="exclaim"),
    "want":           dict(social=+0.30, happy=+0.10, emotion="pleading"),
    "goodbye":        dict(social=-0.20, happy=-0.10, emotion="sad"),
    "long_absence":   dict(stim=+0.80, social=+0.50, happy=+0.45, emotion="happy", gesture="bounce", symbol="heart"),
    # --- ambient
    "sound":          dict(stim=+0.18),
    "touch":          dict(stim=+0.30, social=+0.20, happy=+0.25, emotion="happy", symbol="heart"),
}

# Dimensions decay toward a baseline. Vector uses a per-event XY decay graph;
# one half-life per dimension is the same idea, coarser.
DIM_HALFLIFE = dict(stim=22.0, social=90.0, happy=120.0, confident=150.0)
DIM_BASELINE = dict(stim=0.05, social=0.25, happy=0.35, confident=0.55)

# Repetition penalty: the same event fired again soon has less effect, so
# spamming the wake word cannot peg the face at maximum surprise forever.
REPEAT_WINDOW = 12.0
REPEAT_FACTOR = 0.45

# Suggested emotions fade back to neutral rather than snapping - an instant
# reset is one of the strongest "this is a machine" tells.
EMOTION_HOLD = 7.0
EMOTION_FADE = 3.0

# --------------------------------------------------------- idle behaviour
# "When Vector's mind isn't doing anything and his mind is blank... he'll
# probably pick exploring, observing, or sleeping as his next activity."
# No wheels here, so 'exploring' becomes a wider, slower, further-away drift.
IDLE_ACTIVITIES = ("observing", "drifting", "dozing")
ACTIVITY_HOLD = (6.0, 14.0)

# Sleep debt: Vector gets sleepier the longer he has been awake regardless of
# interaction, and sleeps it off. He will not sleep while stimulated.
SLEEP_RISE = 1.0 / 900.0       # awake -> fully sleepy in ~15 min
SLEEP_FALL = 1.0 / 240.0       # sleeps it off ~4x faster
SLEEP_ENTER = 0.80
SLEEP_STIM_GATE = 0.16

# Asleep, breathing slows right down and deepens - the clearest non-verbal cue
# that something is sleeping rather than just sitting still.
SLEEP_BREATHE_RATE = 0.35     # of the waking rate
SLEEP_BREATHE_DEPTH = 2.2     # of the waking amplitude

# ----------------------------------------------------------- decoration
# Manga iconography ("manpu") - the settled visual shorthand for emotion that
# every companion robot borrows. Drawn as strokes: no fonts, no assets, scales
# with the window.
SYMBOL_LIFE = 2.6             # seconds to rise and fade
SYMBOL_RISE = 0.26            # of window height
SYMBOL_DRIFT = 0.09           # sideways wander, of window width
SYMBOL_SIZE = 0.055           # of window height at birth
SYMBOL_GROW = 1.7             # size multiplier by the end
SYMBOL_EVERY = (2.2, 4.2)     # gap between symbols while one is called for

# Which symbol belongs to which feeling. None = show nothing, which matters:
# a face decorated at all times is a face saying nothing.
EMOTION_SYMBOL = {
    "neutral": None,
    "happy": "sparkle",
    "sad": "tear",
    "curious": "question",
    "surprised": "exclaim",
    "annoyed": "vein",
    "sleepy": "zzz",
    "pleading": "heart",
    "scared": "sweat",
}

SYMBOL_COLOR = {
    "heart": (255, 110, 150),
    "sparkle": (255, 240, 170),
    "sweat": (150, 220, 255),
    "tear": (150, 220, 255),
    "vein": (255, 90, 90),
    "question": (255, 240, 170),
    "exclaim": (255, 240, 170),
    "note": (190, 230, 255),
    "zzz": EYE_COLOR,
}

# Blush. Manga uses pink ovals on the cheeks for "happiness and cheerfulness",
# not romance. Here its opacity follows the happy dimension, so warmth is
# ambient and earned rather than an event that fires.
BLUSH_COLOR = (255, 120, 165)
BLUSH_ALPHA = 118             # at full happiness
BLUSH_W = 0.10                # of window width
BLUSH_H = 0.058               # of window height - rounder than a flat smear
BLUSH_DROP = 0.165            # below the eye centre. Sitting far below reads
                              # as a stain on the screen rather than cheeks.
BLUSH_OUT = 0.35              # pushed outward from each eye, as cheeks are
MIN_SLEEP = 30.0               # never wake purely on debt sooner than this

# gaze
LOOK_LERP = 0.05
LOOK_RANGE_X = 0.34      # how far a full look actually travels. Was 0.55,
                         # which let idle drift far enough to look aimless.
LOOK_RANGE_Y = 0.26
# Saccades, per Disney Research: the eyes flick between three points forming a
# triangle on the face being looked at - their two eyes and their nose - at
# random intervals of 0.1 to 0.5 s. Small, fast and structured, which is what
# separates "alive" from "wandering aimlessly".
SACCADE_EVERY = (0.10, 0.50)
SACCADE_MAG = 0.075          # how wide that triangle is, in look units
SACCADE_POINTS = ((-1.0, -0.35),   # their left eye
                  (1.0, -0.35),    # their right eye
                  (0.0, 0.75))     # their nose
SACCADE_LERP = 0.55          # flick speed: a saccade is near-instant

# Overlapping action: the pupil reaches the new position before the eye body
# does, the way the eyes lead the head. Cheap, and it stops movement looking
# like one rigid slab.
HOTSPOT_LEAD = 2.4

# ---- gaze aversion, from Andrist et al. (HRI 2014), who coded real human
# conversation. (mean, sd) in seconds. Cognitive aversions go UP; the social
# ones go mostly sideways and are much smaller.
AVERSION = {
    "cognitive": dict(length=(3.54, 1.26), gap=(2.0, 0.6), up=0.75, scale=1.00),
    "listening": dict(length=(1.14, 0.27), gap=(7.21, 1.88), up=0.25, scale=0.40),
    "speaking":  dict(length=(1.96, 0.32), gap=(4.75, 1.39), up=0.25, scale=0.40),
}
AVERT_H = 0.62          # sideways magnitude in look units (~28 deg equivalent)
AVERT_V = 0.44          # vertical (~20 deg)

# A gaze shift eases in AND out. Exponential smoothing only eases out, so every
# movement begins at full speed, which reads as a twitch.
SHIFT_TIME = 0.42

# Idle drift is band-limited noise, never a jump. Disney used Perlin noise for
# this exact purpose; one-dimensional value noise is the same idea and needs no
# dependency.
DRIFT_RATE = 0.30       # steps per second through the noise field, i.e. one
                        # new noise value about every 3 s. Traversing it fast
                        # turns smooth noise back into jitter.
DRIFT_AMOUNT = 0.16     # how far it wanders, in look units

# one-shot "get in" gestures
GESTURES = {
    "perk":   dict(dur=0.45, kind="perk"),
    "tilt":   dict(dur=0.90, kind="tilt"),
    "bounce": dict(dur=0.60, kind="bounce"),
    "shake":  dict(dur=0.50, kind="shake"),
}
SHAKE_CYCLES, SHAKE_AMPLITUDE = 3, 0.14
PERK_OVERSHOOT, BOUNCE_HEIGHT, TILT_MAX = 0.16, 0.10, 0.055

# Anticipation: hold a counter-pose this long before releasing the real move.
# ~4 frames at 60fps, matching the 2-3 frames of anticipation that animation
# practice recommends ahead of a snap (plus the frame or two the spring needs
# to get there).
ANTICIPATION_TIME = 0.07

# Asymmetry: the eye nearer the look direction grows, the far one shrinks -
# real Cozmo does this on every saccade. Plus a permanent hair of difference
# between the two eyes, because perfectly identical eyes read as stiff.
# Subtle. At 0.09 the two eyes were visibly different sizes, which reads as a
# mistake rather than as one being nearer.
CURIOUS_GAIN = 0.045
EYE_IMBALANCE = 0.992

LID_CURVE_STEPS = 16


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


class _Noise:
    """One-dimensional value noise: a smooth, band-limited pseudorandom signal.
    Unlike picking a new random target every few seconds it never jumps, and
    unlike a sine wave it never repeats obviously - which is what makes idle
    motion read as alive rather than as either twitchy or mechanical."""

    def __init__(self, seed, n=512):
        rng = random.Random(seed)
        self.v = [rng.uniform(-1.0, 1.0) for _ in range(n)]

    def __call__(self, t):
        n = len(self.v)
        x = (t % n + n) % n
        i = int(x)
        f = x - i
        a, b = self.v[i % n], self.v[(i + 1) % n]
        return a + (b - a) * (f * f * (3 - 2 * f))      # smoothstep


def ease(p):
    """Slow, fast, slow - the velocity profile of a real gaze shift."""
    p = clamp(p)
    return p * p * (3 - 2 * p)


def smooth(cur, target, rate, dt):
    """Exponential approach that does not care about frame rate. `rate` is the
    old per-frame lerp factor, i.e. what it would do in one frame at 60 fps.

    A plain per-frame lerp silently assumes every frame is 1/60 s, so any hitch
    changes how fast things move - which is felt as jerkiness even when no
    frames are actually dropped."""
    k = 1.0 - (1.0 - rate) ** max(0.0, dt * 60.0)
    return cur + (target - cur) * k


def _lid_curve(left, right, edge_y, h, cov_left, cov_right, from_top, bend=0.0):
    """Smoothstep curve for one eyelid, sweeping cov_left -> cov_right.
    `bend` adds coverage in the middle only (zero at both edges), arcing the
    lid into the eye - pycozmo's lid bend, used by happy's smile cut."""
    pts = []
    sign = 1 if from_top else -1
    for i in range(LID_CURVE_STEPS + 1):
        xf = i / LID_CURVE_STEPS
        ease = xf * xf * (3 - 2 * xf)
        cov = lerp(cov_left, cov_right, ease) + bend * math.sin(xf * math.pi)
        pts.append((lerp(left, right, xf), edge_y + sign * h * cov))
    return pts


def _superellipse(cx, cy, a, b, n, steps=SHAPE_STEPS):
    """Outline points of |x/a|^n + |y/b|^n = 1, centred at (cx, cy)."""
    pts = []
    e = 2.0 / n
    for i in range(steps):
        t = 2.0 * math.pi * i / steps
        ct, cs = math.cos(t), math.sin(t)
        pts.append((cx + a * math.copysign(abs(ct) ** e, ct),
                    cy + b * math.copysign(abs(cs) ** e, cs)))
    return pts


def _build_hotspot(size):
    """Radial gradient, white at the centre falling to grey at the rim. Blitted
    over an eye with BLEND_RGBA_MULT it becomes soft spherical shading. Built
    from many concentric circles so there is no visible banding, and rebuilt
    only on resize."""
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 255))
    steps = 110
    c = size // 2
    for i in range(steps, 0, -1):
        f = i / steps
        k = min(1.0, f / HOTSPOT_EDGE)
        v = int(255 * (1.0 - HOTSPOT_FALLOFF * (k ** 1.7)))
        pygame.draw.circle(surf, (v, v, v, 255), (c, c), max(1, int(c * f)))
    return surf


def _build_vgrad(steps=256):
    """1px-wide vertical gradient, scaled to the eye each frame. Cheap, and it
    stays fixed to the eye rather than following the gaze the way the radial
    hotspot does - which is what LOOI's shading looks like."""
    surf = pygame.Surface((1, steps), pygame.SRCALPHA)
    for y in range(steps):
        f = y / (steps - 1.0)
        v = int(max(0, min(255, 255 * lerp(GRAD_TOP, GRAD_BOTTOM, f))))
        surf.set_at((0, y), (v, v, v, 255))
    return surf


def _build_blush(w, h):
    """A soft-edged oval. Concentric ellipses rather than a flat shape, or it
    reads as a sticker stuck on the face."""
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    steps = 22
    for i in range(steps, 0, -1):
        f = i / steps
        a = int(255 * (1.0 - f) ** 1.6)
        rect = pygame.Rect(0, 0, max(1, int(w * f)), max(1, int(h * f)))
        rect.center = (w // 2, h // 2)
        pygame.draw.ellipse(surf, (BLUSH_COLOR[0], BLUSH_COLOR[1],
                                   BLUSH_COLOR[2], a), rect)
    return surf


def _build_scanlines(w, h):
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(0, h, SCANLINE_STEP):
        pygame.draw.line(surf, (0, 0, 0, SCANLINE_ALPHA), (0, y), (w, y))
    return surf


class Eye:
    """One eye. side = -1 left, +1 right, so inner/outer lid coverage mirrors
    correctly for asymmetric emotions like sad."""

    def __init__(self, side):
        self.side = side
        base = EMOTIONS[DEFAULT_EMOTION]
        self.cur = {k: base[k] for k in SHAPE_KEYS}
        self.vel = {k: 0.0 for k in SHAPE_KEYS}
        self.tgt = dict(self.cur)
        self._surf = None
        self._surf_size = (0, 0)

    def set_target(self, shape):
        self.tgt = shape

    def update(self, dt, freq, zeta):
        for k in SHAPE_KEYS:
            self.cur[k], self.vel[k] = spring_step(
                self.cur[k], self.vel[k], self.tgt[k], freq, zeta, dt)

    def draw(self, dest, cx, cy, base_w, base_h, blink_factor, hotspot, hx, hy, tint,
             vgrad=None):
        c = self.cur
        w = base_w * c["ws"] * (1.0 + (1.0 - blink_factor) * BLINK_SQUASH)
        h = max(2.0, base_h * c["hs"] * blink_factor)
        cy = cy + base_h * c["oy"]

        # Render oversized, then shrink on the way out: that is the whole
        # antialiasing strategy, and it smooths the lid cuts as well as the
        # outline. One code path, so nothing can pop mid-morph.
        ss = SUPERSAMPLE
        pad = 4
        out_size = (int(w) + pad * 2, int(h) + pad * 2)
        need = (out_size[0] * ss, out_size[1] * ss)
        if self._surf is None or self._surf_size != need:
            self._surf = pygame.Surface(need, pygame.SRCALPHA)
            self._surf_size = need
        surf = self._surf
        surf.fill((0, 0, 0, 0))

        w, h = w * ss, h * ss
        left, top = float(pad * ss), float(pad * ss)
        right, bottom = left + w, top + h

        # One shape for every preset. Eyelids are carves layered on top, never
        # a different draw path - switching draw paths mid-morph is what pops.
        pygame.draw.polygon(surf, tint, _superellipse(
            left + w / 2, top + h / 2, w / 2, h / 2, EYE_SUPER))

        if self.side < 0:      # left eye: inner edge is on the right
            t_l, t_r = c["top_out"], c["top_in"]
        else:                  # right eye: inner edge is on the left
            t_l, t_r = c["top_in"], c["top_out"]
        b_l = b_r = c["bottom"]

        # Carve with a fully transparent colour: pygame.draw writes raw RGBA on
        # an SRCALPHA surface, so this erases rather than paints.
        # Carve to the eye colour at zero alpha, not to black: shrinking the
        # buffer interpolates RGB as well as alpha, and carving to black would
        # leave a dark fringe around every curve.
        clear = (tint[0], tint[1], tint[2], 0)
        over = pad * ss
        if t_l > 0.001 or t_r > 0.001 or c["tbend"] > 0.001:
            curve = _lid_curve(left, right, top, h, t_l, t_r, from_top=True, bend=c["tbend"])
            pygame.draw.polygon(surf, clear, [(left, top - over)] + curve + [(right, top - over)])
        if b_l > 0.001 or b_r > 0.001 or c["bend"] > 0.001:
            curve = _lid_curve(left, right, bottom, h, b_l, b_r, from_top=False, bend=c["bend"])
            pygame.draw.polygon(surf, clear, [(left, bottom + over)] + curve + [(right, bottom + over)])

        # Vertical gradient, fixed to the eye (bright top -> deeper bottom).
        if vgrad is not None:
            surf.blit(pygame.transform.scale(vgrad, (max(1, int(w)), max(1, int(h)))),
                      (int(left), int(top)), special_flags=pygame.BLEND_RGBA_MULT)

        # Spherical shading. The hotspot centre is the gaze cue.
        hs_size = hotspot.get_width()
        px = left + w * (0.5 + hx * HOTSPOT_RANGE) - hs_size / 2
        py = top + h * (0.5 + hy * HOTSPOT_RANGE) - hs_size / 2
        surf.blit(hotspot, (int(px), int(py)), special_flags=pygame.BLEND_RGBA_MULT)

        # Tilt the finished eye. Mirrored per eye, so a positive angle drops
        # the OUTER corner on both sides (sad/tired) and a negative one drops
        # the inner corners (angry). pygame rotates counter-clockwise.
        ang = c["ang"]
        if abs(ang) > 0.05:
            surf = pygame.transform.rotate(surf, -self.side * ang)
            out_size = (surf.get_width() // ss, surf.get_height() // ss)
        img = pygame.transform.smoothscale(surf, out_size)
        rect = img.get_rect(center=(int(cx), int(cy)))
        dest.blit(img, rect)


class Face:
    # ---------------------------------------------------------------- setup
    def __init__(self):
        self.left, self.right = Eye(-1), Eye(+1)
        self.mode = DEFAULT_MODE
        self.emotion = DEFAULT_EMOTION
        self._emotion_age = 0.0

        self.dims = dict(DIM_BASELINE)
        self._last_event = {}
        self._repeats = {}

        self.amplitude = 0.0
        self._amp = 0.0
        self._amp_external = False

        self._t = 0.0
        self._blink_t = None
        self._next_blink = random.uniform(2.5, 5.0)

        self._look = [0.0, 0.0]          # where the eyes actually point
        self._look_t = [0.0, 0.0]        # where a shift is heading
        self._look_from = [0.0, 0.0]     # where that shift began
        self._shift_t = 1.0              # progress through the current shift
        self._averting = False
        self._avert_left = 0.0           # seconds remaining of this aversion
        self._next_avert = 3.0
        self._noise = (_Noise(11), _Noise(29))
        self._drift = [0.0, 0.0]
        self._next_look = random.uniform(*ACTIVITY_HOLD)
        self._sacc = [0.0, 0.0]
        self._sacc_t = [0.0, 0.0]
        self._sacc_i = 0
        self._next_sacc = random.uniform(*SACCADE_EVERY)
        self._hot = [0.0, 0.0]

        self._symbols = []          # floating manpu
        self._next_symbol = 1.2
        self.activity = "observing"
        self._next_activity = random.uniform(*ACTIVITY_HOLD)
        self.sleep_pressure = 0.0
        self._slept = 0.0

        self._gesture = None
        self._gesture_t = 0.0

        self._motion = dict(EMOTION_MOTION[DEFAULT_EMOTION])
        self._antic_until = -1.0
        self._antic_from = None
        self._antic_amt = 0.0

        self._hotspot = None
        self._hotspot_for = 0
        self._vgrad = None
        self._scanlines = None
        self._scan_for = (0, 0)
        self._blush = None

    # ------------------------------------------------------------- controls
    def set_mode(self, mode):
        if mode not in MODES or mode == self.mode:
            return
        # Entering thinking always gets a flick. RoboEyes signals confusion
        # with MOTION rather than shape, which dodges the anger ambiguity a
        # static asymmetric brow creates.
        if mode == "thinking":
            self.gesture("shake")
        if self.mode == "sleeping":
            self.gesture("perk")
        self.mode = mode

    def set_emotion(self, emotion, hold=True):
        if emotion not in EMOTIONS:
            return
        if emotion != self.emotion:
            strength = EMOTION_MOTION[emotion]["antic"]
            if strength > 0.0:
                # Snapshot where the eyes are NOW; for the next few frames the
                # spring is aimed at a pose on the far side of that, i.e. away
                # from where it is about to go. Then the target flips and the
                # eyes arrive already carrying momentum.
                self._antic_from = dict(self.left.cur)
                self._antic_amt = strength
                self._antic_until = self._t + ANTICIPATION_TIME
        self.emotion = emotion
        self._emotion_age = 0.0 if hold else EMOTION_HOLD

    def set_amplitude(self, a):
        """Live audio envelope, 0..1. Call at audio rate from the TTS playback
        callback (step 2). Until something does, the face synthesises its own
        so `speaking` still looks like speaking."""
        self.amplitude = clamp(a)
        self._amp_external = True

    def look_at(self, x, y=0.0):
        """Gaze target, -1..1. Wire a camera to this later."""
        self._look_t = [clamp(x, -1, 1), clamp(y, -1, 1)]

    def blink(self):
        if self._blink_t is None:
            self._blink_t = 0.0

    def gesture(self, name):
        if name in GESTURES:
            self._gesture, self._gesture_t = name, 0.0

    def event(self, name):
        """Post an emotion event by name. The main driver entry point: the
        voice loop calls face.event("wake_word") and the face works out what
        that means for its state, rather than being handed an expression."""
        spec = EVENTS.get(name)
        if spec is None:
            return
        now = self._t
        last = self._last_event.get(name)
        if last is not None and now - last < REPEAT_WINDOW:
            self._repeats[name] = self._repeats.get(name, 0) + 1
        else:
            self._repeats[name] = 0
        self._last_event[name] = now
        scale = REPEAT_FACTOR ** self._repeats[name]

        for dim in ("stim", "social", "happy", "confident"):
            if dim in spec:
                self.dims[dim] = clamp(self.dims[dim] + spec[dim] * scale, 0.0, 1.0)
        if "emotion" in spec and scale > 0.3:
            self.set_emotion(spec["emotion"])
        if "gesture" in spec and scale > 0.3:
            self.gesture(spec["gesture"])
        if "symbol" in spec and scale > 0.3:
            self.symbol(spec["symbol"])
        if self.mode == "sleeping" and self.dims["stim"] > SLEEP_STIM_GATE:
            self.set_mode("idle")

    @property
    def mood(self):
        """A slow label over the fast dimensions - Vector's 'simple mood' is
        just a name mapped to emotion value ranges."""
        s, h = self.dims["stim"], self.dims["happy"]
        if self.mode == "sleeping":
            return "asleep"
        if h < 0.25:
            return "frustrated"
        return "highstim" if s > 0.6 else "lowstim" if s < 0.2 else "medstim"

    # --------------------------------------------------------------- update
    def update(self, dt):
        self._t += dt
        self._decay_dims(dt)
        self._update_sleep(dt)
        self._update_activity(dt)
        self._update_gaze(dt)
        self._update_blink(dt)
        self._update_amplitude(dt)

        self._emotion_age += dt
        self._blend_motion(dt)
        self._update_symbols(dt)
        if self._gesture is not None:
            self._gesture_t += dt
            if self._gesture_t >= GESTURES[self._gesture]["dur"]:
                self._gesture = None

        shape = self._compose()
        self.left.set_target(shape)
        self.right.set_target(shape)
        self.left.update(dt, self._motion["freq"], self._motion["zeta"])
        self.right.update(dt, self._motion["freq"], self._motion["zeta"])

    def symbol(self, kind):
        """Throw one manpu symbol. Drivers can call this directly for a
        one-off, e.g. a heart when someone says something kind."""
        if kind not in SYMBOL_COLOR:
            return
        self._symbols.append({"kind": kind, "t": 0.0,
                              "side": random.choice((-1, 1)),
                              "sway": random.uniform(-1.0, 1.0),
                              "tilt": random.uniform(-0.22, 0.22)})

    def _wanted_symbol(self):
        """Sleeping always yawns Z's; otherwise the current emotion decides,
        and only while the emotion is fresh - a permanently decorated face
        stops meaning anything."""
        if self.mode == "sleeping":
            return "zzz"
        if self._emotion_age > EMOTION_HOLD:
            return None
        return EMOTION_SYMBOL.get(self.emotion)

    def _update_symbols(self, dt):
        for z in self._symbols:
            z["t"] += dt
        self._symbols = [z for z in self._symbols if z["t"] < SYMBOL_LIFE]
        want = self._wanted_symbol()
        if want is None:
            self._next_symbol = 0.5
            return
        self._next_symbol -= dt
        if self._next_symbol <= 0:
            lo, hi = SYMBOL_EVERY
            self._next_symbol = random.uniform(lo, hi)
            self.symbol(want)

    def _blend_motion(self, dt):
        """Ease the motion signature toward the current emotion's. The spring
        constants themselves blend fast, so an emotion change takes effect on
        the very transition that announces it."""
        tgt = EMOTION_MOTION[self.emotion]
        slow, fast = clamp(dt * 3.0), clamp(dt * 12.0)
        for key in MOTION_KEYS:
            k = fast if key in ("freq", "zeta") else slow
            self._motion[key] = lerp(self._motion[key], tgt[key], k)

    def _decay_dims(self, dt):
        for dim, hl in DIM_HALFLIFE.items():
            base = DIM_BASELINE[dim]
            self.dims[dim] = base + (self.dims[dim] - base) * (0.5 ** (dt / hl))

    def _update_sleep(self, dt):
        if self.mode == "sleeping":
            self.sleep_pressure = clamp(self.sleep_pressure - SLEEP_FALL * dt)
            self._slept += dt
            if self.sleep_pressure <= 0.05 and self._slept > MIN_SLEEP:
                self.set_mode("idle")
        else:
            self._slept = 0.0
            self.sleep_pressure = clamp(self.sleep_pressure + SLEEP_RISE * dt)
            if (self.mode == "idle" and self.sleep_pressure > SLEEP_ENTER
                    and self.dims["stim"] < SLEEP_STIM_GATE):
                self.set_mode("sleeping")

    def _update_activity(self, dt):
        """The idle three-way pick, weighted by stimulation and sleep debt, so
        a stimulated face keeps looking around and a tired one dozes."""
        if self.mode != "idle":
            return
        self._next_activity -= dt
        if self._next_activity > 0:
            return
        s, sp = self.dims["stim"], self.sleep_pressure
        weights = {
            "observing": 1.0,
            "drifting": 0.45 + 1.6 * s,
            "dozing": 0.10 + 3.0 * max(0.0, sp - 0.45) * (1.0 - s),
        }
        r = random.uniform(0, sum(weights.values()))
        for name, wgt in weights.items():
            r -= wgt
            if r <= 0:
                self.activity = name
                break
        self._next_activity = random.uniform(*ACTIVITY_HOLD)

    def _shift_to(self, x, y):
        """Begin an eased move of the gaze to a new point."""
        self._look_from = list(self._look)
        self._look_t = [clamp(x, -1, 1), clamp(y, -1, 1)]
        self._shift_t = 0.0

    def _pick_aversion(self, kind):
        """Direction and size of an aversion, per its conversational function.
        Cognitive ones go up (thinking); social ones go sideways and small."""
        spec = AVERSION[kind]
        if random.random() < spec["up"]:
            x = random.uniform(-0.25, 0.25) * AVERT_H
            y = -AVERT_V
        else:
            x = random.choice((-1, 1)) * AVERT_H * random.uniform(0.7, 1.0)
            y = random.uniform(-0.25, 0.15) * AVERT_V
        k = spec["scale"]
        return x * k, y * k

    def _update_gaze(self, dt):
        style = MODES[self.mode]["gaze"]

        if style == "still":
            self._averting = False
            self._shift_to(0.0, 0.35) if self._shift_t >= 1.0 and \
                abs(self._look_t[1] - 0.35) > 0.01 else None
        elif style == "wander":
            self._next_look -= dt
            if self._next_look <= 0:
                if self.activity == "drifting":
                    self._shift_to(random.uniform(-0.7, 0.7),
                                   random.uniform(-0.4, 0.4))
                    self._next_look = random.uniform(3.0, 6.0)
                elif self.activity == "dozing":
                    self._shift_to(random.uniform(-0.2, 0.2),
                                   random.uniform(0.2, 0.6))
                    self._next_look = random.uniform(4.0, 8.0)
                else:
                    self._shift_to(random.uniform(-0.3, 0.3),
                                   random.uniform(-0.2, 0.2))
                    self._next_look = random.uniform(2.5, 5.0)
        else:
            # lock / avert: hold the person's gaze, but break it periodically.
            # Preferred mutual gaze is only 3-5 s before it gets uncomfortable,
            # so a face that never looks away is a face that stares.
            kind = "cognitive" if style == "avert" else (
                "speaking" if self.mode == "speaking" else "listening")
            spec = AVERSION[kind]
            if self._averting:
                self._avert_left -= dt
                if self._avert_left <= 0:
                    self._averting = False
                    self._shift_to(0.0, 0.0)          # back to their eyes
                    g_m, g_s = spec["gap"]
                    self._next_avert = max(0.8, random.gauss(g_m, g_s))
            else:
                self._next_avert -= dt
                if self._next_avert <= 0:
                    self._averting = True
                    l_m, l_s = spec["length"]
                    self._avert_left = max(0.4, random.gauss(l_m, l_s))
                    self._shift_to(*self._pick_aversion(kind))

        # advance the eased shift
        if self._shift_t < 1.0:
            self._shift_t = min(1.0, self._shift_t + dt / SHIFT_TIME)
            e = ease(self._shift_t)
            bias_x = self._motion["gaze_x"]
            bias_y = self._motion["gaze_y"]
            for i, b in ((0, bias_x), (1, bias_y)):
                a = self._look_from[i]
                z = clamp(self._look_t[i] + b, -1, 1)
                self._look[i] = a + (z - a) * e
        else:
            # settled: hold the target, plus the emotion's pull
            self._look[0] = smooth(self._look[0],
                                   clamp(self._look_t[0] + self._motion["gaze_x"], -1, 1),
                                   LOOK_LERP, dt)
            self._look[1] = smooth(self._look[1],
                                   clamp(self._look_t[1] + self._motion["gaze_y"], -1, 1),
                                   LOOK_LERP, dt)

        # Always-on idle drift, kept SEPARATE from the gaze target. Adding it
        # into _look each frame made it accumulate into a random walk that
        # wandered far outside its own limit.
        if self.mode == "sleeping":
            self._drift = [0.0, 0.0]
        else:
            d = DRIFT_AMOUNT * max(0.4, self._motion["drift"])
            self._drift = [self._noise[0](self._t * DRIFT_RATE) * d,
                           self._noise[1](self._t * DRIFT_RATE) * d * 0.6]

        # saccades: fast flicks between three points on the face being looked
        # at. These move the PUPIL, not the eye.
        drift = max(0.15, self._motion["drift"])
        self._next_sacc -= dt
        if self._next_sacc <= 0 and self.mode != "sleeping":
            nxt = random.randrange(len(SACCADE_POINTS))
            if nxt == self._sacc_i:
                nxt = (nxt + 1) % len(SACCADE_POINTS)
            self._sacc_i = nxt
            px, py = SACCADE_POINTS[nxt]
            mag = SACCADE_MAG * min(1.6, drift)
            self._sacc_t = [px * mag, py * mag]
            lo, hi = SACCADE_EVERY
            self._next_sacc = random.uniform(lo, hi) / min(1.6, drift)
        if self.mode == "sleeping":
            self._sacc_t = [0.0, 0.0]
        self._sacc[0] = smooth(self._sacc[0], self._sacc_t[0], SACCADE_LERP, dt)
        self._sacc[1] = smooth(self._sacc[1], self._sacc_t[1], SACCADE_LERP, dt)

    def _update_blink(self, dt):
        spec = MODES[self.mode]["blink"]
        if spec is not None:
            self._next_blink -= dt
            if self._next_blink <= 0 and self._blink_t is None:
                self.blink()
                lo, hi = spec
                # a dozing face blinks slowly
                if self.mode == "idle" and self.activity == "dozing":
                    lo, hi = lo * 1.6, hi * 1.6
                bm = self._motion["blink_mul"]
                lo, hi = lo * bm, hi * bm
                self._next_blink = random.uniform(lo, hi)
        if self._blink_t is not None:
            self._blink_t += dt
            if self._blink_t >= BLINK_DURATION * self._motion["blink_dur"]:
                self._blink_t = None

    def _update_amplitude(self, dt):
        if not self._amp_external:
            # No audio source attached yet: synthesise a plausible envelope so
            # `speaking` reads as speaking. Two out-of-phase sines so it never
            # sounds like a metronome. Step 2 replaces this with real RMS.
            if self.mode == "speaking":
                t = self._t
                self.amplitude = clamp(
                    0.45 + 0.30 * math.sin(t * 9.0) + 0.22 * math.sin(t * 15.3 + 1.3))
            elif self.mode == "listening":
                self.amplitude = clamp(0.25 + 0.18 * math.sin(self._t * 3.1))
            else:
                self.amplitude = 0.0
        self._amp = lerp(self._amp, self.amplitude, 0.35)

    # -------------------------------------------------------------- compose
    def _compose(self):
        """emotion (shape) x mode (modifier) x live dimensions. One dict, one
        render path - every visual difference is a blended number."""
        e = EMOTIONS[self.emotion]
        m = MODES[self.mode]

        # emotion fades back toward neutral rather than snapping
        if self._emotion_age > EMOTION_HOLD and self.emotion != "neutral":
            f = clamp((self._emotion_age - EMOTION_HOLD) / EMOTION_FADE)
            n = EMOTIONS["neutral"]
            e = {k: lerp(e[k], n[k], f) for k in SHAPE_KEYS}
            if f >= 1.0:
                self.emotion = "neutral"

        # a genuinely happier face opens up a little; a low one settles
        happy_bias = (self.dims["happy"] - DIM_BASELINE["happy"]) * 0.10
        stim_bias = (self.dims["stim"] - DIM_BASELINE["stim"]) * 0.12
        amp = self._amp * m["amp_gain"]

        # dozing droops the lids without changing the chosen emotion
        doze = 0.18 if (self.mode == "idle" and self.activity == "dozing") else 0.0

        g_scale, g_oy, _, _ = self._gesture_offsets()

        shape = dict(
            ws=e["ws"] * m["ws_mul"] * (1.0 + stim_bias * 0.5 + g_scale),
            hs=e["hs"] * m["hs_mul"] * (1.0 + stim_bias + happy_bias + amp + g_scale) * (1.0 - doze),
            oy=e["oy"] + m["oy_add"] + g_oy + doze * 0.4,
            top_in=clamp(e["top_in"] + m["top_add"] + doze, 0.0, 0.85),
            top_out=clamp(e["top_out"] + m["top_add"] + doze, 0.0, 0.85),
            bottom=clamp(e["bottom"] + m["bottom_add"], 0.0, 0.85),
            bend=e["bend"],
            tbend=e["tbend"],
            ang=e["ang"],
        )

        # Anticipation window: aim at the mirror of the real target, so the
        # eyes move the wrong way first. Reflected about where they stood when
        # the emotion changed, so the counter-pose is stable, not drifting.
        if self._t < self._antic_until and self._antic_from is not None:
            a = self._antic_from
            shape = {k: a[k] - (shape[k] - a[k]) * self._antic_amt for k in SHAPE_KEYS}
        if self.mode == "sleeping":
            # Asleep is asleep. Without this the emotion's tilt and asymmetric
            # lids survive into closed eyes, and a shut eye reads as a sad or
            # annoyed squint instead of a sleeping one.
            lid = (shape["top_in"] + shape["top_out"]) * 0.5
            shape["top_in"] = shape["top_out"] = lid
            shape["ang"] = 0.0
            shape["bend"] = shape["tbend"] = 0.0

        return shape

    def _gesture_offsets(self):
        """One-shot 'get in' motion: (scale, vertical, horizontal, tilt)."""
        if self._gesture is None:
            return 0.0, 0.0, 0.0, 0.0
        g = GESTURES[self._gesture]
        p = clamp(self._gesture_t / g["dur"])
        kind = g["kind"]
        if kind == "perk":
            # snap open, then settle - anticipation followed by rest
            return (PERK_OVERSHOOT * math.sin(p * math.pi) * (1 - p * 0.4),
                    -0.03 * math.sin(p * math.pi), 0.0, 0.0)
        if kind == "bounce":
            return 0.0, -BOUNCE_HEIGHT * abs(math.sin(p * math.pi * 2)) * (1 - p), 0.0, 0.0
        if kind == "shake":
            return 0.0, 0.0, math.sin(p * math.pi * SHAKE_CYCLES * 2) * SHAKE_AMPLITUDE * (1 - p), 0.0
        if kind == "tilt":
            # one eye up, one down - the flat-face equivalent of a head tilt,
            # which is the universal "huh?" cue
            return 0.0, 0.0, 0.0, TILT_MAX * math.sin(p * math.pi)
        return 0.0, 0.0, 0.0, 0.0

    def _blink_factor(self):
        if self._blink_t is None:
            return 1.0
        p = self._blink_t / (BLINK_DURATION * self._motion["blink_dur"])
        if p >= 1.0:
            return 1.0
        return 1.0 - math.sin(p * math.pi) * (1.0 - BLINK_MIN)

    # ----------------------------------------------------------------- draw
    def draw(self, surf):
        surf.fill(BG_COLOR)
        W, H = surf.get_size()
        base_w, base_h = W * EYE_W_FRAC, H * EYE_H_FRAC
        gap, cy = W * EYE_GAP_FRAC, H * 0.5

        hs_size = int(max(base_w, base_h) * 2.6 * SUPERSAMPLE)
        if self._hotspot is None or self._hotspot_for != hs_size:
            self._hotspot = _build_hotspot(hs_size)
            self._hotspot_for = hs_size
        if self._scanlines is None or self._scan_for != (W, H):
            self._scanlines = _build_scanlines(W, H)
            self._scan_for = (W, H)
        if self._vgrad is None and (GRAD_TOP != 1.0 or GRAD_BOTTOM != 1.0):
            self._vgrad = _build_vgrad()

        mo = self._motion
        rate, depth = mo["breathe_hz"], mo["breathe"]
        if self.mode == "sleeping":
            rate *= SLEEP_BREATHE_RATE
            depth *= SLEEP_BREATHE_DEPTH
        breathe = math.sin(self._t * 1.6 * rate) * (H * 0.006 * depth)
        bf = self._blink_factor()
        _, _, g_dx, g_tilt = self._gesture_offsets()

        # The EYE only follows the slow gaze drift. Saccades belong to the
        # pupil: in a real face the eyeball flicks inside a head that is not
        # moving, and translating the whole eye four times a second is what
        # made this judder.
        gx = clamp(self._look[0] + self._drift[0], -1.0, 1.0)
        gy = clamp(self._look[1] + self._drift[1], -1.0, 1.0)
        hx = clamp(gx + self._sacc[0], -1.0, 1.0)
        hy = clamp(gy + self._sacc[1], -1.0, 1.0)
        look_x = gx * gap * LOOK_RANGE_X + g_dx * base_w
        look_y = gy * base_h * LOOK_RANGE_Y * 0.5

        # Tremor: the fast small quiver that turns a wide stare into fear and a
        # droopy face into a plea. Two incommensurate rates so it never buzzes.
        if mo["tremor"] > 0.0005:
            hz = mo["tremor_hz"]
            look_x += math.sin(self._t * hz * 6.283) * mo["tremor"] * base_w
            look_y += math.sin(self._t * hz * 5.1 + 1.7) * mo["tremor"] * base_h * 0.6

        # Eyes brighten a little with the voice - a cheap but strong "that
        # sound is coming from me" cue while speaking.
        lift = clamp(0.30 * self._amp * MODES[self.mode]["amp_gain"] * 5, 0.0, 0.35)
        tint = tuple(int(c + (255 - c) * lift) for c in EYE_COLOR)

        # Asymmetry. Looking left grows the left eye and shrinks the right,
        # which reads as the near eye being closer to you - Cozmo's "curious"
        # look. The extra EYE_IMBALANCE keeps the two from ever being twins.
        lm = 1.0 - CURIOUS_GAIN * gx
        rm = (1.0 + CURIOUS_GAIN * gx) * EYE_IMBALANCE

        # Push the eyes apart only as far as needed so they never collide when
        # they grow or squash. At rest this changes nothing.
        squash = 1.0 + (1.0 - bf) * BLINK_SQUASH
        widest = base_w * max(self.left.cur["ws"] * lm, self.right.cur["ws"] * rm) * squash
        gap = max(gap, widest / 2.0 + W * MIN_EYE_CLEARANCE)

        tilt = g_tilt * base_h
        self.left.draw(surf, W * 0.5 - gap + look_x, cy + look_y + breathe - tilt,
                       base_w * lm, base_h * lm, bf, self._hotspot, hx, hy, tint,
                       self._vgrad)
        self.right.draw(surf, W * 0.5 + gap + look_x, cy + look_y + breathe + tilt,
                        base_w * rm, base_h * rm, bf, self._hotspot, hx, hy, tint,
                        self._vgrad)

        self._draw_blush(surf, W, H, gap, cy + look_y + breathe)
        self._draw_symbols(surf, W, H, gap)

        if SCANLINE_ALPHA > 0:
            surf.blit(self._scanlines, (0, 0))

    def _draw_blush(self, surf, W, H, gap, eye_cy):
        """Two soft ovals under the eyes. Opacity follows how happy it is, so
        this is ambient warmth rather than an effect that fires."""
        warmth = clamp((self.dims["happy"] - 0.35) / 0.55)
        if self.emotion in ("happy", "pleading"):
            warmth = max(warmth, 0.55)
        if self.mode == "sleeping":
            warmth *= 0.5
        if warmth < 0.03:
            return
        bw, bh = int(W * BLUSH_W), int(H * BLUSH_H)
        if bw < 2 or bh < 2:
            return
        if self._blush is None or self._blush.get_size() != (bw, bh):
            self._blush = _build_blush(bw, bh)
        tmp = self._blush.copy()
        tmp.set_alpha(int(BLUSH_ALPHA * warmth))
        y = int(eye_cy + H * BLUSH_DROP)
        for side in (-1, 1):
            x = int(W * 0.5 + side * gap * (1.0 + BLUSH_OUT))
            surf.blit(tmp, tmp.get_rect(center=(x, y)))

    def _draw_symbols(self, surf, W, H, gap):
        if not self._symbols:
            return
        for z in self._symbols:
            p = z["t"] / SYMBOL_LIFE
            alpha = min(1.0, p / 0.12) * (1.0 - p) ** 1.4
            if alpha <= 0.01:
                continue
            size = H * SYMBOL_SIZE * (1.0 + (SYMBOL_GROW - 1.0) * p)
            fall = z["kind"] == "tear"
            cx = W * 0.5 + z["side"] * gap * 1.45 + z["sway"] * W * SYMBOL_DRIFT * p
            cy = H * 0.40 + (p * H * SYMBOL_RISE if fall else -p * H * SYMBOL_RISE)
            self._blit_symbol(surf, z["kind"], cx, cy, size, alpha, z["tilt"])

    @staticmethod
    def _blit_symbol(surf, kind, cx, cy, size, alpha, tilt):
        box = int(size * 3) + 8
        tmp = pygame.Surface((box, box), pygame.SRCALPHA)
        ox = oy = box / 2
        r, g, b = SYMBOL_COLOR.get(kind, EYE_COLOR)
        col = (r, g, b, max(0, min(255, int(255 * alpha))))
        t = max(2, int(size * 0.16))
        h = size
        w = size * 0.78

        if kind == "zzz":
            for a, b2 in (((ox - w / 2, oy - h / 2), (ox + w / 2, oy - h / 2)),
                          ((ox + w / 2, oy - h / 2), (ox - w / 2, oy + h / 2)),
                          ((ox - w / 2, oy + h / 2), (ox + w / 2, oy + h / 2))):
                pygame.draw.line(tmp, col, a, b2, t)
        elif kind == "heart":
            rr = h * 0.28
            pygame.draw.circle(tmp, col, (int(ox - rr * 0.72), int(oy - rr * 0.45)), int(rr))
            pygame.draw.circle(tmp, col, (int(ox + rr * 0.72), int(oy - rr * 0.45)), int(rr))
            pygame.draw.polygon(tmp, col, [(ox - rr * 1.55, oy - rr * 0.2),
                                           (ox + rr * 1.55, oy - rr * 0.2),
                                           (ox, oy + rr * 1.6)])
        elif kind == "sparkle":
            a = h * 0.55
            b2 = h * 0.16
            pygame.draw.polygon(tmp, col, [(ox, oy - a), (ox + b2, oy - b2),
                                           (ox + a, oy), (ox + b2, oy + b2),
                                           (ox, oy + a), (ox - b2, oy + b2),
                                           (ox - a, oy), (ox - b2, oy - b2)])
        elif kind in ("sweat", "tear"):
            rr = h * 0.30
            pygame.draw.circle(tmp, col, (int(ox), int(oy + rr * 0.45)), int(rr))
            pygame.draw.polygon(tmp, col, [(ox - rr * 0.92, oy + rr * 0.30),
                                           (ox + rr * 0.92, oy + rr * 0.30),
                                           (ox, oy - rr * 1.55)])
        elif kind == "vein":
            # the crucifix-shaped popping vein
            a = h * 0.42
            for dx, dy in ((0, 1), (1, 0)):
                pygame.draw.line(tmp, col, (ox - a * dx, oy - a * dy),
                                 (ox + a * dx, oy + a * dy), t + 1)
            for sx in (-1, 1):
                pygame.draw.line(tmp, col, (ox + sx * a * 0.55, oy - a * 0.55),
                                 (ox + sx * a * 0.95, oy - a * 0.95), t)
        elif kind == "question":
            pts = []
            for i in range(11):
                ang = math.pi * (1.15 - 1.35 * i / 10.0)
                pts.append((ox + math.cos(ang) * h * 0.30,
                            oy - h * 0.22 - math.sin(ang) * h * 0.30))
            pts.append((ox + h * 0.02, oy + h * 0.16))
            pygame.draw.lines(tmp, col, False, pts, t)
            pygame.draw.circle(tmp, col, (int(ox + h * 0.02), int(oy + h * 0.44)), t)
        elif kind == "exclaim":
            pygame.draw.line(tmp, col, (ox, oy - h * 0.50), (ox, oy + h * 0.14), t + 1)
            pygame.draw.circle(tmp, col, (int(ox), int(oy + h * 0.44)), t)
        elif kind == "note":
            pygame.draw.circle(tmp, col, (int(ox - h * 0.16), int(oy + h * 0.34)),
                               int(h * 0.20))
            pygame.draw.line(tmp, col, (ox + h * 0.02, oy + h * 0.34),
                             (ox + h * 0.02, oy - h * 0.44), t)
            pygame.draw.line(tmp, col, (ox + h * 0.02, oy - h * 0.44),
                             (ox + h * 0.34, oy - h * 0.26), t)

        if abs(tilt) > 0.01:
            tmp = pygame.transform.rotate(tmp, tilt * 40.0)
        surf.blit(tmp, tmp.get_rect(center=(int(cx), int(cy))))

    @staticmethod
    def _blit_z(surf, cx, cy, size, alpha, tilt):
        pad = int(size * 0.9)
        box = int(size * 2) + pad
        tmp = pygame.Surface((box, box), pygame.SRCALPHA)
        w, h = size * 0.78, size
        ox, oy = box / 2, box / 2
        col = (EYE_COLOR[0], EYE_COLOR[1], EYE_COLOR[2],
               max(0, min(255, int(255 * alpha))))
        thick = max(2, int(size * 0.16))
        pts = [((ox - w / 2, oy - h / 2), (ox + w / 2, oy - h / 2)),
               ((ox + w / 2, oy - h / 2), (ox - w / 2, oy + h / 2)),
               ((ox - w / 2, oy + h / 2), (ox + w / 2, oy + h / 2))]
        for a, b in pts:
            pygame.draw.line(tmp, col, a, b, thick)
        if abs(tilt) > 0.01:
            tmp = pygame.transform.rotate(tmp, tilt * 40.0)
        surf.blit(tmp, tmp.get_rect(center=(int(cx), int(cy))))

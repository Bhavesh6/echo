#!/usr/bin/env python3
"""
anim.py - hand-authored animation, layered over the live face.

Taby's expressions are exported video: 128x64 monochrome frames baked at 8-12
fps. They look hand-crafted because a person animated every frame - and they
are also completely inert. A baked frame cannot know how loud you are talking,
how long since you last spoke, or which mood it is in.

Cozmo solved this properly and this copies its approach. From the Vector
technical manual's list of animation tracks:

    117.14. PROCEDURALFACE
            ProceduralFaceKeyFrame  ProceduralFace[]

Their animators keyframed the PROCEDURAL FACE PARAMETERS over time rather than
painting pixels. So an authored animation still composes with everything else
the face is doing - it is hand-tuned AND alive.

An animation here is a JSON file of tracks, each track a list of
[time, value] keyframes against one named parameter:

    {"name": "delight", "tracks": {
        "arc":  [[0.0, 0.0], [0.12, 0.95], [1.1, 0.95], [1.5, 0.0]],
        "hs":   [[0.0, 1.0], [0.08, 1.30], [0.28, 1.0]]
    }}

Values are absolute for that parameter and easing is smoothstep, which is the
ease-in/ease-out that gaze and animation research both keep pointing at.
Anything not mentioned in a track is left entirely alone, so an animation can
touch one number without freezing the rest of the face.
"""
import json
import os

ANIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "animations")

# Which names a track may target. Kept explicit so a typo in a hand-written
# file fails loudly at load rather than silently doing nothing at runtime.
SHAPE_TARGETS = ("ws", "hs", "oy", "top_in", "top_out", "bottom",
                 "bend", "tbend", "ang")
EXTRA_TARGETS = ("heart", "arc", "pupil",
                 "mouth_curve", "mouth_w", "mouth_open", "mouth_cat",
                 "brow_tilt", "brow_lift", "brow_show")
TARGETS = SHAPE_TARGETS + EXTRA_TARGETS


def _smoothstep(p):
    p = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)
    return p * p * (3.0 - 2.0 * p)


class Clip:
    """One authored animation."""

    def __init__(self, name, tracks, loop=False, blend=0.10):
        self.name = name
        self.loop = bool(loop)
        self.blend = float(blend)      # seconds to fade in and out
        self.tracks = {}
        for target, keys in tracks.items():
            if target not in TARGETS:
                raise ValueError("%s: unknown animation target %r"
                                 % (name, target))
            ks = sorted((float(t), float(v)) for t, v in keys)
            if ks:
                self.tracks[target] = ks
        self.length = max((ks[-1][0] for ks in self.tracks.values()), default=0.0)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d.get("name") or os.path.splitext(os.path.basename(path))[0],
                   d.get("tracks", {}), d.get("loop", False),
                   d.get("blend", 0.10))

    def sample(self, target, t):
        """The value of one track at time t, or None if this clip does not
        touch that parameter - which is what lets an animation move the mouth
        while the eyes carry on doing whatever they were doing."""
        ks = self.tracks.get(target)
        if not ks:
            return None
        if t <= ks[0][0]:
            return ks[0][1]
        if t >= ks[-1][0]:
            return ks[-1][1]
        for i in range(len(ks) - 1):
            t0, v0 = ks[i]
            t1, v1 = ks[i + 1]
            if t0 <= t <= t1:
                span = t1 - t0
                p = 0.0 if span <= 0 else (t - t0) / span
                return v0 + (v1 - v0) * _smoothstep(p)
        return ks[-1][1]


class Player:
    """Plays one clip at a time. One is deliberate: two authored animations
    fighting over the same parameter is a mess nobody can debug, and a face
    doing two set-pieces at once reads as broken rather than expressive."""

    def __init__(self, directory=ANIM_DIR):
        self.directory = directory
        self.clips = {}
        self.current = None
        self.t = 0.0
        self.reload()

    def reload(self):
        """Re-read every clip from disk. Called on a key press while tuning, so
        edits to a JSON file show up without restarting."""
        self.clips = {}
        if not os.path.isdir(self.directory):
            return self
        for f in sorted(os.listdir(self.directory)):
            if not f.lower().endswith(".json"):
                continue
            try:
                c = Clip.load(os.path.join(self.directory, f))
                self.clips[c.name] = c
            except Exception as e:
                print("animation %s failed to load: %s" % (f, e))
        return self

    def play(self, name):
        c = self.clips.get(name)
        if c is None:
            return False
        self.current, self.t = c, 0.0
        return True

    def stop(self):
        self.current = None

    @property
    def playing(self):
        return self.current is not None

    def update(self, dt):
        if self.current is None:
            return
        self.t += dt
        if self.t >= self.current.length:
            if self.current.loop:
                self.t = self.t % max(1e-6, self.current.length)
            else:
                self.current = None

    def weight(self):
        """Fades in at the start and out at the end, so an authored clip never
        snaps on top of the live face."""
        c = self.current
        if c is None:
            return 0.0
        b = max(1e-6, c.blend)
        if c.loop:
            return min(1.0, self.t / b)
        return min(1.0, self.t / b, max(0.0, c.length - self.t) / b)

    def sample(self, target):
        """(value, weight) for one parameter, or None if nothing is driving it."""
        if self.current is None:
            return None
        v = self.current.sample(target, self.t)
        if v is None:
            return None
        return v, self.weight()

    def names(self):
        return sorted(self.clips)

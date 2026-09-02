#!/usr/bin/env python3
"""
agenda.py - the things Echo can actually DO.

Until now Echo talked well and did nothing. Taby's pitch is the useful insight
here: "local AI with a little personality - it is harder to ignore than another
window in the background." The face is what makes a productivity tool stick,
and Echo already has a far better face than anything comparable. What it was
missing was a job.

Tasks, reminders, habits, notes and focus timers, all in one JSON file next to
the code. Deliberately plain: this is a desk companion, not a project planner,
and every field here has to survive being spoken aloud.

Nothing in here talks to an LLM or a screen. brain.py decides WHAT to do,
run.py decides when to SAY something; this only remembers.
"""
import json
import os
import time
import uuid
from datetime import datetime, timedelta

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "agenda.json")

# Reminders fire a little early rather than late - being told at 17:59 that
# something is due at 18:00 is useful; being told at 18:01 is not.
DUE_LEAD = 30.0          # seconds
MAX_PROMPT_TASKS = 12    # what gets injected into every turn, so keep it small


def _now():
    return datetime.now()


def _iso(dt):
    return dt.replace(microsecond=0).isoformat() if dt else None


def _parse(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", ""))
    except (ValueError, TypeError):
        return None


def speak_when(dt, now=None):
    """A due time as someone would say it, not as a computer would print it."""
    if dt is None:
        return ""
    now = now or _now()
    today = now.date()
    delta = (dt.date() - today).days
    tm = dt.strftime("%I:%M %p").lstrip("0").replace(":00 ", " ")
    if delta == 0:
        return "today at %s" % tm
    if delta == 1:
        return "tomorrow at %s" % tm
    if 1 < delta < 7:
        return "%s at %s" % (dt.strftime("%A"), tm)
    return "%s at %s" % (dt.strftime("%d %B").lstrip("0"), tm)


class Agenda:
    def __init__(self, path=DEFAULT_PATH):
        self.path = path
        self.data = {"tasks": [], "habits": [], "notes": []}
        self.load()

    # ------------------------------------------------------------ storage
    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                loaded = json.load(f)
            for k in self.data:
                if isinstance(loaded.get(k), list):
                    self.data[k] = loaded[k]
        except Exception:
            pass          # first run, or a corrupted file: start clean
        return self

    def save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)     # atomic: a crash cannot truncate it
        except Exception as e:
            print("could not save agenda: %s" % e)

    # -------------------------------------------------------------- tasks
    def add_task(self, text, due=None):
        text = (text or "").strip()
        if not text:
            return None
        t = {"id": uuid.uuid4().hex[:8], "text": text, "due": _iso(_parse(due)),
             "done": False, "created": _iso(_now()), "notified": False}
        self.data["tasks"].append(t)
        self.save()
        return t

    def open_tasks(self):
        return [t for t in self.data["tasks"] if not t.get("done")]

    def _match(self, items, text):
        """Loosest sensible match. Spoken instructions never repeat a task
        word for word, so exact matching would fail almost every time."""
        text = (text or "").strip().lower()
        if not text:
            return None
        for t in items:
            if t["text"].lower() == text:
                return t
        for t in items:
            if text in t["text"].lower() or t["text"].lower() in text:
                return t
        words = {w for w in text.split() if len(w) > 3}
        best, score = None, 0
        for t in items:
            tw = {w for w in t["text"].lower().split() if len(w) > 3}
            n = len(words & tw)
            if n > score:
                best, score = t, n
        return best

    def complete_task(self, text):
        t = self._match(self.open_tasks(), text)
        if t:
            t["done"] = True
            t["done_at"] = _iso(_now())
            self.save()
        return t

    def drop_task(self, text):
        t = self._match(self.open_tasks(), text)
        if t:
            self.data["tasks"].remove(t)
            self.save()
        return t

    def due_now(self, now=None):
        """Tasks whose time has come and that have not been announced yet.
        Marking them notified here means a reminder fires exactly once."""
        now = now or _now()
        out = []
        for t in self.open_tasks():
            due = _parse(t.get("due"))
            if due and not t.get("notified") and \
                    (due - now).total_seconds() <= DUE_LEAD:
                t["notified"] = True
                out.append(t)
        if out:
            self.save()
        return out

    # ------------------------------------------------------------- habits
    def add_habit(self, text, cadence="daily"):
        text = (text or "").strip()
        if not text:
            return None
        h = {"id": uuid.uuid4().hex[:8], "text": text, "cadence": cadence,
             "last": None, "streak": 0, "created": _iso(_now())}
        self.data["habits"].append(h)
        self.save()
        return h

    def tick_habit(self, text):
        """Mark a habit done today, and keep the streak. The streak is the
        whole point of tracking a habit, so it is worth getting right: it only
        continues if the previous tick was yesterday."""
        h = self._match(self.data["habits"], text)
        if not h:
            return None
        today = _now().date()
        last = _parse(h.get("last"))
        if last and last.date() == today:
            return h                        # already done today, no double count
        h["streak"] = (h.get("streak", 0) + 1
                       if last and last.date() == today - timedelta(days=1)
                       else 1)
        h["last"] = _iso(_now())
        self.save()
        return h

    def habits_due(self):
        today = _now().date()
        out = []
        for h in self.data["habits"]:
            last = _parse(h.get("last"))
            if not last or last.date() < today:
                out.append(h)
        return out

    # -------------------------------------------------------------- notes
    def add_note(self, text):
        text = (text or "").strip()
        if not text:
            return None
        n = {"id": uuid.uuid4().hex[:8], "text": text, "created": _iso(_now())}
        self.data["notes"].append(n)
        self.save()
        return n

    def recent_notes(self, n=5):
        return list(reversed(self.data["notes"][-n:]))

    # ------------------------------------------------------------- prompt
    def context(self):
        """Injected every turn so it can answer questions about the list
        without a round trip. Kept short - this is paid for on every reply."""
        now = _now()
        bits = []
        tasks = self.open_tasks()
        if tasks:
            bits.append("Their open tasks:")
            for t in tasks[:MAX_PROMPT_TASKS]:
                due = _parse(t.get("due"))
                bits.append("- %s%s" % (t["text"],
                                        (" (due %s)" % speak_when(due, now)) if due else ""))
            if len(tasks) > MAX_PROMPT_TASKS:
                bits.append("- ...and %d more" % (len(tasks) - MAX_PROMPT_TASKS))
        habits = self.data["habits"]
        if habits:
            due = {h["id"] for h in self.habits_due()}
            bits.append("Their habits:")
            for h in habits[:6]:
                bits.append("- %s (%s, streak %d%s)"
                            % (h["text"], h.get("cadence", "daily"),
                               h.get("streak", 0),
                               ", not done today" if h["id"] in due else ", done today"))
        notes = self.recent_notes(3)
        if notes:
            bits.append("Recent notes: " + "; ".join(n["text"] for n in notes))
        if not bits:
            return ""
        return ("\nTHEIR AGENDA (use it to answer questions; do not read the "
                "whole list aloud unless asked):\n" + "\n".join(bits))


class Timer:
    """A single focus timer. One is enough - a desk companion nagging you
    about four concurrent countdowns is the opposite of focus."""

    def __init__(self):
        self.label = ""
        self.ends_at = 0.0
        self.running = False
        self._fired = False

    def start(self, minutes, label="focus"):
        minutes = max(0.1, min(180.0, float(minutes)))
        self.label = (label or "focus").strip()
        self.ends_at = time.time() + minutes * 60.0
        self.running = True
        self._fired = False
        return minutes

    def stop(self):
        was = self.running
        self.running = False
        return was

    @property
    def remaining(self):
        return max(0.0, self.ends_at - time.time()) if self.running else 0.0

    def finished(self):
        """True exactly once, when the timer runs out."""
        if self.running and not self._fired and self.remaining <= 0.0:
            self._fired = True
            self.running = False
            return True
        return False

    def spoken(self):
        r = int(self.remaining)
        if r <= 0:
            return "no timer running"
        if r < 60:
            return "%d seconds left" % r
        return "%d minutes left" % round(r / 60.0)

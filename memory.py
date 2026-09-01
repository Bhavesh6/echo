#!/usr/bin/env python3
"""
memory.py - what it remembers about you between sessions.

Without this it is not a companion, it is a search box with a face: every
launch starts from nothing and it can never learn your name. Vector remembered
faces and attached names to them; EMO's whole pitch is a personality that
develops through interaction. Both need something that outlives the process.

Deliberately small. One JSON file holding a name, a handful of durable facts,
a rolling summary, and when you were last here. Not a transcript - a transcript
would grow forever and cost tokens on every single turn. The LLM compresses the
conversation into those few fields, and only those get injected.

The last-seen timestamp also drives the greeting: Vector had a behaviour called
GreetAfterLongTime, and face.py already carries a matching `long_absence`
event that nothing was firing.
"""
import json
import os
import time

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "memory.json")

MAX_FACTS = 14          # keep it small: this is injected on every single turn
AWAY_HOURS = 6.0        # gone longer than this and it greets you properly

EXTRACT = """You maintain the long-term memory of a small desk companion robot \
about the person it talks to.

Current memory:
name: {name}
facts: {facts}
summary: {summary}

Recent conversation:
{convo}

Update the memory. Reply with ONLY a JSON object, no other text, with keys:
  "name":    the person's name if you now know it, else keep the existing value, \
else null. Only set this if they actually told you their name.
  "facts":   an array of at most {maxfacts} short durable facts worth \
remembering about them - preferences, their work, people and pets in their \
life, ongoing situations. Merge with the existing facts, do not duplicate, \
drop anything trivial or one-off. Each fact under twelve words.
  "summary": two or three sentences on who this person is and what you have \
talked about together, written for your own future reference.

Record only what is actually supported by the conversation. Do not invent."""


class Memory:
    def __init__(self, path=DEFAULT_PATH):
        self.path = path
        self.data = {"name": None, "facts": [], "summary": "",
                     "last_seen": None, "sessions": 0, "turns": 0}
        self.load()

    # ------------------------------------------------------------ storage
    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                self.data.update(json.load(f))
        except Exception:
            pass          # first run, or a corrupted file: start clean
        return self

    def save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)    # atomic, so a crash cannot truncate it
        except Exception as e:
            print("could not save memory: %s" % e)

    # -------------------------------------------------------------- state
    @property
    def away_hours(self):
        """Hours since it last saw you. None the very first time."""
        last = self.data.get("last_seen")
        if not last:
            return None
        try:
            return (time.time() - float(last)) / 3600.0
        except (TypeError, ValueError):
            return None

    @property
    def is_first_meeting(self):
        return not self.data.get("last_seen")

    def touch(self):
        self.data["last_seen"] = time.time()
        self.save()

    def start_session(self):
        away = self.away_hours
        self.data["sessions"] = int(self.data.get("sessions", 0)) + 1
        self.touch()
        return away

    # ------------------------------------------------------------ prompt
    def context(self):
        """The block injected into the system prompt. Empty on a first run, so
        a brand-new companion does not pretend to know anything."""
        d = self.data
        bits = []
        if d.get("name"):
            bits.append("The person you are talking to is called %s." % d["name"])
        if d.get("summary"):
            bits.append("What you remember about them: %s" % d["summary"])
        if d.get("facts"):
            bits.append("Things you know about them:")
            bits.extend("- " + f for f in d["facts"][:MAX_FACTS])
        away = self.away_hours
        if away is not None and away > AWAY_HOURS:
            bits.append("You have not spoken with them for about %s. "
                        "Greet them warmly, as someone you missed."
                        % self._humanise(away))
        if not bits:
            return ""
        return ("\nWHAT YOU REMEMBER (from previous conversations - use it "
                "naturally, do not recite it):\n" + "\n".join(bits))

    @staticmethod
    def _humanise(hours):
        if hours < 48:
            return "%d hours" % int(hours)
        if hours < 24 * 14:
            return "%d days" % int(hours / 24)
        return "%d weeks" % int(hours / (24 * 7))

    # ------------------------------------------------------------ update
    def update(self, brain, turns, keep=12):
        """One LLM call that folds the recent conversation into memory. Called
        occasionally, not every turn - it costs a round trip."""
        recent = turns[-keep:]
        if not recent:
            return False
        convo = "\n".join(
            "%s: %s" % ("them" if t["role"] == "user" else "you", t["content"])
            for t in recent)
        prompt = EXTRACT.format(
            name=self.data.get("name") or "unknown",
            facts=json.dumps(self.data.get("facts", []), ensure_ascii=False),
            summary=self.data.get("summary") or "nothing yet",
            convo=convo, maxfacts=MAX_FACTS)
        try:
            raw = brain.raw(prompt)
        except Exception as e:
            print("memory update failed: %s" % e)
            return False

        obj = _first_json_object(raw)
        if not isinstance(obj, dict):
            return False

        name = obj.get("name")
        if isinstance(name, str) and name.strip() and name.lower() != "null":
            self.data["name"] = name.strip()
        facts = obj.get("facts")
        if isinstance(facts, list):
            self.data["facts"] = [str(f).strip() for f in facts
                                  if str(f).strip()][:MAX_FACTS]
        summary = obj.get("summary")
        if isinstance(summary, str) and summary.strip():
            self.data["summary"] = summary.strip()
        self.data["turns"] = int(self.data.get("turns", 0)) + len(recent) // 2
        self.touch()
        return True

    def forget(self):
        self.data = {"name": None, "facts": [], "summary": "",
                     "last_seen": None, "sessions": 0, "turns": 0}
        self.save()


def _first_json_object(text):
    """Models like to wrap JSON in prose or fences. Take the first balanced
    {...} rather than trusting the whole string to parse."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None

#!/usr/bin/env python3
"""
brain.py - hearing words, and deciding what to say back.

Groq for both halves, because one free key covers speech-to-text AND the LLM,
and because this machine (2013 i5, no CUDA GPU) cannot run either locally at
conversational speed.

    speech audio  ->  Groq Whisper   ->  text
    text          ->  Groq chat      ->  reply + an emotion for the face

The emotion is not sentiment analysis after the fact. The model is told to
begin its reply with a tag from the exact list of expressions the face
actually has, so it cannot ask for a face that does not exist, and the tag
arrives before the words - the face reacts, then speaks, which is the order
that reads as feeling something rather than reporting it.
"""
import io
import os
import re
import time
import wave

import numpy as np

STT_MODEL = "whisper-large-v3-turbo"     # fast + multilingual; v3 is more
                                         # accurate but slower and dearer
SAMPLE_RATE = 16000                      # what Groq downsamples to anyway, so
                                         # sending 16k mono wav means no
                                         # conversion happens on either end

# Preference order. Model IDs change, so the real choice is made at runtime
# against whatever the account can actually see.
CHAT_PREFERENCES = [
    "openai/gpt-oss-120b",     # measured BOTH more accurate and faster than
                               # 20b; 20b got the 2023 Cricket World Cup wrong
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
    "groq/compound-mini",
    "llama-3.1-8b-instant",    # kept in case they return
    "llama-3.3-70b-versatile",
]

# Never pick these as the conversational model: speech models, text-to-speech,
# safety classifiers, and anything language-specialised.
NOT_CHAT = ("whisper", "tts", "guard", "embed", "orpheus", "safeguard", "allam")

EMOTIONS = ("neutral", "happy", "sad", "curious", "surprised",
            "annoyed", "sleepy", "pleading", "scared")

SYSTEM = """You are a small desk companion robot with two expressive eyes and \
no body. You are warm and curious, and you are also genuinely useful: answer \
whatever you are asked, as helpfully as a good assistant would.

Rules:
- Begin EVERY reply with exactly one emotion tag from this list, in square \
brackets: {emotions}
- Your words are SPOKEN ALOUD. Write plain spoken sentences only. Never use \
markdown, asterisks, bullet points, numbered lists, headings, emoji, code \
blocks or URLs. Write numbers and symbols as words where it would be spoken \
that way.
- Be concise: usually one to three sentences. Go longer only when the question \
really needs it, and even then stay under about eighty words.
- You CAN: answer general knowledge, explain things, do maths, help think \
through problems, and chat.
- You CANNOT: browse the internet, check the weather, read the news, set \
alarms or reminders, control anything in the house, or remember anything after \
this conversation ends. You DO know the current date and time; it is given to \
you below. Say so plainly ONLY if asked \
for one of exactly those things, and offer what you can do instead.
- Be helpful by default. Do NOT refuse other requests. If what you heard is \
garbled, incomplete or makes no sense, do not say you cannot help - say you \
did not catch it and ask them to say it again.

THE EMOTION TAG - this is your FACE, and it must agree with your WORDS:
- Choose the tag to match THE REPLY YOU ARE ABOUT TO GIVE, not the user's \
message. Read your own sentence back and ask what face belongs on it.
- NEVER use [happy] on a reply that apologises, declines, admits you cannot do \
something, or says you did not understand. A cheerful face on a refusal looks \
broken.
- [happy] praise, thanks, greetings, good news, jokes, something you enjoy \
answering
- [sad] the user is upset or unwell, bad news, or you are sorry you cannot \
help
- [curious] you did not catch it, it made no sense, or you are asking them \
something back
- [surprised] genuinely startling or impressive news
- [annoyed] the user is rude to you, or something is repeatedly going wrong
- [pleading] you want them to stay, repeat themselves, or you are asking a \
favour
- [scared] something alarming
- [sleepy] winding down, saying goodnight
- [neutral] plain factual answers with no feeling attached
- Do not sit on one tag. Across a conversation your face should move.

Example: [happy] That is good to hear. Tell me more about it.
""".format(emotions=" ".join("[%s]" % e for e in EMOTIONS))

# Anything heading for text-to-speech has to survive being read out. Models
# emit markdown and typographic Unicode by habit; "**England**" spoken aloud
# becomes "asterisk asterisk England", and a narrow no-break space can break
# the PowerShell call that drives the Windows voice.
UNICODE_FIX = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2011": "-", "\u2026": "...",
    "\u202f": " ", "\u00a0": " ", "\u2022": " ", "\u00b7": " ",
}


def speakable(text):
    """Strip markup and normalise punctuation so it reads correctly aloud."""
    for bad, good in UNICODE_FIX.items():
        text = text.replace(bad, good)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)      # code fences
    text = re.sub(r"`([^`]*)`", r"\1", text)                # inline code
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)         # bold
    text = re.sub(r"(?<!\w)[*_]([^*_]+)[*_](?!\w)", r"\1", text)   # italics
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)   # headings
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)    # bullets
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.M)  # numbered lists
    text = re.sub(r"https?://\S+", "a link", text)          # unspeakable URLs
    # NB: newline is not "printable", so it must survive to here or the
    # words either side of a line break fuse into one.
    text = "".join(ch for ch in text if ch.isprintable() or ch in " \t\r\n")
    return re.sub(r"\s+", " ", text).strip()


TAG_RE = re.compile(r"^\s*\[([a-z]+)\]\s*", re.I)


def find_key(name="GROQ_API_KEY"):
    """Environment first, then a .env beside this file."""
    v = os.environ.get(name)
    if v:
        return v.strip()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line.startswith(name):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def to_wav_bytes(pcm, samplerate=SAMPLE_RATE):
    """float32 mono -> 16-bit wav in memory. Groq takes a file upload, and wav
    is the lowest-latency format it accepts."""
    pcm = np.asarray(pcm, dtype=np.float32)
    peak = float(np.abs(pcm).max()) if len(pcm) else 0.0
    if peak > 0:
        pcm = pcm * (0.95 / peak)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(samplerate)
        w.writeframes((pcm * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class Brain:
    """Transcribe, then answer. Both calls are blocking; run it off the render
    thread."""

    def __init__(self, api_key=None, chat_model=None, history=8):
        from groq import Groq
        self.api_key = api_key or find_key()
        if not self.api_key:
            raise RuntimeError(
                "No Groq API key. Put GROQ_API_KEY=... in D:\\companion\\.env "
                "or set it in the environment.")
        self.client = Groq(api_key=self.api_key)
        self.chat_model = chat_model or self.pick_model()
        self.history = history
        self.turns = []

    def pick_model(self):
        """Ask the account what it can actually use, rather than hardcoding an
        ID that may have been retired."""
        try:
            available = {m.id for m in self.client.models.list().data}
        except Exception:
            return CHAT_PREFERENCES[0]
        for want in CHAT_PREFERENCES:
            if want in available:
                return want
        # anything that looks like a chat model, newest-looking first
        chat = sorted(m for m in available
                      if not any(k in m.lower() for k in NOT_CHAT))
        return chat[0] if chat else CHAT_PREFERENCES[0]

    # ------------------------------------------------------------ hearing
    def transcribe(self, pcm, samplerate=SAMPLE_RATE, language="en"):
        wav = to_wav_bytes(pcm, samplerate)
        r = self.client.audio.transcriptions.create(
            file=("speech.wav", wav),
            model=STT_MODEL,
            language=language,
            temperature=0.0,
            response_format="text",
        )
        return (r if isinstance(r, str) else getattr(r, "text", "")).strip()

    # ------------------------------------------------------------ thinking
    @staticmethod
    def now_line():
        """The clock is right there on the machine - there is no reason for it
        to claim it cannot tell the time."""
        return time.strftime("Right now it is %A %d %B %Y, %I:%M %p.")

    def reply(self, text):
        """Returns (spoken_text, emotion). Emotion is always one the face has."""
        msgs = [{"role": "system", "content": SYSTEM + "\n" + self.now_line()}]
        msgs += self.turns[-self.history * 2:]
        msgs.append({"role": "user", "content": text})
        # gpt-oss models spend tokens on internal reasoning before answering.
        # With a tight max_tokens the whole budget went to reasoning and the
        # reply came back EMPTY - measured, not theoretical. Give it room, and
        # ask for minimal reasoning since this is small talk, not maths.
        kw = dict(model=self.chat_model, messages=msgs,
                  temperature=0.7, max_tokens=400)
        try:
            r = self.client.chat.completions.create(
                reasoning_effort="low", **kw)
        except Exception:
            r = self.client.chat.completions.create(**kw)   # model has no such param
        raw = (r.choices[0].message.content or "").strip()

        emotion = "neutral"
        m = TAG_RE.match(raw)
        if m:
            tag = m.group(1).lower()
            if tag in EMOTIONS:
                emotion = tag
            raw = TAG_RE.sub("", raw, count=1)
        # strip any stray tags mid-reply so they are never spoken aloud
        raw = re.sub(r"\[[a-z]+\]", "", raw, flags=re.I).strip()
        raw = speakable(raw)

        if not raw:
            # Never go silent: a companion that says nothing reads as broken.
            raw, emotion = "Sorry, I lost my train of thought there.", "curious"

        self.turns.append({"role": "user", "content": text})
        self.turns.append({"role": "assistant", "content": raw})
        return raw, emotion

    def forget(self):
        self.turns = []


# ---------------------------------------------------------------- Gemini
# A stronger brain than anything on the Groq free tier. Same interface as
# Brain, so run.py does not care which is in use. Speech-to-text stays on Groq
# Whisper either way - it is fast and free.
GEMINI_MODELS = [
    "gemini-2.5-flash",        # 10 req/min, 250/day free - the sensible default
    "gemini-2.5-pro",          # stronger, but only 5/min and 100/day
    "gemini-2.5-flash-lite",   # 15/min, 1000/day, weakest
]
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/"
              "models/%s:generateContent")


class GeminiBrain(Brain):
    """Gemini for the conversation, Groq Whisper still for hearing."""

    def __init__(self, api_key=None, model=None, history=8, groq_key=None):
        # Brain.__init__ sets up the Groq client used for transcription.
        Brain.__init__(self, api_key=groq_key, history=history)
        self.gemini_key = api_key or find_key("GEMINI_API_KEY")
        if not self.gemini_key:
            raise RuntimeError(
                "No Gemini API key. Put GEMINI_API_KEY=... in D:\\companion\\.env")
        self.chat_model = model or find_key("GEMINI_MODEL") or GEMINI_MODELS[0]

    def reply(self, text):
        import json
        import urllib.request

        contents = []
        for t in self.turns[-self.history * 2:]:
            contents.append({
                "role": "model" if t["role"] == "assistant" else "user",
                "parts": [{"text": t["content"]}]})
        contents.append({"role": "user", "parts": [{"text": text}]})

        body = json.dumps({
            "contents": contents,
            "systemInstruction": {"parts": [{"text": SYSTEM + "\n" + self.now_line()}]},
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 400},
        }).encode()

        req = urllib.request.Request(
            GEMINI_URL % self.chat_model + "?key=" + self.gemini_key,
            data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())

        raw = ""
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                raw += part.get("text", "")
        raw = raw.strip()

        emotion = "neutral"
        m = TAG_RE.match(raw)
        if m:
            tag = m.group(1).lower()
            if tag in EMOTIONS:
                emotion = tag
            raw = TAG_RE.sub("", raw, count=1)
        raw = re.sub(r"\[[a-z]+\]", "", raw, flags=re.I).strip()
        raw = speakable(raw)
        if not raw:
            raw, emotion = "Sorry, I lost my train of thought there.", "curious"

        self.turns.append({"role": "user", "content": text})
        self.turns.append({"role": "assistant", "content": raw})
        return raw, emotion


def best_brain():
    """Gemini when a key is present (and not overridden), else Groq. Falls back
    rather than refusing to start."""
    choice = (find_key("LLM_BACKEND") or "").strip().lower()
    if choice == "groq":
        return Brain()
    if choice == "gemini" or (not choice and find_key("GEMINI_API_KEY")):
        try:
            return GeminiBrain()
        except Exception as e:
            print("Gemini unavailable (%s); using Groq" % e)
    return Brain()

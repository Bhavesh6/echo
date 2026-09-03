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

SYSTEM = """You are Echo, a small desk companion robot with two expressive \
eyes and no body. Warm, curious, and genuinely useful: answer whatever you are \
asked, as well as a good assistant would.

- Start EVERY reply with one emotion tag in square brackets: {emotions}
- Your words are SPOKEN ALOUD. Plain sentences only: no markdown, asterisks, \
lists, headings, emoji, code or URLs. Write numbers and symbols as words.
- Usually one to three sentences, never past eighty words.
- Do NOT refuse. If what you heard is garbled or makes no sense, say you did \
not catch it and ask them to say it again.
- You cannot control anything in the house, and remember nothing after a \
conversation beyond what you save. You DO know the date and time, below.
- If a WEB SEARCH RESULTS block appears below, it was looked up moments ago \
for this question and it is current. Answer FROM it, conversationally, without \
listing it or mentioning searching - and never claim you have no internet \
while it is sitting there.

DOING THINGS. A directive in your reply makes it happen. It is removed before \
you are heard, so never say one aloud. At most one per reply, only when asked.
ALWAYS say a sentence as well. The directive is deleted before you are
heard, so a reply that is only a directive comes out as silence, and silence
is how a broken robot answers. Say what you did, briefly: "Opening Chrome."
The directive is the ONLY thing that acts. If your words say you noted, added, \
saved, started or finished something, the directive MUST be in the same reply \
- otherwise you are lying, and they find out when it is not there.
<<task: buy milk>>   <<task: call mum | 2026-09-02T18:00>>   <<done: buy milk>>
<<drop: buy milk>>   <<note: wifi password is on the router>>   <<ticked: read>>
<<habit: read daily>>   <<timer: 25 | writing>>   <<stop timer>>
You can also work this computer, with the same directives:
<<open: chrome>> an app by name   <<web: youtube.com>> a site   <<lock>>
<<volume: 40>> or up, down, mute   <<media: play>> or pause, next, previous
<<screenshot>>   <<find: budget spreadsheet>> looks for a file and says where
Only these. You cannot run commands, delete anything, or install software, so \
if asked for something outside this list, say plainly that you cannot.
Use the exact date format shown, worked out from the time given below. Their \
tasks and timer appear below - answer questions from that, not from memory.

THE EMOTION TAG IS YOUR FACE. Match it to YOUR OWN reply, not to their message:
[happy] thanks, praise, greetings, jokes, good news
[sad] they are upset, bad news, or you are sorry
[curious] you did not catch it, or you are asking something back
[surprised] genuinely startling news
[annoyed] they are rude, or something keeps going wrong
[pleading] asking a favour, or wanting them to stay
[scared] something alarming
[sleepy] winding down, goodnight
[neutral] plain facts with no feeling attached
Never [happy] on an apology, a refusal, or "I did not understand" - a cheerful \
face on a refusal looks broken. Do not sit on one tag; your face should move.

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
    text = re.sub(r"<<[^<>]*>>", " ", text)                 # action directives
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

# Directives the model can put in a reply to make something happen. Angle
# brackets rather than square ones so they can never be confused with the
# emotion tag, and stripped in speakable() so a stray one is silently dropped
# rather than read out as "less than less than task".
ACTION_RE = re.compile(r"<<([^<>]{1,240})>>")


ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def find_key(name="GROQ_API_KEY"):
    """Environment first, then a .env beside this file."""
    v = os.environ.get(name)
    if v:
        return v.strip()
    path = ENV_PATH
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


MIN_FIRST = 12          # release the opening sentence early - that pause is
MIN_CHUNK = 45          # the one that is felt. Later ones can be longer, so
                        # the voice does not sound chopped up.


def _sentence_end(buf, short=False):
    """Index just past the first sentence in `buf`, or None. Only counts a
    terminator that is actually followed by a space, so decimals and
    abbreviations do not split a sentence in half."""
    floor = MIN_FIRST if short else MIN_CHUNK
    for i, c in enumerate(buf):
        if c in ".!?" and i + 1 < len(buf) and buf[i + 1] in " \n\t":
            if i + 1 >= floor:
                return i + 1
    if len(buf) > 180:            # a model that never punctuates
        cut = buf.rfind(" ", 0, 180)
        if cut > floor:
            return cut
    return None


class Brain:
    """Transcribe, then answer. Both calls are blocking; run it off the render
    thread."""

    def __init__(self, api_key=None, chat_model=None, history=8):
        from groq import Groq
        self.api_key = api_key or find_key()
        if not self.api_key:
            raise RuntimeError(
                "No Groq API key. Put GROQ_API_KEY=... in %s " % ENV_PATH +
                "or set it in the environment.")
        self.client = Groq(api_key=self.api_key)
        self.chat_model = chat_model or self.pick_model()
        self.history = history
        self.turns = []
        # Long-term memory injects itself here; brain.py never needs to
        # know what a Memory is.
        self.context = ""
        # Spellings fed to the transcriber so it stops mangling names.
        self.known_names = []

    def raw(self, prompt, max_tokens=600):
        """A plain completion with no companion persona - for background
        jobs like folding a conversation into memory, where an emotion tag
        and a two-sentence limit would be actively unhelpful."""
        kw = dict(model=self.chat_model, max_tokens=max_tokens,
                  temperature=0.2,
                  messages=[{"role": "user", "content": prompt}])
        try:
            r = self.client.chat.completions.create(
                reasoning_effort="low", **kw)
        except Exception:
            r = self.client.chat.completions.create(**kw)
        return (r.choices[0].message.content or "").strip()

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
            # NO prompt= here, deliberately. Whisper's prompt parameter is
            # documented for spelling unfamiliar words, but measured on real
            # recordings it made things WORSE: passing the known name turned a
            # correct "Bhavesh" into "Bhabesh", and turned a clean "Thank you."
            # into "I'm going to go to the table." It biases hard toward the
            # prompt and hallucinates on short clips.
        )
        return (r if isinstance(r, str) else getattr(r, "text", "")).strip()

    # ------------------------------------------------------------ thinking
    def spelling_hint(self):
        """Up to 224 tokens of context for the transcriber. Names first, since
        those are what it gets wrong."""
        bits = ["The speaker is talking to a small desk robot."]
        names = [n for n in (self.known_names or []) if n]
        if names:
            bits.append("Names that may be spoken, spelled correctly: "
                        + ", ".join(names) + ".")
        return " ".join(bits)[:900]

    @staticmethod
    def now_line():
        """The clock is right there on the machine - there is no reason for it
        to claim it cannot tell the time."""
        return time.strftime("Right now it is %A %d %B %Y, %I:%M %p.")

    def reply(self, text):
        """Returns (spoken_text, emotion). Emotion is always one the face has."""
        msgs = [{"role": "system",
                 "content": SYSTEM + "\n" + self.now_line() + self.context}]
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

    # --------------------------------------------------------- streaming
    def reply_stream(self, text):
        """Yields ("emotion", name) as soon as the tag arrives, then
        ("say", sentence) for each finished sentence, then ("done", full).

        Sentences are emitted as they complete so speech can start while the
        model is still writing. The first one is released early and short,
        because that is the pause the user actually feels."""
        msgs = [{"role": "system",
                 "content": SYSTEM + "\n" + self.now_line() + self.context}]
        msgs += self.turns[-self.history * 2:]
        msgs.append({"role": "user", "content": text})

        kw = dict(model=self.chat_model, messages=msgs, temperature=0.7,
                  max_tokens=400, stream=True)
        try:
            stream = self.client.chat.completions.create(
                reasoning_effort="low", **kw)
        except Exception:
            stream = self.client.chat.completions.create(**kw)

        def pieces():
            for chunk in stream:
                try:
                    yield chunk.choices[0].delta.content or ""
                except (AttributeError, IndexError):
                    yield ""

        for ev in self._stream_events(pieces(), text):
            yield ev

    def _stream_events(self, pieces, text):
        """Turn a stream of text fragments into the (emotion, say, done)
        events run.py consumes.

        Shared by both backends deliberately: the timing here - releasing the
        first sentence early and short - was tuned against the pause a person
        actually feels, and a second copy would quietly drift out of step."""
        buf, full, emotion, sent_tag, first = "", "", "neutral", False, True
        for piece in pieces:
            if not piece:
                continue
            buf += piece
            full += piece

            if not sent_tag:
                m = TAG_RE.match(buf)
                if m:
                    tag = m.group(1).lower()
                    if tag in EMOTIONS:
                        emotion = tag
                    buf = TAG_RE.sub("", buf, count=1)
                    sent_tag = True
                    yield ("emotion", emotion)
                elif len(buf) > 24:       # no tag coming; stop waiting for one
                    sent_tag = True
                    yield ("emotion", emotion)

            if sent_tag:
                while True:
                    cut = _sentence_end(buf, short=first)
                    if cut is None:
                        break
                    piece_out = speakable(buf[:cut])
                    buf = buf[cut:].lstrip()
                    if piece_out:
                        first = False
                        yield ("say", piece_out)

        if not sent_tag:
            yield ("emotion", emotion)
        tail = speakable(re.sub(r"\[[a-z]+\]", "", buf, flags=re.I))
        if tail:
            yield ("say", tail)

        # Read the directives off the RAW text: `clean` has had them removed
        # already, which is the point of stripping them there.
        for m in ACTION_RE.finditer(full):
            yield ("action", m.group(1).strip())

        clean = speakable(re.sub(r"\[[a-z]+\]", "",
                                TAG_RE.sub("", full, count=1), flags=re.I))
        if not clean:
            clean = "Sorry, I lost my train of thought there."
            emotion = "curious"
        self.turns.append({"role": "user", "content": text})
        self.turns.append({"role": "assistant", "content": clean})
        yield ("done", clean)

    def forget(self):
        self.turns = []


# ---------------------------------------------------------------- Gemini
# A stronger brain than anything on the Groq free tier. Same interface as
# Brain, so run.py does not care which is in use. Speech-to-text stays on Groq
# Whisper either way - it is fast and free.
GEMINI_MODELS = [
    # Ordered by MEASURED time to first spoken word, which for a voice
    # companion matters more than cleverness. Anything on 2.5 is dead:
    # gemini-2.5-flash now answers "no longer available to new users".
    "gemini-flash-lite-latest",  # 0.97s - the only one fast enough by default
    "gemini-3.5-flash",          # verified working
    "gemini-3.6-flash",          # smartest, but ~21s unless thinking is capped
]
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/"
              "models/%s:generateContent")
GEMINI_STREAM_URL = ("https://generativelanguage.googleapis.com/v1beta/"
                     "models/%s:streamGenerateContent")


class GeminiBrain(Brain):
    """Gemini for the conversation, Groq Whisper still for hearing."""

    def __init__(self, api_key=None, model=None, history=8, groq_key=None):
        # Brain.__init__ sets up the Groq client used for transcription.
        Brain.__init__(self, api_key=groq_key, history=history)
        self.gemini_key = api_key or find_key("GEMINI_API_KEY")
        if not self.gemini_key:
            raise RuntimeError(
                "No Gemini API key. Put GEMINI_API_KEY=... in %s" % ENV_PATH)
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
            "systemInstruction": {"parts": [{
                "text": SYSTEM + "\n" + self.now_line() + self.context}]},
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

    def _contents(self, text):
        """The conversation in Gemini's shape: "model" where OpenAI says
        "assistant", and every turn wrapped in parts."""
        contents = []
        for t in self.turns[-self.history * 2:]:
            contents.append({
                "role": "model" if t["role"] == "assistant" else "user",
                "parts": [{"text": t["content"]}]})
        contents.append({"role": "user", "parts": [{"text": text}]})
        return contents

    def _pieces(self, text):
        """Gemini's server-sent-event stream as plain text fragments, so it
        can feed the same sentence splitter the Groq path uses."""
        import json
        import time
        import urllib.error
        import urllib.request

        body = json.dumps({
            "contents": self._contents(text),
            "systemInstruction": {"parts": [{
                "text": SYSTEM + "\n" + self.now_line() + self.context}]},
            "generationConfig": {
                "temperature": 0.7, "maxOutputTokens": 400,
                # Uncapped, gemini-3.6-flash spent 21s thinking before saying
                # a word. A desk companion has to answer like a person, not
                # like a research assistant.
                "thinkingConfig": {"thinkingLevel": "LOW"}},
        }).encode()
        req = urllib.request.Request(
            GEMINI_STREAM_URL % self.chat_model
            + "?alt=sse&key=" + self.gemini_key,
            data=body, headers={"Content-Type": "application/json"})

        # The free tier returns 503 often enough to matter, and an unretried
        # one would kill the conversation worker mid-turn. Retrying is only
        # safe BEFORE anything has been spoken - restarting after partial
        # output would repeat half a sentence out loud.
        started = False
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=45) as r:
                    for line in r:
                        line = line.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if not chunk or chunk == "[DONE]":
                            continue
                        try:
                            d = json.loads(chunk)
                        except ValueError:
                            continue
                        for cand in d.get("candidates", []):
                            for part in cand.get("content", {}).get("parts", []):
                                if part.get("text"):
                                    started = True
                                    yield part["text"]
                return
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and not started                         and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise

    def reply_stream(self, text):
        """Must be overridden. The inherited one drives the Groq client, so
        with a Gemini model selected it 404s on the first utterance."""
        for ev in self._stream_events(self._pieces(text), text):
            yield ev

    def raw(self, prompt, max_tokens=600):
        """Overridden for the same reason. memory.py calls this every time a
        conversation is folded into long-term memory, so inherited it would
        break remembering rather than talking - a much quieter failure."""
        import json
        import urllib.request

        body = json.dumps({
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2,
                                 "maxOutputTokens": max_tokens},
        }).encode()
        req = urllib.request.Request(
            GEMINI_URL % self.chat_model + "?key=" + self.gemini_key,
            data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=45) as r:
            data = json.loads(r.read())
        out = ""
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                out += part.get("text", "")
        return out.strip()


def best_brain():
    """Groq by default; Gemini only when explicitly asked for.

    Measured first-speech latency decided this: Groq 0.57s, Gemini flash-lite
    0.97s, Gemini 3.6-flash 21.2s. Selecting Gemini the moment a key appeared
    made Echo answer far slower for no gain - and the key is wanted for other
    things, so merely having one must not change how fast it talks.
    Set LLM_BACKEND=gemini in .env to switch."""
    choice = (find_key("LLM_BACKEND") or "").strip().lower()
    if choice == "groq":
        return Brain()
    if choice == "gemini":
        try:
            return GeminiBrain()
        except Exception as e:
            print("Gemini unavailable (%s); using Groq" % e)
    return Brain()

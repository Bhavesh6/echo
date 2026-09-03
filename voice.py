#!/usr/bin/env python3
"""
voice.py - audio out, and the level signal that drives the eyes.

The one thing that matters here: the eye amplitude is measured INSIDE the
playback callback, on the exact block being handed to the sound card - not
when audio arrives from the network or is loaded from disk. Measuring on
arrival makes the face lead the voice by however deep the buffer is, and a
face that moves before its own voice reads as badly dubbed video. It is the
single easiest way to destroy the illusion.

Two voices, same interface, so the rest of the program does not care which:
    SapiVoice        - Windows' built-in TTS. Free, offline, no key. Good
                       enough to build and verify the whole chain against.
    ElevenLabsVoice  - the real one. Asks for pcm_16000 so the samples arrive
                       raw and no decoding is needed to get the envelope,
                       and because 16 kHz is the ReSpeaker's native rate.

Neither one knows anything about faces. main() wires level -> face.
"""
import collections
import os
import queue
import subprocess
import tempfile
import threading
import time
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000          # native rate of the ReSpeaker Lite; also whisper's
BLOCKSIZE = 256              # ~16 ms per block: fine-grained enough that the
                             # eyes track syllables rather than whole words
LEVEL_GAIN = 3.2             # speech rms is small; scale it into 0..1
LEVEL_FLOOR = 0.012          # below this treat as silence, so room noise and
                             # inter-word gaps do not jitter the face


def output_device(name_part=None, api="WASAPI"):
    """Index of an output device by name fragment, or None for the default."""
    if not name_part:
        return None
    for i, d in enumerate(sd.query_devices()):
        a = sd.query_hostapis(d["hostapi"])["name"]
        if name_part.lower() in d["name"].lower() and api in a \
                and d["max_output_channels"] > 0:
            return i
    return None


class Player:
    """Streams float32 mono PCM and exposes `level` (0..1) for the current
    block. Non-blocking: `play()` returns immediately."""

    def __init__(self, samplerate=SAMPLE_RATE, device=None, blocksize=BLOCKSIZE):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self._want = device            # None means "whatever is default now"
        self._buf = np.zeros(0, dtype=np.float32)
        self._pos = 0
        self._lock = threading.Lock()
        self._level = 0.0
        self._scale = 1.0
        self._opened_name = None
        self._last_reopen = 0.0
        self.reopens = 0
        self._open()

    @staticmethod
    def _device_name(idx):
        try:
            if idx is None:
                idx = sd.default.device[1]
            return sd.query_devices(idx)["name"]
        except Exception:
            return None

    def _open(self):
        self.stream = sd.OutputStream(
            samplerate=self.samplerate, channels=1, dtype="float32",
            blocksize=self.blocksize, device=self._want,
            callback=self._callback)
        self.stream.start()
        self._opened_name = self._device_name(self._want)

    def healthy(self):
        """False when the slot we opened no longer holds the same device.

        A dead OUTPUT cannot be spotted from the signal the way a dead input
        can: it accepts everything written to it and discards it silently, so
        the level meter keeps moving while the room stays quiet. The device
        identity is the only thing that actually changes.
        """
        name = self._device_name(self._want)
        return name is None or self._opened_name is None or \
            name == self._opened_name

    def reopen(self):
        now = time.time()
        if now - self._last_reopen < 5.0:
            return False
        self._last_reopen = now
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        try:
            self._open()
        except Exception:
            return False
        self.reopens += 1
        return True

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            chunk = self._buf[self._pos:self._pos + frames]
            self._pos = min(self._pos + frames, len(self._buf))
        if len(chunk) < frames:
            chunk = np.concatenate(
                [chunk, np.zeros(frames - len(chunk), dtype=np.float32)])
        outdata[:, 0] = chunk
        # Measured here, on the block actually going to the speaker.
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        self._level = 0.0 if rms < LEVEL_FLOOR else min(1.0, rms * LEVEL_GAIN)

    @property
    def level(self):
        return self._level

    @property
    def busy(self):
        with self._lock:
            return self._pos < len(self._buf)

    def play(self, pcm):
        pcm = np.asarray(pcm, dtype=np.float32)
        peak = float(np.abs(pcm).max()) if len(pcm) else 0.0
        self._scale = (0.90 / peak) if peak > 0 else 1.0
        with self._lock:
            self._buf, self._pos = pcm * self._scale, 0

    def enqueue(self, pcm):
        """Append to whatever is already playing, for streamed speech. The
        gain from the FIRST piece is reused for the rest of the utterance -
        normalising each sentence on its own peak would make quiet sentences
        suddenly loud."""
        pcm = np.asarray(pcm, dtype=np.float32)
        with self._lock:
            playing = self._pos < len(self._buf)
            if not playing:
                peak = float(np.abs(pcm).max()) if len(pcm) else 0.0
                self._scale = (0.90 / peak) if peak > 0 else 1.0
                self._buf, self._pos = pcm * self._scale, 0
            else:
                rest = self._buf[self._pos:]
                self._buf = np.concatenate([rest, pcm * self._scale])
                self._pos = 0

    def stop(self):
        with self._lock:
            self._pos = len(self._buf)

    def close(self):
        try:
            self.stream.stop(); self.stream.close()
        except Exception:
            pass


def load_wav(path):
    """Mono float32 at the file's own rate."""
    with wave.open(path, "rb") as w:
        n, ch, sw, sr = (w.getnframes(), w.getnchannels(),
                         w.getsampwidth(), w.getframerate())
        raw = w.readframes(n)
    dt = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    a = np.frombuffer(raw, dtype=dt).astype(np.float32)
    a /= float(2 ** (8 * sw - 1)) if sw > 1 else 128.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def resample(a, src, dst):
    """Linear resample. Good enough for speech, and avoids a scipy dependency."""
    if src == dst or len(a) == 0:
        return a
    n = int(round(len(a) * dst / float(src)))
    return np.interp(np.linspace(0, len(a) - 1, n),
                     np.arange(len(a)), a).astype(np.float32)


class SapiVoice:
    """Windows' built-in speech synthesiser, via PowerShell to a wav file.
    Not pretty, but it is free, offline, and lets the whole audio -> face
    chain be built and tested before any API key exists."""

    name = "sapi"

    def __init__(self, voice=None):
        self.voice = voice

    def pcm(self, text):
        path = os.path.join(tempfile.gettempdir(), "companion_tts.wav")
        pick = ('$s.SelectVoice("%s");' % self.voice) if self.voice else ""
        script = (
            'Add-Type -AssemblyName System.Speech;'
            '$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;'
            + pick +
            '$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo('
            '%d, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,'
            ' [System.Speech.AudioFormat.AudioChannel]::Mono);'
            '$s.SetOutputToWaveFile("%s", $f);'
            '$s.Speak(%s);'
            '$s.Dispose();' % (SAMPLE_RATE, path, _ps_quote(text)))
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                        "-Command", script],
                       check=True, capture_output=True)
        a, sr = load_wav(path)
        return resample(a, sr, SAMPLE_RATE)


def _ps_quote(s):
    return "'" + str(s).replace("'", "''") + "'"


class ElevenLabsVoice:
    """ElevenLabs TTS, asking for raw pcm_16000.

    Raw PCM rather than mp3 for two reasons: the envelope that drives the eyes
    is then available without decoding anything, and 16 kHz matches the
    ReSpeaker's native rate so nothing resamples anywhere in the chain.
    """

    name = "elevenlabs"
    # Free-tier keys cannot use Voice LIBRARY voices through the API - only
    # the built-in ones. Verified working on this account: Bella, Adam,
    # Antoni. Rachel and library voices return 402 paid_plan_required.
    DEFAULT_VOICE = "EXAVITQu4vr4xnSDxMaL"      # Bella - warm, suits a small robot
    ALTERNATIVES = {"bella": "EXAVITQu4vr4xnSDxMaL",
                    "adam": "pNInz6obpgDQGcFmaJgB",
                    "antoni": "ErXwobaYiN019PkySvjV"}
    MODEL = "eleven_flash_v2_5"                  # ~75 ms, the low-latency one

    def __init__(self, api_key=None, voice_id=None, model=None):
        self.api_key = api_key or find_api_key()
        self.voice_id = (voice_id or find_api_key("ELEVENLABS_VOICE_ID")
                         or self.DEFAULT_VOICE)
        self.model = model or self.MODEL
        if not self.api_key:
            raise RuntimeError(
                "No ElevenLabs API key. Set ELEVENLABS_API_KEY, or put it in "
                "a file named .env next to this one as ELEVENLABS_API_KEY=...")

    def pcm(self, text):
        import json
        import urllib.request
        url = ("https://api.elevenlabs.io/v1/text-to-speech/%s/stream"
               "?output_format=pcm_%d" % (self.voice_id, SAMPLE_RATE))
        body = json.dumps({"text": text, "model_id": self.model}).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"xi-api-key": self.api_key,
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
        return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)


def find_api_key(name="ELEVENLABS_API_KEY"):
    """Environment first, then a .env beside this file. The key is never
    hardcoded and .env is gitignored."""
    key = os.environ.get(name)
    if key:
        return key.strip()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line.startswith(name):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def best_voice(prefer_elevenlabs=True):
    """ElevenLabs when a key is available, otherwise SAPI so the program always
    runs. Force one with VOICE_BACKEND=sapi (or elevenlabs) in .env - handy
    while debugging, since ElevenLabs free characters are finite and test
    sentences burn through them fast."""
    choice = (find_api_key("VOICE_BACKEND") or "").strip().lower()
    if choice == "sapi":
        return SapiVoice()
    if choice == "elevenlabs":
        return ElevenLabsVoice()
    if prefer_elevenlabs and find_api_key():
        try:
            return ElevenLabsVoice()
        except Exception as e:
            print("ElevenLabs unavailable (%s); using Windows TTS" % e)
    return SapiVoice()


class Speaker:
    """A voice plus a player. `say()` synthesises off the main thread and
    reports progress through callbacks, so the render loop never stalls."""

    def __init__(self, voice=None, device=None,
                 on_start=None, on_end=None):
        self.voice = voice or best_voice()
        self.player = Player(device=device)
        self.on_start, self.on_end = on_start, on_end
        self._jobs = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def say(self, text):
        self._jobs.put(("say", text))

    def stream(self, text):
        """Speak this piece, appending to anything already playing."""
        self._jobs.put(("stream", text))

    def beep(self, name, force=False):
        """One non-verbal sound, played immediately - these are punctuation,
        so they are dropped rather than queued if speech is already going.
        Cozmo's audio lead: "if you have voice-over for every single thing he
        does, it totally gets old".

        `force` is for the wake chirp, and only that. Dropping it makes being
        heard indistinguishable from being ignored, which is the one thing the
        sound exists to tell you - so instead it interrupts. That is also the
        correct behaviour on its own terms: someone addressing you by name
        while you are mid-sentence wants you to stop talking and listen.
        """
        try:
            import sounds
        except Exception:
            return
        pcm = sounds.get(name)
        if pcm is None:
            return
        if self.player.busy:
            if not force:
                return
            self.player.stop()
        self.player.play(pcm)

    def _worker(self):
        while True:
            kind, text = self._jobs.get()
            try:
                pcm = self.voice.pcm(text)
            except Exception as e:
                # A present-but-unusable key would otherwise make it silent on
                # every single sentence. Fall back to the built-in voice and
                # keep talking - being heard matters more than sounding good.
                print("TTS failed (%s): %s" % (self.voice.name, e))
                if not isinstance(self.voice, SapiVoice):
                    print("falling back to the Windows voice")
                    self.voice = SapiVoice()
                    try:
                        pcm = self.voice.pcm(text)
                    except Exception as e2:
                        print("fallback also failed: %s" % e2)
                        continue
                else:
                    continue
            if self.on_start:
                self.on_start(text)
            if kind == "stream":
                self.player.enqueue(pcm)
            else:
                self.player.play(pcm)
                while self.player.busy:
                    threading.Event().wait(0.02)
            if self.on_end:
                self.on_end(text)

    @property
    def level(self):
        return self.player.level

    @property
    def busy(self):
        return self.player.busy

    def close(self):
        self.player.close()


# ---------------------------------------------------------------- input
# Mic capture with a simple energy VAD. Same rule as playback: the level is
# computed in the audio callback, so what the face reacts to is what the mic
# actually just heard.
#
# The noise floor is LEARNED rather than fixed, because a fixed threshold is
# wrong in every room. Speech is declared when the level runs well above that
# floor for a few blocks, and dropped only after it stays below for longer -
# the asymmetry (fast to start, slow to stop) stops it flapping on and off
# during the short gaps inside normal speech.

VAD_START = 5.0          # level must exceed floor * this to count as speech
VAD_STOP = 2.2           # ...and fall below floor * this to stop
VAD_ATTACK = 5           # consecutive blocks above (~80 ms) before speech
                         # starts - long enough that a key click or a knock
                         # on the desk cannot trigger it
VAD_RELEASE = 70         # consecutive blocks below (~1.1 s) before speech is
                         # considered finished. 420 ms was far too eager: it
                         # ended the turn on every breath, so one sentence
                         # arrived as three fragments and each transcribed to
                         # gibberish. Assistants generally wait about a second.
MIC_GAIN = 4.0

# The floor is adaptive, which means in a QUIET room it falls close to zero and
# then any faint sound is several times the floor. An absolute minimum is what
# actually stops that: real speech at a normal distance sits well above this,
# ordinary room tone does not.
VAD_ABS_MIN = 0.022

# Keep this much audio from BEFORE speech was detected. VAD_ATTACK means we
# only notice ~80 ms in, and whisper needs the start of the first word or it
# guesses. Cheap insurance against clipped openings.
PRE_ROLL_BLOCKS = 40         # ~0.64 s
MAX_UTTERANCE = 20.0         # seconds; stops a stuck VAD eating all memory


def _find_wake_model():
    """Any .onnx/.tflite dropped in the wake/ folder becomes the wake word.
    This is how a trained "hey echo" arrives without touching code."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), WAKE_MODEL_DIR)
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".onnx", ".tflite")):
            return os.path.join(d, f)
    return None


# A live microphone always carries a noise floor. A run of EXACT zeros is
# never a quiet room - it is a stream that is no longer connected to anything.
DEAD_SECONDS = 3.0
DEAD_BLOCKS = int(DEAD_SECONDS * SAMPLE_RATE / BLOCKSIZE)
REOPEN_EVERY = 5.0           # never thrash the driver faster than this


def input_device(name_part="respeaker", api="WASAPI"):
    for i, d in enumerate(sd.query_devices()):
        a = sd.query_hostapis(d["hostapi"])["name"]
        if name_part and name_part.lower() in d["name"].lower()                 and api in a and d["max_input_channels"] > 0:
            return i
    return None


class Listener:
    """Live mic level plus is-someone-talking. No transcription - this is the
    layer that makes the face visibly react to your voice."""

    def __init__(self, device="respeaker", samplerate=SAMPLE_RATE,
                 blocksize=BLOCKSIZE):
        # Keep the NAME, not just the index. Indices are positions in a
        # list that Windows reorders whenever a virtual audio device comes or
        # goes, so an index cached at startup can stop meaning this device.
        self.device_name = device if isinstance(device, str) else None
        if isinstance(device, str):
            device = input_device(device)
        self.device = device
        self._dead = 0
        self._last_reopen = 0.0
        self._reopens = 0
        self.level = 0.0
        self.speaking = False
        self.wake = None            # attach a WakeWord to gate on a phrase
        self.floor = 0.004
        self.threshold = VAD_ABS_MIN
        self.sensitivity = 1.0      # >1 = harder to trigger, <1 = easier.
                                    # Adjustable live; the right value depends
                                    # on the room and how far away you sit.
        self.muted = False          # held True while WE are talking
        self.forced = False         # held True while push-to-talk is pressed
        self._release = False       # set once, to close a held utterance
        self._above = 0
        self._below = 0
        self._pre = collections.deque(maxlen=PRE_ROLL_BLOCKS)
        self._utt = []
        self._ready = None
        self._ulock = threading.Lock()
        self.stream = sd.InputStream(
            samplerate=samplerate, channels=1, dtype="float32",
            blocksize=blocksize, device=device, callback=self._callback)
        self.stream.start()

    def _callback(self, indata, frames, time_info, status):
        block = indata[:, 0].copy()
        # Watch for the stream going dead. Only while unmuted: we deliberately
        # zero the input while speaking, and that is not a fault.
        if not self.muted:
            self._dead = 0 if np.abs(block).max() > 0.0 else self._dead + 1
        if self.wake is not None and not self.muted:
            self.wake.feed(block)
        rms = float(np.sqrt(np.mean(block ** 2)))
        was_speaking = self.speaking
        if self.muted:
            # We are speaking. Without a speaker in the ReSpeaker's own jack
            # there is no echo cancellation, so the mic would hear us and we
            # would interrupt ourselves forever. Keep adapting the floor, but
            # report nothing.
            self.level, self.speaking = 0.0, False
            self._above = self._below = 0
            return
        # Track the quiet background. Rises slowly, falls fast, so a burst of
        # speech cannot drag the floor up and deafen us to the rest of it.
        if rms < self.floor:
            self.floor += (rms - self.floor) * 0.25
        else:
            self.floor += (rms - self.floor) * 0.0015
        self.floor = max(self.floor, 1e-5)

        self.level = min(1.0, rms * MIC_GAIN)
        # Must clear BOTH the adaptive floor and the absolute minimum.
        start = max(self.floor * VAD_START, VAD_ABS_MIN * self.sensitivity)
        stop = max(self.floor * VAD_STOP, VAD_ABS_MIN * 0.6 * self.sensitivity)
        self.threshold = start
        if rms > start:
            self._above += 1
            self._below = 0
        elif rms < stop:
            self._below += 1
            self._above = 0
        if not self.speaking and self._above >= VAD_ATTACK:
            self.speaking = True
        elif self.speaking and self._below >= VAD_RELEASE:
            self.speaking = False

        # Push-to-talk overrides the detector completely. While the key is
        # held every block is kept whatever its level, so a quiet or hesitant
        # sentence cannot be cut in half by the VAD - which is the whole point
        # of holding a button instead of trusting a threshold. Releasing ends
        # the utterance on the very next block rather than waiting out
        # VAD_RELEASE, because letting go is a far clearer end-of-speech
        # signal than a second of silence.
        if self.forced:
            self.speaking = True
            self._above = self._below = 0
        elif self._release:
            self._release = False
            self.speaking = False

        # Keep the audio itself, not just its level - this is what gets
        # transcribed.
        if self.speaking:
            if not was_speaking:
                self._utt = list(self._pre)      # start with the pre-roll
            self._utt.append(block)
            if len(self._utt) * len(block) > MAX_UTTERANCE * self.stream.samplerate:
                self.speaking = False
        else:
            self._pre.append(block)
            if was_speaking and self._utt:
                with self._ulock:
                    self._ready = np.concatenate(self._utt)
                self._utt = []

    def healthy(self):
        """False once the stream has delivered nothing but exact zeros for
        DEAD_SECONDS. Checked from the main loop rather than acted on in the
        audio callback, because reopening a stream from inside its own
        callback deadlocks."""
        return self._dead < DEAD_BLOCKS

    def reopen(self):
        """Find the device again by name and restart the stream.

        Returns True if it actually reopened. Rate-limited: a genuinely
        unplugged microphone would otherwise be reopened sixty times a second
        for as long as the program runs.
        """
        now = time.time()
        if now - self._last_reopen < REOPEN_EVERY:
            return False
        self._last_reopen = now
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        idx = self.device
        if self.device_name:
            found = input_device(self.device_name)
            if found is not None:
                idx = found
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=BLOCKSIZE, device=idx, callback=self._callback)
            self.stream.start()
        except Exception:
            return False
        self.device = idx
        self._dead = 0
        self._reopens += 1
        return True

    def press(self):
        """Push-to-talk pressed: discard anything in flight and capture from
        the pre-roll, so the first syllable is not clipped."""
        self.restart_utterance()
        self.forced = True

    def release(self):
        """Push-to-talk released: close the utterance on the next block."""
        self.forced = False
        self._release = True

    def restart_utterance(self):
        """Throw away whatever has been captured so far and begin again from
        the pre-roll. Called when the wake phrase fires, so anything said
        before being addressed does not end up in the request."""
        with self._ulock:
            self._ready = None
        self._utt = list(self._pre)

    def take_utterance(self):
        """The audio of the last completed utterance, once, or None."""
        with self._ulock:
            u, self._ready = self._ready, None
        return u

    def close(self):
        try:
            self.stream.stop(); self.stream.close()
        except Exception:
            pass


# ------------------------------------------------------------- wake word
# Anki put trigger-word detection at priority 2 in Vector's behaviour tree -
# only physical safety outranked hearing its name. Without it a companion
# reacts to the television, to other people, to you muttering at your desk.
# openWakeWord only ships six phrases and "echo" is not one of them, so until
# a custom model is trained the built-in stand-in is used. Drop a trained
# model at wake/hey_echo.onnx (or set WAKE_MODEL in .env) and it takes over
# with no code change.
WAKE_MODEL_DIR = "wake"
WAKE_PHRASE = "hey_jarvis"   # also: alexa, hey_mycroft, hey_rhasspy
WAKE_THRESHOLD = 0.28        # 0..1. Measured on real recordings from this
                             # mic: a clear 'hey jarvis' scores 0.92, one run
                             # into the next words scores 0.36, and ordinary
                             # speech scores 0.001. 0.5 missed the second; at
                             # 0.28 both fire with a 250x margin over noise.
WAKE_FRAME = 1280            # samples openWakeWord expects (80 ms at 16 kHz)
WAKE_COOLDOWN = 2.0          # seconds before it can fire again


class WakeWord:
    """Listens for a spoken trigger phrase. Fed from the mic callback, but all
    the work happens on a worker thread so the audio path stays cheap."""

    def __init__(self, phrase=None, threshold=None, framework="onnx"):
        from openwakeword.model import Model
        self.phrase = phrase or find_api_key("WAKE_PHRASE") or WAKE_PHRASE
        self.threshold = float(find_api_key("WAKE_THRESHOLD")
                               or threshold or WAKE_THRESHOLD)
        # A custom model file wins over the built-in phrases.
        custom = find_api_key("WAKE_MODEL") or _find_wake_model()
        if custom:
            self.phrase = os.path.splitext(os.path.basename(custom))[0]
            self.model = Model(wakeword_models=[custom],
                               inference_framework=(
                                   "tflite" if custom.endswith(".tflite")
                                   else "onnx"))
            self.custom = custom
        else:
            self.model = Model(wakeword_models=[self.phrase],
                               inference_framework=framework)
            self.custom = None
        self.score = 0.0
        self.enabled = True
        self._pending = False
        self._last = 0.0
        self._buf = np.zeros(0, dtype=np.float32)
        # ~4 s of audio. The worker runs about 15x faster than realtime so
        # it should never fill, but dropping is still better than blocking
        # the audio callback if the machine stalls.
        self._q = queue.Queue(maxsize=256)
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def feed(self, block):
        """Called from the audio callback. Must stay cheap - just buffer."""
        if not self.enabled:
            return
        try:
            self._q.put_nowait(block)
        except queue.Full:
            pass          # dropping a frame is better than stalling audio

    def _worker(self):
        import time as _t
        while True:
            block = self._q.get()
            self._buf = np.concatenate([self._buf, block])
            while len(self._buf) >= WAKE_FRAME:
                frame = self._buf[:WAKE_FRAME]
                self._buf = self._buf[WAKE_FRAME:]
                pcm16 = (np.clip(frame, -1, 1) * 32767).astype(np.int16)
                try:
                    out = self.model.predict(pcm16)
                except Exception:
                    continue
                score = float(max(out.values())) if out else 0.0
                self.score = score
                now = _t.time()
                if score >= self.threshold and now - self._last > WAKE_COOLDOWN:
                    self._last = now
                    with self._lock:
                        self._pending = True

    def take(self):
        """True once per detection."""
        with self._lock:
            hit, self._pending = self._pending, False
        return hit

#!/usr/bin/env python3
"""
run.py - drives Echo.

Two drivers today:
  KEYBOARD - you are the voice loop, poke it by hand
  DEMO     - a scripted conversation on a timer, so you can watch it behave
             with nobody touching it (press D)

Neither one is the real driver. Step 2 replaces DemoDriver with the
ElevenLabs loop, calling exactly the same face.event() / set_mode() /
set_amplitude() methods. Nothing in face.py changes when that happens.

    python run.py

Keys
  modes      1 sleeping  2 idle  3 listening  4 thinking  5 speaking
  emotions   Q neutral  W happy  E sad  R curious  T surprised  Y annoyed  U sleepy
             Z pleading (aww)   X scared
  events     A wake_word   S not_understood   D(hold) --  F compliment
             G affection   H greeting         J abuse   K long_absence
             N startled    M want
  speech     V speak a line   C stop speaking
  mic        M mute/unmute   [ more sensitive   ] less sensitive
  wake       L  turn the wake word off/on (open mic)
  anims      P  re-read animations/*.json without restarting

  pixels     0  LED / smooth      - and =  coarser / finer
  talk       SPACE (hold) push-to-talk - no wake phrase needed
  other      B blink       TAB demo on/off    I info overlay
             O fullscreen  ESC quit

Talk to it and it answers: Groq Whisper transcribes you, an LLM writes a short
reply and picks the expression, and it speaks back. Needs GROQ_API_KEY in a
.env file next to this one. Without a key the face and the demo still run,
just without understanding. Speech uses ElevenLabs if ELEVENLABS_API_KEY is
present, otherwise Windows' built-in voice. Either way the eyes ride the real
audio envelope.
"""
import os
import random
import re
import sys
import time

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session.log")


def save_utterance(pcm, keep=6):
    """Keep the last few utterances as wav files so the audio can be heard,
    not just guessed at from numbers."""
    try:
        import wave
        import numpy as _np
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")
        os.makedirs(d, exist_ok=True)
        old = sorted(os.listdir(d))
        for f in old[:max(0, len(old) - keep + 1)]:
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass
        path = os.path.join(d, time.strftime("%H%M%S") + ".wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((_np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    except Exception:
        pass


def log(msg):
    """Append-and-flush, so the file is readable while the app is still up.
    This is how anyone (including me) can tell what it actually heard."""
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write("%s  %s" % (time.strftime("%H:%M:%S"), msg) + chr(10))
            f.flush()
    except Exception:
        pass

import pygame

import agenda as agenda_mod
import search
import face as face_mod
from face import Face, FPS

import queue
import threading

try:
    import voice as voice_mod
except Exception as _e:                      # keep the face usable without audio
    voice_mod = None
    print("voice disabled (%s)" % _e)

try:
    import anim as anim_mod
except Exception as _e:
    anim_mod = None
    print("animations disabled (%s)" % _e)

try:
    import memory as memory_mod
except Exception as _e:
    memory_mod = None
    print("memory disabled (%s)" % _e)

try:
    import brain as brain_mod
except Exception as _e:                      # no key, no network: still runs
    brain_mod = None
    print("brain disabled (%s)" % _e)


class Conversation:
    """Runs transcribe+reply off the render thread. Results come back through
    a queue so the face is only ever touched from the main loop."""

    def __init__(self, brain, location=None, agenda=None, timer=None):
        self.brain = brain
        self.agenda = agenda
        self.timer = timer
        # Where the person is, for "nearest X" questions. There is no way to
        # infer this and guessing a city would be worse than admitting it, so
        # it comes from LOCATION in .env or stays None.
        self.location = location
        self.jobs = queue.Queue()
        self.results = queue.Queue()
        self.working = False
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, pcm):
        self.working = True
        self.jobs.put(pcm)

    def _worker(self):
        while True:
            pcm = self.jobs.get()
            try:
                # What did we actually send? A pinned meter does not prove
                # clipping; the sample values do.
                import numpy as _np
                peak = float(_np.abs(pcm).max()) if len(pcm) else 0.0
                clipped = float((_np.abs(pcm) > 0.99).mean()) if len(pcm) else 0.0
                rms = float(_np.sqrt(_np.mean(pcm * pcm))) if len(pcm) else 0.0
                log("audio: %.2fs  peak %.3f  rms %.4f  clipped %.2f%%%s"
                    % (len(pcm) / 16000.0, peak, rms, clipped * 100,
                       "   <-- CLIPPING, transcription will be garbled"
                       if (clipped > 0.0005 or peak >= 0.99) else ""))
                save_utterance(pcm)
                if peak < MIN_UTTERANCE_PEAK or rms < MIN_UTTERANCE_RMS:
                    log("nothing audible in that (peak %.3f rms %.4f) - not "
                        "transcribing" % (peak, rms))
                    self.results.put(("nothing", None))
                    continue
                heard = self.brain.transcribe(pcm)
                if not heard.strip():
                    log("transcription was empty")
                    self.results.put(("nothing", None))
                    continue
                # Noise transcribes to stray fragments. Do not hand those to
                # the LLM - it answers a non-question with a refusal.
                if len(heard.strip()) < 4 or len(heard.split()) < 2:
                    log("too short to be a sentence, ignoring: %r" % heard)
                    self.results.put(("nothing", None))
                    continue
                log("you said: %s" % heard)

                # Some questions cannot be answered from what the model knows -
                # today's date, the weather, where the nearest anything is. Look
                # those up and put the results in the system prompt for THIS
                # turn only, then put the prompt back. Injecting into the system
                # prompt rather than into the user's words keeps the search out
                # of the conversation history, so it cannot confuse the next
                # turn or end up folded into long-term memory.
                extra = ""
                try:
                    extra = search.context_for(heard, self.location)
                except Exception as e:
                    log("search failed: %s" % e)
                # The agenda goes in fresh every turn rather than being folded
                # into brain.context once: it changes as you talk, and a list
                # that is one exchange out of date is worse than no list.
                if self.agenda is not None:
                    try:
                        extra += self.agenda.context()
                        if self.timer is not None and self.timer.running:
                            extra += ("\nA timer is running for %s: %s."
                                      % (self.timer.label, self.timer.spoken()))
                    except Exception as e:
                        log("agenda context failed: %s" % e)
                if extra:
                    log("looking it up: %s" % extra.strip().splitlines()[0][:70])
                # Stream it: the emotion tag arrives first so the face can
                # react immediately, and each finished sentence is spoken
                # while the rest is still being written.
                import time as _t
                t0 = _t.perf_counter()
                said_any = False
                saved_ctx = self.brain.context
                if extra:
                    self.brain.context = saved_ctx + extra
                for kind, val in self.brain.reply_stream(heard):
                    if kind == "emotion":
                        self.results.put(("emotion", (heard, val)))
                    elif kind == "say":
                        if not said_any:
                            log("first words at %.2fs" % (_t.perf_counter() - t0))
                            said_any = True
                        self.results.put(("say", val))
                    elif kind == "action":
                        self.results.put(("action", val))
                    elif kind == "done":
                        log("reply: %s" % val)
                        self.results.put(("done", val))
                self.brain.context = saved_ctx
            except Exception as e:
                log("brain error: %s" % e)
                self.results.put(("error", str(e)))
            finally:
                self.working = False


DEMO_LINES = [
    "Oh, hello there. I was just looking around.",
    "Yes? I am listening.",
    "That is interesting. Tell me more about it.",
    "I like talking with you.",
    "Sorry, I did not understand that one.",
]


class DemoDriver:
    """A scripted conversation, looping. Exists to prove the face behaves
    without a human at the keyboard - the same reason Anki could leave a Vector
    on a desk and have it look alive."""

    SCRIPT = [
        # (seconds to wait, mode or None, event or None)
        (6.0, "idle", None),
        (0.0, None, "sound"),          # something moved in the room
        (5.0, None, None),
        (0.0, "listening", "wake_word"),   # 1st beat: heard my name
        (2.4, None, "speech_start"),
        (2.0, "thinking", "speech_end"),   # 2nd beat: working on it
        (1.6, "speaking", "understood"),   # 3rd beat: got it
        (4.5, "idle", None),
        (7.0, None, None),
        (0.0, "listening", "wake_word"),
        (2.0, "thinking", "speech_end"),
        (1.4, "idle", "not_understood"),   # the "huh?" head tilt
        (5.0, None, None),
        (0.0, "listening", "wake_word"),
        (2.2, "speaking", "compliment"),   # you said something nice
        (4.0, "idle", None),
        (9.0, None, None),
        (0.0, "listening", "wake_word"),
        (1.8, "speaking", "affection"),
        (4.0, "idle", "goodbye"),
        (12.0, None, None),
    ]

    def __init__(self):
        self.i = 0
        self.wait = 1.0

    def update(self, dt, face, speaker=None):
        self.wait -= dt
        if self.wait > 0:
            return
        delay, mode, event = self.SCRIPT[self.i]
        if mode == "speaking" and speaker is not None and not speaker.busy:
            speaker.say(random.choice(DEMO_LINES))
        if mode:
            face.set_mode(mode)
        if event:
            face.event(event)
        self.i = (self.i + 1) % len(self.SCRIPT)
        # a little jitter so it never feels metronomic
        self.wait = max(0.05, self.SCRIPT[self.i][0] * random.uniform(0.85, 1.15))


# How long the conversation stays open after the wake phrase, refreshed by
# every exchange. Long enough for a back-and-forth without repeating yourself,
# short enough that it stops listening when you walk away.
AWAKE_WINDOW = 30.0
# Below this there is nothing to transcribe. Whisper does not return an empty
# string for silence - it invents. Fed a buffer of zeros it reliably produces
# "Thank you.", which then gets answered as though it were said. Measured in
# this room: the noise floor peaks around 0.011, ordinary speech above 0.17,
# so anything under 0.04 carries no voice at all.
MIN_UTTERANCE_PEAK = 0.04
MIN_UTTERANCE_RMS = 0.0015
# How long to stay deaf AFTER we stop talking. The ReSpeaker cancels echo only
# for sound played through its own jack; with PC speakers it has no idea what
# we sounded like, so the only defence is not listening while we talk. Ending
# that the instant playback stops is too early - the room is still ringing, and
# the speaker buffer is still draining - and the tail gets captured as a new
# utterance, transcribed as garbage, and answered. That is the loop where it
# appears to hear things nobody said.
MUTE_TAIL = 0.35
MEMORY_EVERY = 3          # exchanges between memory updates

LINES = [
    "Hello. I am your companion.",
    "I am listening. What would you like to talk about?",
    "That is a good question. Let me think about it for a moment.",
    "I did not quite catch that. Could you say it again?",
    "Watch my eyes while I talk. They follow the sound of my own voice.",
]

MODE_KEYS = {
    pygame.K_1: "sleeping", pygame.K_2: "idle", pygame.K_3: "listening",
    pygame.K_4: "thinking", pygame.K_5: "speaking",
}
EMOTION_KEYS = {
    pygame.K_q: "neutral", pygame.K_w: "happy", pygame.K_e: "sad",
    pygame.K_r: "curious", pygame.K_t: "surprised", pygame.K_y: "annoyed",
    pygame.K_u: "sleepy", pygame.K_z: "pleading", pygame.K_x: "scared",
}
EVENT_KEYS = {
    pygame.K_a: "wake_word", pygame.K_s: "not_understood", pygame.K_f: "compliment",
    pygame.K_g: "affection", pygame.K_h: "greeting", pygame.K_j: "abuse",
    pygame.K_k: "long_absence", pygame.K_n: "startled",
}


def _remember(mem, brain):
    """Fold the conversation into long-term memory, off the render thread."""
    try:
        if mem.update(brain, brain.turns):
            brain.context = mem.context()
            brain.known_names = ([mem.data["name"]]
                                 if mem.data.get("name") else [])
            log("memory updated: name=%s facts=%d"
                % (mem.data.get("name"), len(mem.data.get("facts", []))))
    except Exception as e:
        log("memory update failed: %s" % e)


def apply_action(raw, agenda, timer):
    """Carry out one <<...>> directive. Returns a line for the log, or None.

    Everything here is best-effort on purpose. A malformed directive is a
    model mistake, and the right response to one is to drop it quietly - the
    reply has already been spoken by the time we get here, so raising would
    only take down the conversation after the user has been answered.
    """
    if agenda is None:
        return None
    body = (raw or "").strip()
    low = body.lower()

    if low.startswith("stop timer") or low.startswith("timer stop"):
        return "timer cancelled" if timer and timer.stop() else None

    if ":" not in body:
        return None
    verb, rest = body.split(":", 1)
    verb, rest = verb.strip().lower(), rest.strip()
    if not rest:
        return None

    # "text | extra" - a due time for a task, a label for a timer
    part, extra = (rest.split("|", 1) + [""])[:2]
    part, extra = part.strip(), extra.strip()

    if verb == "task":
        t = agenda.add_task(part, extra or None)
        return "task added: %s%s" % (
            part, " (due %s)" % extra if extra and t and t.get("due") else "")
    if verb == "done":
        t = agenda.complete_task(part)
        return "task completed: %s" % t["text"] if t else "no task matching %r" % part
    if verb == "drop":
        t = agenda.drop_task(part)
        return "task dropped: %s" % t["text"] if t else "no task matching %r" % part
    if verb == "note":
        agenda.add_note(part)
        return "note saved: %s" % part
    if verb == "habit":
        agenda.add_habit(part)
        return "habit tracked: %s" % part
    if verb == "ticked":
        h = agenda.tick_habit(part)
        return ("habit ticked: %s (streak %d)" % (h["text"], h.get("streak", 0))
                if h else "no habit matching %r" % part)
    if verb == "timer" and timer is not None:
        try:
            mins = float(re.findall(r"[\d.]+", part)[0])
        except (IndexError, ValueError):
            return None
        got = timer.start(mins, extra or "focus")
        return "timer started: %g min (%s)" % (got, timer.label)
    return None


def sd_name(idx):
    try:
        import sounddevice as sd
        return sd.query_devices(idx)["name"]
    except Exception:
        return str(idx)


def draw_hud(screen, font, face, demo_on, voice_label="-", level=0.0,
             mic=0.0, hearing=False, thresh=0.0, sens=1.0, heard_text="",
             wake_state="-"):
    lines = [
        "mode      %s" % face.mode,
        "emotion   %s" % face.emotion,
        "activity  %s" % (face.activity if face.mode == "idle" else "-"),
        "mood      %s" % face.mood,
        "",
        "stim      %.2f" % face.dims["stim"],
        "social    %.2f" % face.dims["social"],
        "happy     %.2f" % face.dims["happy"],
        "confident %.2f" % face.dims["confident"],
        "sleepdebt %.2f" % face.sleep_pressure,
        "",
        "demo      %s" % ("ON" if demo_on else "off"),
        "voice     %s" % voice_label,
        "out level %.2f" % level,
        "mic level %.2f" % mic,
        "hearing   %s" % ("YES" if hearing else "-"),
        "mic thres %.3f" % thresh,
        "heard     %s" % (heard_text[:22] if heard_text else "-"),
        "sensitiv. %.2f  [ easier  ] harder" % sens,
        "wake      %s" % wake_state,
    ]
    y = 10
    for line in lines:
        if line:
            img = font.render(line, True, (90, 110, 125))
            screen.blit(img, (12, y))
        y += 15
    # stimulation bar - the clearest way to see the decay working
    w = int(150 * face.dims["stim"])
    pygame.draw.rect(screen, (30, 40, 50), pygame.Rect(12, y + 6, 150, 5))
    if w:
        pygame.draw.rect(screen, (0, 190, 220), pygame.Rect(12, y + 6, w, 5))


def main():
    pygame.init()
    pygame.display.set_caption("Echo")
    W, H = 1024, 600
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,couriernew,monospace", 13)

    face = Face()
    anims = None
    if anim_mod is not None:
        anims = anim_mod.Player()
        face.anim = anims
        print("animations: %s" % (", ".join(anims.names()) or "none found"))
    demo = DemoDriver()
    speaker = None
    if voice_mod is not None:
        try:
            speaker = voice_mod.Speaker()
            print("voice: %s" % speaker.voice.name)
        except Exception as e:
            print("voice unavailable: %s" % e)
    line_i = 0
    listener = None
    if voice_mod is not None:
        try:
            listener = voice_mod.Listener()
            dev = listener.device
            print("mic: %s" % (sd_name(dev) if dev is not None else "system default"))
        except Exception as e:
            print("mic unavailable: %s" % e)
    wake = None
    last_talking = 0.0
    # These need no key and never fail, so they are created unconditionally -
    # tasks and timers should still work on a machine with no network at all.
    agenda = agenda_mod.Agenda()
    timer = agenda_mod.Timer()
    if agenda.open_tasks() or agenda.data.get("habits"):
        print("agenda: %d open task(s), %d habit(s)"
              % (len(agenda.open_tasks()), len(agenda.data.get("habits", []))))
    awake_until = 0.0
    wake_gate = True
    if listener is not None and voice_mod is not None:
        try:
            wake = voice_mod.WakeWord()
            listener.wake = wake
            msg = ("wake word ACTIVE: say '%s' to start talking to it"
                   % wake.phrase.replace("_", " "))
            print(msg); log(msg)
        except Exception as e:
            msg = "wake word UNAVAILABLE (%s) - it will answer any speech" % e
            print(msg); log(msg)
    heard = False
    peak = 0.0
    mem = None
    if memory_mod is not None:
        mem = memory_mod.Memory()
        away = mem.start_session()
        log("memory: name=%s facts=%d sessions=%d away=%s"
            % (mem.data.get("name"), len(mem.data.get("facts", [])),
               mem.data.get("sessions"),
               "first meeting" if away is None else "%.1fh" % away))
    exchanges = 0
    prev_mode = face.mode
    if anims is not None:
        anims.play("startup")
    convo = None
    if brain_mod is not None and voice_mod is not None:
        try:
            b = brain_mod.best_brain()
            convo = Conversation(b, location=brain_mod.find_key("LOCATION"),
                                 agenda=agenda, timer=timer)
            if mem is not None:
                b.context = mem.context()
                b.known_names = [mem.data["name"]] if mem.data.get("name") else []
            print("brain: %s | stt=%s chat=%s"
                  % (type(b).__name__, brain_mod.STT_MODEL, b.chat_model))
        except Exception as e:
            print("brain unavailable: %s" % e)
    last_heard = ""
    log("--- started | voice=%s mic=%s ---"
        % (speaker.voice.name if speaker else "off",
           "on" if listener else "off"))
    demo_on = True
    show_hud = True
    fullscreen = False

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((ev.w, ev.h), pygame.RESIZABLE)
            elif ev.type == pygame.KEYUP:
                if ev.key == pygame.K_SPACE and listener is not None                         and listener.forced:
                    listener.release()
                    awake_until = time.time() + AWAKE_WINDOW
                    log("push-to-talk released")
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k in MODE_KEYS:
                    demo_on = False
                    face.set_mode(MODE_KEYS[k])
                elif k in EMOTION_KEYS:
                    demo_on = False
                    face.set_emotion(EMOTION_KEYS[k])
                elif k in EVENT_KEYS:
                    demo_on = False
                    face.event(EVENT_KEYS[k])
                elif k == pygame.K_v and speaker is not None:
                    demo_on = False
                    log("spoke: %s" % LINES[line_i % len(LINES)])
                    speaker.say(LINES[line_i % len(LINES)])
                    line_i += 1
                elif k == pygame.K_c and speaker is not None:
                    speaker.player.stop()
                elif k == pygame.K_LEFTBRACKET and listener is not None:
                    listener.sensitivity = max(0.25, listener.sensitivity / 1.3)
                    log("mic sensitivity -> %.2f (easier to trigger)"
                        % listener.sensitivity)
                elif k == pygame.K_RIGHTBRACKET and listener is not None:
                    listener.sensitivity = min(8.0, listener.sensitivity * 1.3)
                    log("mic sensitivity -> %.2f (harder to trigger)"
                        % listener.sensitivity)
                elif k == pygame.K_m and listener is not None:
                    listener.muted = not listener.muted
                    log("mic %s" % ("MUTED" if listener.muted else "live"))
                elif k == pygame.K_l:
                    wake_gate = not wake_gate
                    log("wake gate %s" % ("ON" if wake_gate else "OFF - open mic"))
                elif k == pygame.K_p and anims is not None:
                    anims.reload()
                    log("reloaded animations: %s" % ", ".join(anims.names()))
                elif k == pygame.K_b:
                    face.blink()
                elif k == pygame.K_0:
                    face_mod.PIXEL_MODE = not face_mod.PIXEL_MODE
                    face._px = None
                    log("pixel mode %s (%d px wide)"
                        % ("ON" if face_mod.PIXEL_MODE else "OFF",
                           face_mod.PIXEL_COLS))
                elif k in (pygame.K_MINUS, pygame.K_EQUALS):
                    face_mod.PIXEL_COLS = max(48, min(320,
                        face_mod.PIXEL_COLS + (16 if k == pygame.K_EQUALS
                                               else -16)))
                    face._px = None
                    log("pixel resolution %d px wide" % face_mod.PIXEL_COLS)
                elif k == pygame.K_SPACE and listener is not None:
                    # Hold to talk. Interrupts speech, because wanting to cut
                    # in is exactly why you would grab the key mid-sentence.
                    if speaker is not None and speaker.busy:
                        speaker.player.stop()
                    listener.press()
                    face.set_mode("listening")
                    awake_until = time.time() + AWAKE_WINDOW
                    demo_on = False
                    log("push-to-talk held")
                elif k == pygame.K_TAB:
                    demo_on = not demo_on
                elif k == pygame.K_i:
                    show_hud = not show_hud
                elif k == pygame.K_o:
                    fullscreen = not fullscreen
                    screen = pygame.display.set_mode(
                        (0, 0), pygame.FULLSCREEN) if fullscreen else \
                        pygame.display.set_mode((W, H), pygame.RESIZABLE)

        if demo_on:
            demo.update(dt, face, speaker)

        # Poll the speaker from THIS thread; never let the TTS worker touch
        # the face. The level is read every frame straight from the playback
        # callback, so the eyes move with the audio rather than ahead of it.
        now = time.time()
        if wake is not None and wake.take():
            log("wake word heard")
            if speaker is not None:
                speaker.beep("wake", force=True)   # never silently dropped
            face.event("wake_word")          # perk + surprise, as Vector did
            face.set_mode("listening")
            if listener is not None:
                # Drop whatever was being captured before we were addressed.
                listener.restart_utterance()
            awake_until = now + AWAKE_WINDOW
            demo_on = False
        awake = (wake is None) or (not wake_gate) or (now < awake_until)

        talking = speaker is not None and speaker.busy
        if talking:
            last_talking = now
        if listener is not None:
            # No echo cancellation without a speaker in the ReSpeaker's own
            # jack, so the mic would otherwise hear us and we would react to
            # ourselves forever. Deafen it while we talk.
            listener.muted = talking or (now - last_talking) < MUTE_TAIL

        if talking:
            face.set_mode("speaking")
            face.set_amplitude(speaker.level)
        elif face.mode == "speaking":
            face.set_mode("idle")
            face.set_amplitude(0.0)

        if listener is not None and not talking:
            peak = max(peak, listener.level)
            if listener.speaking and not awake:
                # It hears the room but has not been addressed. Stimulation
                # rises - it notices - but it does not interrupt.
                if not heard:
                    heard = True
                    face.event("sound")
                    log("heard speech but not addressed - ignoring "
                        "(say the wake phrase)")
                listener.take_utterance()    # discard, do not transcribe
            elif listener.speaking:
                if not heard:
                    heard = True
                    peak = listener.level
                    demo_on = False          # you talking beats the script
                    face.event("speech_start")
                    log("HEARD YOU - speech started")
                face.set_mode("listening")
                face.set_amplitude(listener.level)
            elif heard and not awake:
                heard = False
                face.set_amplitude(0.0)
            elif heard:
                heard = False
                log("speech ended (peak level %.2f)" % peak)
                peak = 0.0
                face.event("speech_end")
                face.set_amplitude(0.0)
                utt = listener.take_utterance()
                # Ignore anything too short to be a sentence - a cough or a
                # door closing is not worth a round trip.
                if convo is not None and utt is not None \
                        and len(utt) > 0.4 * voice_mod.SAMPLE_RATE:
                    convo.submit(utt)
                    face.set_mode("thinking")
                    if anims is not None:
                        anims.play("focus")
                else:
                    face.set_mode("idle")

        # Drain the worker. Face changes happen here, on the main thread.
        # Checked every frame, not once per exchange: a reminder that only
        # arrives when you happen to speak first is not a reminder. Held back
        # while it is already talking so it never speaks over itself.
        if speaker is not None and not speaker.busy:
            fired = []
            try:
                for t in agenda.due_now():
                    fired.append("A reminder: %s." % t["text"])
                if timer.finished():
                    fired.append("Your %s timer is done." % timer.label)
            except Exception as e:
                log("reminder check failed: %s" % e)
            if fired:
                line = " ".join(fired)
                log("reminder fired: %s" % line)
                face.set_emotion("surprised")
                face.event("wake_word")
                speaker.say(line)
                awake_until = time.time() + AWAKE_WINDOW

        if convo is not None:
            for _ in range(8):          # several may land in one frame
                try:
                    kind, payload = convo.results.get_nowait()
                except queue.Empty:
                    break
                if kind == "emotion":
                    last_heard, emotion = payload
                    face.set_emotion(emotion)   # before a word is spoken
                    if anims is not None:
                        # Authored set-pieces for the moments that deserve
                        # one; everything else stays fully procedural.
                        if emotion == "pleading" or (
                                emotion == "happy" and face.dims["happy"] > 0.92):
                            anims.play("love")
                        elif emotion == "happy" and face.dims["happy"] > 0.75:
                            anims.play("delight")
                        elif emotion == "sleepy":
                            anims.play("relax")
                        elif emotion == "neutral":
                            anims.play("nod")
                    face.event("understood")
                    awake_until = time.time() + AWAKE_WINDOW
                elif kind == "action":
                    # NB: this needs its own branch. Falling through to the
                    # else below would chirp "did not understand" at the user
                    # every time the model successfully did what they asked.
                    try:
                        msg = apply_action(payload, agenda, timer)
                        if msg:
                            log("action: %s" % msg)
                    except Exception as e:
                        log("action failed (%r): %s" % (payload, e))
                elif kind == "say":
                    if speaker is not None:
                        speaker.stream(payload)
                elif kind == "done":
                    awake_until = time.time() + AWAKE_WINDOW
                    exchanges += 1
                    if mem is not None and exchanges % MEMORY_EVERY == 0:
                        threading.Thread(
                            target=_remember, args=(mem, b), daemon=True).start()
                    if speaker is None:
                        face.set_mode("idle")
                else:
                    # Nothing usable came back. A chirp says so faster than a
                    # sentence would, and does not get old the way a repeated
                    # apology does.
                    face.event("not_understood")
                    if speaker is not None:
                        speaker.beep("confused")
                    face.set_mode("idle")

        if face.mode != prev_mode:
            if speaker is not None:
                if face.mode == "sleeping":
                    speaker.beep("sleep")
                elif prev_mode == "sleeping":
                    speaker.beep("waking")
        
        face.update(dt)
        face.draw(screen)
        if show_hud:
            draw_hud(screen, font, face, demo_on,
                     speaker.voice.name if speaker else "off",
                     speaker.level if speaker else 0.0,
                     listener.level if listener else 0.0,
                     bool(listener and listener.speaking),
                     listener.threshold if listener else 0.0,
                     listener.sensitivity if listener else 1.0,
                     last_heard,
                     ("off (open mic)" if not wake_gate else
                      "AWAKE %.0fs" % max(0, awake_until - time.time())
                      if awake else "asleep - say the phrase"))
        pygame.display.flip()

    if mem is not None and speaker is not None and exchanges:
        log("saving memory before exit...")
        _remember(mem, b)
    log("--- closed ---")
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()

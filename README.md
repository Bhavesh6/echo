# companion

A small desk companion with two expressive eyes that listens, thinks and talks
back. Cozmo/Vector-inspired, running on a PC.

You speak, it hears you, transcribes, answers, and says the reply out loud —
with the eyes reacting the whole way through.

```
mic → wake word → VAD → Groq Whisper → LLM → TTS → speaker
         │         │         │          │     │
         └─────────┴─────────┴──────────┴─────┴──→ face
                                        │
                              long-term memory (JSON)
```

It only answers once you say the wake phrase — "hey jarvis" by default. That
opens a conversation window which every reply extends, so you say it once and
then talk normally. Outside the window it still notices sound (stimulation
rises, the eyes react) but it does not interrupt. Anki put trigger-word
detection at priority 2 in Vector's behaviour tree, with only physical safety
above it.

It also remembers you between sessions — your name, a handful of durable
facts, and a rolling summary, compressed by the LLM into a small JSON file
rather than a transcript that would grow forever. Come back after a few hours
and it greets you as someone it missed.

## Why the face is built this way

The eyes are drawn from numbers every frame — no sprites, no image files. The
approach follows Anki's real Cozmo/Vector model, reverse-engineered from the
[Vector Technical Reference Manual](https://randym32.github.io/Vector-TRM.pdf)
and [pycozmo](https://github.com/zayfod/pycozmo).

**Mode and emotion are orthogonal.** Mode is what it's *doing*
(sleeping/idle/listening/thinking/speaking) and drives behaviour: blink rate,
gaze style, motion. Emotion is how it *feels* and drives shape. `listening +
happy` is a different face from `listening + worried`. Vector kept these apart
too — quoting the manual, *"where the eyes are looking is controlled within the
procedural face manager, rather than in the animation files."*

**A static shape can't carry an emotion.** Sad and pleading share a silhouette;
sleepy and scared share a narrowed one. What separates them is *motion*: sad
looks down and moves slowly, pleading has the same lid droop but is big, looks
up, and carries a small fast quiver. Every emotion has a motion signature —
blink rate and duration, gaze bias, tremor, drift, breathing.

**Expressions morph on damped springs, not a shared lerp.** One global morph
rate makes everything arrive with identical timing, which reads dead. Surprise
reaches 90% in 0.12s and overshoots 11%; sleepy takes 0.73s with none. Each
change is preceded by an *anticipation* counter-pose — the eyes move the wrong
way for ~4 frames before the snap.

**Other details that matter more than they sound:** the eyes are shaded by a
movable radial hotspot (that's what sells "it's looking at something"), the
outline is a superellipse rather than a rounded rect (which would give flat
walls at this aspect ratio), everything is supersampled 2× for antialiasing,
and stimulation is a decaying scalar so idle behaviour gets busier when
something's happening and drifts toward sleep when nothing is.

## Setup

Needs Python 3, and a free [Groq](https://console.groq.com) API key.

```bash
pip install pygame sounddevice numpy groq openwakeword onnxruntime
```

Create a `.env` file next to the code:

```
GROQ_API_KEY=gsk_...
```

Optional:

```
ELEVENLABS_API_KEY=sk_...        # nicer voice; falls back to Windows TTS
ELEVENLABS_VOICE_ID=...          # free tier can't use Voice Library voices
GEMINI_API_KEY=...               # stronger brain than Groq's free models
VOICE_BACKEND=sapi               # force a TTS backend
LLM_BACKEND=groq                 # force a brain
WAKE_PHRASE=hey_jarvis           # or alexa, hey_mycroft, hey_rhasspy
WAKE_THRESHOLD=0.5               # lower = easier to trigger
```

Then:

```bash
python run.py
```

It runs without any keys at all — you get the face and a scripted demo, just
no understanding.

## Controls

| | |
|---|---|
| `1`–`5` | sleeping / idle / listening / thinking / speaking |
| `Q W E R T Y U Z X` | neutral, happy, sad, curious, surprised, annoyed, sleepy, pleading, scared |
| `A S F G H J K N` | wake word, not-understood, compliment, affection, greeting, abuse, long-absence, startled |
| `V` / `C` | speak a line / stop speaking |
| `M` · `[` `]` | mute mic · more / less sensitive |
| `L` | turn the wake word off (open mic) |
| `TAB` `I` `O` `SPACE` `ESC` | demo · info overlay · fullscreen · blink · quit |

## Files

| | |
|---|---|
| `face.py` | the renderer and all behaviour. No audio, no network. |
| `voice.py` | audio in and out. `Listener` (mic + VAD + capture), `Player`, TTS backends. |
| `brain.py` | speech-to-text and the LLM. Emits an emotion tag with every reply. |
| `memory.py` | what it remembers about you between sessions. |
| `run.py` | the loop that wires them together. |

The face never decides its own expression from content — drivers call
`set_mode()`, `set_emotion()`, `event()` and `set_amplitude()`. That's why the
keyboard, the scripted demo and the real voice loop all drive it identically.

## Notes from building it

- Audio level for the eyes is measured **inside the playback callback**, on the
  block going to the sound card. Measure it on arrival instead and the face
  leads its own voice — it reads as badly dubbed video.
- End-of-speech needs ~1.1s of silence. At 0.42s it ended the turn on every
  breath, so one sentence arrived as three fragments and each transcribed to
  gibberish.
- A voice-activity threshold can't be purely relative. The adaptive noise floor
  collapses in a quiet room and then any faint sound is "5× the floor". There's
  an absolute minimum too.
- Anything going to TTS is stripped of markdown first. `**England**` spoken
  aloud is "asterisk asterisk England".
- The emotion tag must be chosen for *the reply about to be given*, not the
  user's message — otherwise refusals arrive wearing a happy face.

## Wake word

`openWakeWord`, running locally on the CPU — no key, no network, nothing
leaves the machine until you actually address it. Measured on a 2013 i5 at
**5.3 ms per 80 ms frame, about 7% of one core**. Inference runs on its own
thread rather than in the audio callback, which fires every 16 ms and would
lose a third of its budget to it.

## Memory

`memory.json`, gitignored because it is about you. The LLM folds each
conversation into a name, up to fourteen short facts, and a two or three
sentence summary — only those get injected into the prompt, so cost stays flat
however long you talk to it. Updated every few exchanges and again on exit.

## Hardware

Runs on any PC with a mic. Developed with a
[ReSpeaker Lite](https://wiki.seeedstudio.com/reSpeaker_usb_v3/) USB 2-mic
array (16 kHz native, hardware echo cancellation — though AEC only works with a
speaker in its own 3.5mm jack, which is also what barge-in would need).

## License

MIT

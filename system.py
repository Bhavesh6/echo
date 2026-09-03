#!/usr/bin/env python3
"""
system.py - the small set of things Echo is allowed to do to this machine.

A voice assistant that can only talk is a search box with a face. But the
obvious way to give one hands - let the model write a shell command - is a bad
idea HERE specifically, and the evidence is in this project's own session log:

    you said: Dead? Yes.
    you said: Need? Help me the current weather.
    you said: ...Prime Minister Modi has met President Vladimir Putin...

The first two are ordinary sentences wrecked by a clipping microphone. The
third is a television. Any of them reaching a shell is a bad day, and none of
them is hypothetical.

So this is a NAMED SET of actions, not an interpreter. Each one takes a single
argument, validates it, and can fail safely. Applications are resolved against
an index of what is actually installed - a misheard word opens the wrong app,
which is a shrug, rather than running something, which is not. Nothing here
deletes, kills, installs, or elevates.

Everything returns a short sentence suitable for reading aloud, because that
is what happens to it.
"""
import ctypes
import os
import re
import shutil
import subprocess
import time
import webbrowser

# Where to look for installed applications. These are the two Start Menu
# trees Windows itself uses, so the index is exactly "things with a shortcut".
START_MENUS = [
    os.path.join(os.environ.get("APPDATA", ""),
                 r"Microsoft\Windows\Start Menu\Programs"),
    os.path.join(os.environ.get("PROGRAMDATA", ""),
                 r"Microsoft\Windows\Start Menu\Programs"),
]

# Folders worth searching for a file by name. The whole disk is too slow to
# answer out loud, and the interesting files are nearly always in one of these.
SEARCH_DIRS = [
    os.path.expanduser(p) for p in
    ("~/Desktop", "~/Documents", "~/Downloads", "~/Pictures", "~/Music",
     "~/Videos")
]
MAX_HITS = 5
SEARCH_TIMEOUT = 4.0        # a spoken answer is not worth waiting longer for

SHOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots")

# Windows virtual key codes for the media keys on a keyboard.
VK = {"play": 0xB3, "pause": 0xB3, "next": 0xB0, "previous": 0xB1,
      "prev": 0xB1, "stop": 0xB2, "mute": 0xAD}

# Windows' own apps mostly have no Start Menu shortcut, so the index alone
# cannot reach Calculator or Settings. These are fixed, known-safe strings -
# not paths assembled from anything the model said.
BUILTIN_APPS = {
    "calculator": "calc:", "settings": "ms-settings:", "notepad": "notepad",
    "paint": "mspaint", "camera": "microsoft.windows.camera:",
    "photos": "ms-photos:", "maps": "bingmaps:", "clock": "ms-clock:",
    "calendar": "outlookcal:", "mail": "outlookmail:",
    "file explorer": "explorer", "explorer": "explorer",
    "task manager": "taskmgr", "control panel": "control",
    "character map": "charmap", "snipping tool": "ms-screenclip:",
}

_app_index = None
_app_index_at = 0.0
INDEX_TTL = 300.0           # rescanned occasionally so new installs show up


# --------------------------------------------------------------- utilities
def _norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def app_index(force=False):
    """{spoken name: shortcut path} for everything on the Start Menu.

    This IS the allowlist. Nothing outside it can be launched, which is what
    makes voice control of an application safe: the worst case is opening
    something you did not ask for.
    """
    global _app_index, _app_index_at
    if _app_index is not None and not force and \
            time.time() - _app_index_at < INDEX_TTL:
        return _app_index
    found = {}
    for root in START_MENUS:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if not f.lower().endswith((".lnk", ".url")):
                    continue
                name = _norm(os.path.splitext(f)[0])
                # Installers and uninstallers share names with the real thing
                # and are never what anyone means out loud.
                if not name or any(w in name for w in
                                   ("uninstall", "readme", "documentation",
                                    "release notes", "repair")):
                    continue
                found.setdefault(name, os.path.join(dirpath, f))
    _app_index, _app_index_at = found, time.time()
    return found


def _match_app(want):
    """Loosest sensible match, RANKED. Nobody says an application's full
    shortcut name - "open chrome" means Google Chrome.

    Ranking matters more than it looks: taking the first containment hit made
    "chrome" open Chrome Remote Desktop, because dictionary order put it
    first. A shorter name containing the whole phrase is nearly always the
    thing meant, so candidates are scored and the tightest one wins.
    """
    want = _norm(want)
    if not want:
        return None, None
    if want in BUILTIN_APPS:
        return want, BUILTIN_APPS[want]
    idx = app_index()
    if want in idx:
        return want, idx[want]

    words = [w for w in want.split() if len(w) > 1]
    best, best_score = None, 0.0
    for name in idx:
        nwords = name.split()
        if name == want:
            score = 100.0
        elif name.endswith(" " + want):
            # The distinctive word is usually the LAST one: "chrome" is the
            # head noun in "google chrome" but a modifier in "chrome remote
            # desktop", and ranking prefixes first opened the wrong one.
            score = 80.0
        elif name.startswith(want + " "):
            score = 60.0
        elif want in name:
            # whole phrase present: the less padding around it, the better
            score = 50.0 + 10.0 / (1 + len(name) - len(want))
        else:
            hit = sum(1 for w in words if w in nwords)
            if not hit:
                continue
            score = 10.0 * hit / max(1, len(nwords))
        score -= len(nwords) * 0.1          # prefer the simpler name
        if score > best_score:
            best, best_score = name, score
    if best:
        return best, idx[best]
    for name, cmd in BUILTIN_APPS.items():
        if want in name or name in want:
            return name, cmd
    return None, None


# ----------------------------------------------------------------- actions
def open_app(arg):
    name, path = _match_app(arg)
    if not path:
        return "I could not find an app called %s." % arg
    try:
        if path in BUILTIN_APPS.values():
            # A fixed literal from the table above, never anything spoken.
            subprocess.Popen(["cmd", "/c", "start", "", path], shell=False)
        else:
            os.startfile(path)       # launches the shortcut, not a command
    except Exception as e:
        return "I could not open %s: %s" % (name, e)
    return "Opening %s." % name


def open_web(arg):
    """Only http and https, and only a URL. Anything else - file paths,
    javascript, data URIs - is refused rather than guessed at."""
    url = (arg or "").strip().strip("<>\"'")
    if not url:
        return None
    if not re.match(r"^https?://", url, re.I):
        if not re.match(r"^[\w.-]+\.[a-z]{2,}(/|$)", url, re.I):
            return "That does not look like a web address."
        url = "https://" + url
    if not re.match(r"^https?://[\w.-]+", url, re.I):
        return "That does not look like a web address."
    webbrowser.open(url)
    return "Opening %s." % re.sub(r"^https?://(www\.)?", "", url).split("/")[0]


def search_web(arg):
    q = (arg or "").strip()
    if not q:
        return None
    from urllib.parse import quote_plus
    webbrowser.open("https://duckduckgo.com/?q=" + quote_plus(q))
    return "Searching the web for %s." % q


def set_volume(arg):
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from comtypes import CLSCTX_ALL, cast, POINTER
    dev = AudioUtilities.GetSpeakers()
    vol = cast(dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None),
               POINTER(IAudioEndpointVolume))
    word = _norm(arg)
    cur = vol.GetMasterVolumeLevelScalar()
    if word in ("mute", "silence", "quiet"):
        vol.SetMute(1, None)
        return "Muted."
    if word in ("unmute", "unquiet"):
        vol.SetMute(0, None)
        return "Unmuted."
    if word in ("up", "louder", "higher"):
        target = min(1.0, cur + 0.15)
    elif word in ("down", "quieter", "lower", "softer"):
        target = max(0.0, cur - 0.15)
    elif word == "max":
        target = 1.0
    else:
        n = re.findall(r"\d+", word)
        if not n:
            return None
        target = max(0.0, min(1.0, int(n[0]) / 100.0))
    vol.SetMute(0, None)
    vol.SetMasterVolumeLevelScalar(target, None)
    return "Volume %d percent." % round(target * 100)


def media(arg):
    key = VK.get(_norm(arg).split(" ")[0])
    if key is None:
        return None
    if not media_app_running():
        return ("I could not start any music - there is no music app or "
                "browser open for the media keys to reach.")
    ctypes.windll.user32.keybd_event(key, 0, 0, 0)
    ctypes.windll.user32.keybd_event(key, 0, 2, 0)      # 2 = KEYEVENTF_KEYUP
    return "Done."


def lock_screen(_arg=None):
    ctypes.windll.user32.LockWorkStation()
    return "Locking the screen."


def find_file(arg):
    """Name search across the usual folders, time-boxed. Deliberately not the
    whole disk: an answer nobody waits for is not an answer."""
    want = _norm(arg)
    if not want:
        return None
    words = [w for w in want.split() if w]
    hits, t0 = [], time.time()
    for root in SEARCH_DIRS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirs, files in os.walk(root):
            if time.time() - t0 > SEARCH_TIMEOUT:
                break
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                if all(w in _norm(f) for w in words):
                    hits.append(os.path.join(dirpath, f))
                    if len(hits) >= MAX_HITS:
                        break
            if len(hits) >= MAX_HITS:
                break
        if len(hits) >= MAX_HITS:
            break
    if not hits:
        return "I could not find anything called %s." % arg
    if len(hits) == 1:
        return "I found it in %s." % os.path.basename(os.path.dirname(hits[0]))
    return "I found %d, the first is in %s." % (
        len(hits), os.path.basename(os.path.dirname(hits[0])))


def system_info(arg=None):
    """Battery, disk and memory, in the phrasing a person would use."""
    import psutil
    what = _norm(arg)
    bits = []
    asked_battery = "batter" in what or "power" in what
    if not what or asked_battery:
        b = psutil.sensors_battery()
        if b is not None:
            bits.append("battery %d percent%s" % (
                round(b.percent), ", charging" if b.power_plugged else ""))
        elif asked_battery:
            # A desktop has no battery. Saying nothing made it answer a
            # battery question with disk and processor figures, which reads
            # as evasive rather than as the simple fact that there isn't one.
            return "This machine has no battery - it is a desktop, on mains power."
    if not what or "disk" in what or "space" in what or "storage" in what:
        u = shutil.disk_usage(os.path.expanduser("~"))
        bits.append("%d gigabytes free of %d" %
                    (u.free // (1024 ** 3), u.total // (1024 ** 3)))
    if not what or "memor" in what or "ram" in what:
        m = psutil.virtual_memory()
        bits.append("memory %d percent used" % round(m.percent))
    if not what or "cpu" in what or "processor" in what:
        bits.append("processor %d percent" % round(psutil.cpu_percent(0.3)))
    return ("This machine: " + ", ".join(bits) + ".") if bits else None


def clipboard(arg=None):
    """Read the clipboard when called bare, write to it with an argument."""
    import tkinter
    r = tkinter.Tk()
    r.withdraw()
    try:
        if arg:
            r.clipboard_clear()
            r.clipboard_append(arg)
            r.update()
            return "Copied."
        text = r.clipboard_get()
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return "The clipboard is empty."
        return "The clipboard has: %s" % (text[:300])
    except Exception:
        return "The clipboard is empty."
    finally:
        r.destroy()


def screenshot(_arg=None):
    from PIL import ImageGrab
    os.makedirs(SHOTS_DIR, exist_ok=True)
    path = os.path.join(SHOTS_DIR, time.strftime("shot-%Y%m%d-%H%M%S.png"))
    ImageGrab.grab().save(path)
    return "Saved a screenshot."


def running_apps(_arg=None):
    """What has a window open - not every process, which is a list nobody
    wants read to them."""
    import psutil
    names = set()
    for p in psutil.process_iter(["name"]):
        n = (p.info.get("name") or "").lower()
        if n.endswith(".exe") and n not in (
                "svchost.exe", "runtimebroker.exe", "explorer.exe",
                "dwm.exe", "csrss.exe", "conhost.exe", "sihost.exe"):
            names.add(os.path.splitext(n)[0])
    if not names:
        return None
    top = sorted(names)[:6]
    return "Running: " + ", ".join(top) + "."


# The whole surface, and the only thing run.py may reach. Adding a capability
# is adding a line here; there is no path to anything not listed.
ACTIONS = {
    "open": open_app,
    "web": open_web,
    "websearch": search_web,
    "volume": set_volume,
    "media": media,
    "lock": lock_screen,
    "find": find_file,
    "system": system_info,
    "clipboard": clipboard,
    "screenshot": screenshot,
    "running": running_apps,
}


def run(verb, arg=None):
    """Carry out one named action. Returns a sentence to speak, or None.

    Never raises: this is called from the conversation loop, after the reply
    has already been spoken, and taking the conversation down at that point
    would be a worse outcome than any failed action.
    """
    fn = ACTIONS.get((verb or "").strip().lower())
    if fn is None:
        return None
    try:
        return fn(arg)
    except Exception as e:
        return "I could not do that: %s" % str(e)[:80]


# ---------------------------------------------------------------- questions
# Commands and questions need opposite timing. "Open Chrome" can happen after
# the reply is spoken - the words and the action agree either way. But "how
# much battery have I got" cannot: the answer has to be IN the reply, so it
# must be gathered BEFORE the model writes one. Same split, and the same
# solution, as the web search.
QUESTION_WORDS = {
    "system": ("battery", "charged", "power left", "disk", "storage",
               "space left", "free space", "memory", "ram", "cpu",
               "processor", "how fast is"),
    "clipboard": ("clipboard", "copied", "what did i copy"),
    "running": ("running", "open right now", "what is open", "programs open",
                "apps are open"),
}


def context_for(text):
    """Machine state to add to the prompt for this turn, or "".

    Only gathered when actually asked about - reading battery, disk and CPU
    every turn would put a paragraph of numbers in front of the model on
    every "hello", and it would start volunteering them.
    """
    t = " " + (text or "").lower().strip() + " "
    bits = []
    for verb, words in QUESTION_WORDS.items():
        if any(w in t for w in words):
            try:
                # ONLY system_info takes the question as a filter. Passing it
                # to the others is destructive: clipboard() writes when given
                # an argument, so "what is on my clipboard" overwrote the
                # clipboard with that very sentence.
                got = run(verb, text) if verb == "system" else run(verb)
            except Exception:
                got = None
            if got:
                bits.append(got)
    if not bits:
        return ""
    return ("\nTHIS MACHINE, read just now - answer from it, conversationally:\n"
            + "\n".join(bits))


# Applications that respond to the media keys. Sending VK_MEDIA_PLAY_PAUSE
# into a desktop with no player open does nothing at all - the key goes
# nowhere and the call returns success, so Echo said "Starting the music now"
# to a silent room. Better to find out first and say so.
MEDIA_APPS = ("spotify", "vlc", "wmplayer", "musicbee", "foobar2000",
              "itunes", "aimp", "winamp", "groove", "music.ui", "deezer",
              "chrome", "msedge", "firefox", "brave", "opera")


def media_app_running():
    try:
        import psutil
    except ImportError:
        return True          # cannot tell; assume yes rather than refuse
    try:
        for pr in psutil.process_iter(["name"]):
            n = (pr.info.get("name") or "").lower()
            if any(a in n for a in MEDIA_APPS):
                return True
    except Exception:
        return True
    return False

#!/usr/bin/env python3
"""
art.py - generate reference art for the face, using Gemini's image models.

Every number in face.py - eye width, corner rounding, gradient, brow weight -
was chosen by guessing at what the face SHOULD look like. That is the actual
bottleneck: not the rendering, which is fine, but the absence of a target to
render toward.

This calls "nano banana" (the Gemini image models) to produce that target, so
the parameters can be tuned against a picture instead of an opinion.

    python art.py                 the default set of expressions
    python art.py "a prompt"      one image from your own prompt
    python art.py --pro           use nano banana pro (slower, better)

Images land in art/. Nothing here touches the running face.
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

from brain import ENV_PATH, find_key

MODEL_FLASH = "gemini-2.5-flash-image"        # nano banana
MODEL_PRO = "gemini-3-pro-image-preview"      # nano banana pro
URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
       "%s:generateContent?key=%s")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "art")

# The constant half of every prompt. Repeated verbatim so the nine expressions
# come back as one consistent character rather than nine different robots -
# which is the usual failure of generating a reference sheet in one shot.
STYLE = (
    "A cute desk-robot face on a pure black background. ONLY two large glowing "
    "cyan eyes, wider than they are tall with soft rounded corners, plus two "
    "thin cyan eyebrow strokes above them and one small thin cyan mouth curve "
    "below. No nose, no head outline, no body, no text, nothing else. The eyes "
    "have a vertical gradient, bright cyan at the top fading to deeper blue at "
    "the bottom, with a soft glowing highlight inside each. Cozmo and WALL-E "
    "inspired, clean vector shapes with soft glow bloom against black. "
    "Wide 16:9 framing, eyes centred. "
)

# One per emotion in face.py's EMOTIONS, so each generated image has exactly
# one set of numbers it is the reference for.
SHOTS = {
    "neutral": "Calm, warm, resting expression. Eyes open and level.",
    "happy": "Genuinely delighted: eyes crinkled shut into upward crescents, "
             "eyebrows raised, a wide warm smile, small blush marks below the "
             "outer eye corners.",
    "sad": "Sad: the outer corners of the eyes drooping down, inner eyebrow "
           "ends lifted, a small downward mouth.",
    "curious": "Curious and interested: one eyebrow raised higher than the "
               "other, eyes slightly widened, head-tilted feeling.",
    "surprised": "Startled: eyes suddenly very wide and round, eyebrows high, "
                 "small round open mouth.",
    "annoyed": "Annoyed: inner eyebrow ends pulled sharply down, eyes narrowed "
               "to flat slits, mouth a short flat line.",
    "sleepy": "Sleepy: eyes mostly closed to low horizontal slits, eyebrows "
              "relaxed and low, tiny mouth.",
    "pleading": "Adoring and pleading: the eyes themselves are glowing cyan "
                "hearts, a small cat-shaped w mouth, blush marks, short "
                "excitement lines radiating from the outer eye corners.",
    "scared": "Frightened: eyes wide with very small bright pupils, eyebrows "
              "lifted and tilted, a small wavering mouth.",
}


def generate(prompt, model, key, tries=3):
    """One image, as raw bytes. Returns None if the model sent only text,
    which it does when it decides the prompt is a question rather than a
    request for a picture."""
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req = urllib.request.Request(
        URL % (model, key), data=body,
        headers={"Content-Type": "application/json"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:800]
            # Two different 429s. "limit: 0" means this key is not entitled
            # to the model at all - every Gemini image model is paid-only, so
            # retrying is guaranteed to fail and just wastes minutes. Only a
            # real rate limit is worth waiting out.
            if "limit: 0" in detail:
                print("    NOT AVAILABLE on a free key: image generation on the"
                      " Gemini API requires billing enabled.")
                return None
            if e.code == 429 and attempt < tries - 1:
                print("    rate limited, waiting 30s")
                time.sleep(30)
                continue
            print("    HTTP %s: %s" % (e.code, detail))
            return None
        except Exception as e:
            print("    %s" % e)
            return None
    else:
        return None

    for cand in data.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    text = json.dumps(data)[:300]
    print("    no image in response: %s" % text)
    return None


def main():
    key = find_key("GEMINI_API_KEY")
    if not key:
        print("No Gemini API key.")
        print("Get one free at https://aistudio.google.com/apikey and put")
        print("    GEMINI_API_KEY=your_key")
        print("in %s" % ENV_PATH)
        return 1

    args = [a for a in sys.argv[1:] if a != "--pro"]
    model = MODEL_PRO if "--pro" in sys.argv else MODEL_FLASH
    os.makedirs(OUT_DIR, exist_ok=True)
    print("model: %s" % model)

    shots = ({"custom": args[0]} if args
             else {k: STYLE + v for k, v in SHOTS.items()})
    if args:
        shots["custom"] = args[0]

    made = 0
    for name, prompt in shots.items():
        print("  %s ..." % name)
        img = generate(prompt, model, key)
        if img:
            path = os.path.join(OUT_DIR, "%s.png" % name)
            with open(path, "wb") as f:
                f.write(img)
            print("    %s  (%d KB)" % (path, len(img) // 1024))
            made += 1
    print("%d image(s) in %s" % (made, OUT_DIR))
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())

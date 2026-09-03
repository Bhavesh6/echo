#!/usr/bin/env python3
"""
search.py - the small part of the world Echo is allowed to look up.

A language model cannot know today's date, the weather, or where the nearest
post office is. Asked one of those it does one of two things, and both are
bad: it refuses, or it invents a confident answer. Inventing is much worse,
because a companion that makes up an address is worse than useless.

This gives it a way to actually look. DuckDuckGo needs no API key, no account
and no billing - which matters here, because both of the obvious alternatives
turned out to be paid: Gemini's image models report a free-tier limit of zero,
and its google_search grounding returns quota-exhausted on the first call.

Deliberately NOT a general tool-calling loop. A search costs about two
seconds, which is a long silence in a spoken conversation, so it runs only for
the questions that genuinely cannot be answered from what the model already
knows. Everything else goes straight to the brain at its usual speed.
"""
import re

MAX_RESULTS = 3          # three is enough to answer; more just adds latency
TIMEOUT = 6.0            # a spoken answer is not worth waiting longer than this
SNIPPET = 220            # characters kept per result

# Questions that need the world, not the model. Split by kind because the
# location ones additionally need to know WHERE the person is, and saying
# "I do not know where you are" is far better than guessing a city.
LOCAL_WORDS = ("near me", "nearest", "near by", "nearby", "closest",
               "around here", "in my area", "close to me")
FRESH_WORDS = ("today", "todays", "today's", "tonight", "tomorrow",
               "right now", "currently", "this week", "this month",
               "latest", "recent", "news", "weather", "temperature",
               "forecast", "price of", "stock", "score", "who won",
               "what time is", "open now", "opening hours", "released")

# A question about a named thing the model may simply not have: these are
# weak signals on their own, so they only count when the sentence is a
# question.
LOOKUP_HINTS = ("where is", "where can i", "how much does", "how far",
                "phone number", "address of", "directions to")

# Fresh questions that are ALSO about where you are. These contain no "near
# me", so they were being searched without a location - and "the weather right
# now" then returns whichever city the results happen to favour. Answering
# someone in one city with another city's weather is worse than not answering:
# it is confidently wrong, and it sounds completely right.
GEO_WORDS = ("weather", "temperature", "forecast", "sunset", "sunrise",
             "traffic", "open now", "opening hours", "air quality")


def needs_search(text):
    """(needed, kind) - kind is "local", "fresh" or None.

    Kept as plain string matching rather than asking a model. A round trip to
    decide whether to do a round trip would cost more than the search itself.
    """
    t = " " + (text or "").lower().strip() + " "
    if any(w in t for w in LOCAL_WORDS):
        return True, "local"
    if any(w in t for w in FRESH_WORDS):
        return True, "fresh"
    if "?" in t or t.lstrip().startswith(("where", "when", "how much", "who is")):
        if any(w in t for w in LOOKUP_HINTS):
            return True, "fresh"
    return False, None


def lookup(query, n=MAX_RESULTS):
    """A few web results, or [] if anything at all goes wrong. Searching must
    never be able to take the conversation down with it."""
    try:
        from ddgs import DDGS
    except ImportError:
        return []
    try:
        with DDGS(timeout=TIMEOUT) as d:
            return list(d.text(query, max_results=n))
    except Exception:
        return []


# Conversational wrapping that is part of talking, not part of the question.
# Left in, it becomes search terms and drags the results off target.
FILLER = re.compile(
    r"^(hey |ok |okay |so |um |uh |please |can you |could you |would you "
    r"|do you know |tell me |show me |find me |find |i want to know |what is "
    r"the |whats the |where is the )+", re.I)


def _clean(s):
    """Search snippets arrive with collapsed spacing, stray markup and broken
    encodings. Everything in here may end up being spoken aloud, so it is
    worth more scrubbing than a screen would need."""
    s = (s or "")
    # U+FFFD and friends: mojibake from the result page. Spoken, these become
    # noise or make the sentence stop dead.
    s = "".join(c for c in s
                if c == "\n" or (32 <= ord(c) < 127) or ord(c) > 160)
    s = s.replace("�", " ")
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)   # "-Today's Date is"
    s = re.sub(r"\s+", " ", s).strip()
    return s[:SNIPPET]


def _query(text, location=None):
    """The question, stripped down to something worth searching for."""
    q = FILLER.sub("", (text or "").strip()).strip(" ?.!")
    if location:
        q = re.sub("|".join(LOCAL_WORDS), " ", q, flags=re.I)
        q = re.sub(r"\s+", " ", q).strip()
        q = "%s %s" % (q, location)
    return q or (text or "").strip()


def context_for(text, location=None):
    """The block to add to the system prompt for this turn, or "".

    Returns a note rather than results when the question is about somewhere
    nearby and we do not know where that is - a companion should ask where you
    live, not silently search a city it guessed.
    """
    needed, kind = needs_search(text)
    if not needed:
        return ""

    geo = kind == "local" or any(w in text.lower() for w in GEO_WORDS)
    if geo and not location:
        return ("\nThe person asked about something that depends on where "
                "they are, but you do not know where they live and have no "
                "way to find out. Say so plainly in one sentence and ask "
                "which town or area they are in, so you can remember it. Do "
                "not guess a place, and do not answer for a different city.")

    query = _query(text, location if geo else None)

    results = lookup(query)
    if not results:
        return ("\nYou tried to look this up and the search failed. Say you "
                "could not find it right now, in one short sentence. Do not "
                "invent an answer.")

    lines = ["\nWEB SEARCH RESULTS for %r - use these to answer, and say it "
             "conversationally rather than reading them out:" % query]
    for r in results:
        title = _clean(r.get("title"))
        body = _clean(r.get("body"))
        lines.append("- %s: %s" % (title, body))
    return "\n".join(lines)

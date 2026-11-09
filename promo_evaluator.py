"""
promo_evaluator.py - Module 3: Promo Evaluation Engine.

For every candidate in candidate_promos.json it scores six dimensions on 0-10, each computed from the
screenplay text/timing/delivery of the clip itself (never from the candidate's own score alone):

    hook               first 3 seconds: how fast and how hard does something happen?
    cliffhanger        last lines: is something left open?
    standalone         could a stranger follow the situation?
    intensity          emotional force of delivery, exchange speed, sound
    coherence          one connected beat with clean edges and a rising shape
    character_clarity  can a stranger tell who is who?

then a weighted overall (+ a balance factor), HARD REJECTION GATES for the assignment's fundamental
requirements, three grounded explanations (each cites item ids + verbatim quotes that can be audited
against the raw screenplay), and a final recommendation.

An optional LLM verdict (the `semantic` block from semantic_judge.py) is blended into hook / cliffhanger /
standalone: dim = 0.4 * rules + 0.6 * llm.  Candidates without a verdict are scored by rules alone and are
labelled `rules_only` so the reader knows the confidence.  The LLM is swappable (llm_client.py).

Usage:
  python promo_evaluator.py extracted_scenes.json candidate_promos.json -o evaluated_promos.json
         [--llm none|anthropic|openai-compat|replay:FILE] [--llm-cache FILE]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # sibling modules (works under python -I)
from candidate_generator import (BACKREF, REACT, YOU, add_dialogue_dynamics, build_events,
                                extract_event_signals, score_interval, squash)

EVAL_VERSION = "1.0"

# --------------------------------------------------------------------------- weights (and why)
WEIGHTS: Dict[str, float] = {
    "hook": 0.25,               # assignment criterion A. Without a stop-the-scroll opening nobody sees the rest.
    "cliffhanger": 0.25,        # assignment criterion B. The promo's whole job is to make people go and watch.
    "standalone": 0.20,         # assignment criterion C. Slightly lower: screenplay text can only approximate it,
                                #   and a clip that confuses a stranger is also caught by the hard gate below.
    "intensity": 0.12,          # emotion is what makes a hook and cliffhanger land, but it is not a requirement
                                #   by itself (a quiet whispered threat can beat a shouting match).
    "coherence": 0.10,          # one clean beat; edge defects are cheap to fix by trimming, so it weighs less.
    "character_clarity": 0.08,  # mostly a sub-component of standalone; kept separate so the report shows why.
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

WEIGHT_RATIONALE = {
    "hook": "Assignment criterion A (first 3 seconds). Weighted equal-highest because a failed hook means the clip is never watched.",
    "cliffhanger": "Assignment criterion B. Equal-highest: a promo that resolves its own tension gives viewers no reason to open the series.",
    "standalone": "Assignment criterion C. 0.20 because we can only approximate a stranger's understanding from text, and a hard gate already removes clips that clearly need prior context.",
    "intensity": "Emotional force makes hooks and cliffhangers land, but is not required on its own, so it is a supporting 0.12.",
    "coherence": "A promo should be one connected beat. Edge defects are fixable by trimming, so 0.10.",
    "character_clarity": "Overlaps with standalone (a stranger must know who is who). Kept separate at 0.08 so the report shows which of the two failed.",
}

# Hard rejection gates: the assignment's fundamental requirements. Failing ANY gate => rejected,
# whatever the weighted score says (a weighted average must not let a great hook hide a missing cliffhanger).
GATES: Dict[str, float] = {
    "min_total_sec": 20.0, "max_total_sec": 60.0,
    "min_hook": 4.0, "min_cliffhanger": 4.0, "min_standalone": 4.0, "min_coherence": 4.0,
    "min_dialogue_lines": 3,
}
# Recommendation bands on the final (balance-adjusted) score.
BANDS = {"top_pick": 7.0, "recommend": 6.0, "maybe": 5.0}
REVIEW_MARGIN = 2.0       # rules-only candidates this close to a soft gate go to LLM review instead of being rejected
BALANCE_FLOOR = 0.80      # final = weighted * (0.80 + 0.20 * min(hook, cliffhanger, standalone)/10)

# --------------------------------------------------------------------------- vocabulary
STAKES = re.compile(
    r"\b(husband|wife|marri\w*|duke|duchess|lord|lady|king|queen|prince|princess|heir|bride|groom|wedding|"
    r"mother|father|brother|sister|son|daughter|child|baby|inherit\w*|estate|fortune|ring|gallows|hang\w*|"
    r"prison|arrest\w*|trial|poison\w*|kill\w*|die|dead|death|murder\w*|secret|truth|lie[sd]?|liar|betray\w*|"
    r"traitor|divorce\w*|ruin\w*|soldier|war|sword|crown|throne|debt|curse\w*|thief|theft|steal|stole\w*)\b", re.I)
GENERIC_SPEAKER = re.compile(r"\b(MYSTERIOUS|BEARDED|BLOND|ELDER|STRANGER|FORMER|OFF-SCREEN|THUG|TALL|BALD|THIRD|"
                             r"SCARRED|LOYALIST|ARISTOCRATIC|PEASANT|NOBLEMAN|NOBLEWOMAN|NOBLE|TOWNSMAN|VOICE|"
                             r"MATRIARCH|GOLD GOWN)\b")
OPENER_CONT = re.compile(r"^\s*(then|but|and|so|because|which|or|yet|still|however)\b", re.I)
STOP = {"the", "and", "his", "her", "she", "him", "you", "your", "with", "that", "this", "from", "have", "had",
        "was", "for", "are", "not", "but", "all", "out", "who", "what", "will", "can", "now"}
WORD = re.compile(r"[a-z']{4,}")


def r1(x: float) -> float:
    return round(max(0.0, min(10.0, x)), 1)


def clip_text(s: str, n: int = 14) -> str:
    """First n words of a script line, copied verbatim (ellipsis only when truncated)."""
    w = s.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def quote(e: Dict[str, Any], n: int = 14) -> Dict[str, str]:
    return {"item_id": e["item_id"], "speaker": e.get("speaker") or e["kind"].upper(),
            "ts": e["start_ts"], "quote": clip_text(e["text"], n)}


def fmt_q(q: Dict[str, str]) -> str:
    return f'{q["speaker"]} ({q["ts"]}): "{q["quote"]}"'


# --------------------------------------------------------------------------- pack-level context
def pack_context(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Data-derived facts: who the leads are, which speaker names are real proper names."""
    lines, narrative = Counter(), []
    for ep in doc["episodes"]:
        for sc in ep["scenes"]:
            for it in sc["items"]:
                if it["kind"] == "dialogue":
                    lines[it.get("character_canon") or it.get("character")] += 1
                elif it["kind"] in ("action", "camera"):
                    narrative.append(it["text"])
    blob = " ".join(narrative)
    proper = {s for s in lines if len(s.split()) == 1 and re.search(r"\b" + re.escape(s.title()) + r"\b", blob)}
    return {"leads": [s for s, _ in lines.most_common(2)], "proper": proper, "lines": lines}


# --------------------------------------------------------------------------- window extraction
def episode_events(ep: Dict[str, Any]) -> List[Dict[str, Any]]:
    evs = build_events(ep)
    for e in evs:
        extract_event_signals(e)
    add_dialogue_dynamics(evs)
    return evs


def window_events(c: Dict[str, Any], evs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seg in c["segments"]:
        out += [e for e in evs if seg["start_sec"] - 1e-6 <= e["start"] < seg["end_sec"]]
    return out


# --------------------------------------------------------------------------- dimensions
def score_hook(c, win) -> Tuple[float, Dict[str, Any]]:
    t0 = c["segments"][0]["start_sec"]
    zone = [e for e in win if e["start"] < t0 + 3.0 and e["kind"] != "camera"]
    dl = [e for e in zone if e["kind"] == "dialogue"]
    first = dl[0] if dl else None
    latency = (first["start"] - t0) if first else None
    lat_s = 0.0 if latency is None else 1.0 if latency <= 0.8 else 0.7 if latency <= 1.5 else 0.4 if latency <= 3 else 0.0
    zint = squash(score_interval(zone)["total"], 1.4) if zone else 0.0
    tier = max([e["tier"] for e in dl], default=0) / 3.0
    punct = 0.0
    if first:
        txt = first["text"]
        punct = 1.0 if ("?" in txt or "!" in txt or YOU.search(txt)) else 0.4
    mid = bool(first and (OPENER_CONT.search(first["text"]) or BACKREF.search(first["text"])))
    feat = 0.30 * lat_s + 0.35 * zint + 0.20 * tier + 0.15 * punct - (0.15 if mid else 0.0)
    rule = 10 * (0.5 * c["rule_scores"]["hook"] + 0.5 * max(feat, 0.0))
    info = {"first_line": quote(first) if first else None, "latency_sec": None if latency is None else round(latency, 2),
            "zone_tier": max([e["tier"] for e in dl], default=0), "zone_signals": sorted(
                {s["cat"] for e in zone for s in e["signals"] if s["cat"]}), "mid_conversation_opener": mid,
            "lead_in_sec": c["hook_zone"]["lead_in_sec"], "rule_component": round(rule, 2)}
    return rule, info


def last_dialogue(win):
    d = [e for e in win if e["kind"] == "dialogue"]
    return d[-1] if d else None


ARRIVAL = re.compile(r"\b(door|latch|knock\w*|footsteps?|enters?|bursts?|steps? in|arrives?|appears?|emerges?|"
                     r"gunshot|scream\w*|shatter\w*|crash\w*|slam\w*)\b", re.I)
DEMAND = re.compile(r"^\s*(tell me|tell us|bring|give me|choose|decide|answer|speak|kneel|confess|swear|sign|come with|get out|leave|name|prove)\b|\bor (else|you|your|he|she|they)\b", re.I)
TRAIL = ("...", "\u2026", "-", "\u2014")


def score_cliff(c, win) -> Tuple[float, Dict[str, Any]]:
    """
    End-of-clip score built from the ENDING itself (the candidate-level rule score has a flat floor and
    does not discriminate). Additive rubric, 0-10; each contributing fact is named in `notes`.
    """
    end_t = c["segments"][-1]["end_sec"]
    tail = [e for e in win if e["start"] >= end_t - 5.0]
    dl = [e for e in win if e["kind"] == "dialogue"]
    last = dl[-1] if dl else None
    s, notes = 2.0, []
    ep_cut = bool(c["tail_features"].get("episode_end_cut"))
    reach = 5.0 if ep_cut else 3.5            # at an episode cut a held closing shot is part of the beat
    if last and end_t - last["end"] <= reach:
        txt = last["text"].rstrip()
        if txt.endswith("?"):
            s += 4.0
            notes.append("final line is an unanswered question")
            if last["tier"] >= 1 or YOU.search(txt):
                s += 0.8
        if txt.endswith(TRAIL):
            s += 2.5
            notes.append("final line trails off or is cut")
        prev = dl[-2] if len(dl) >= 2 else None
        if prev and prev["text"].rstrip().endswith("?") and prev.get("speaker") != last.get("speaker") \
                and not txt.endswith("?"):
            s -= 1.5
            notes.append("the last question is answered inside the clip")
        sig = score_interval([e for e in tail if e["kind"] == "dialogue"][-2:])
        hot = [k for k, v in sig["cats"].items() if v["corroborated"] and k in
               ("threat", "revelation", "betrayal", "twist", "suspense")]
        if hot:
            s += 2.5
            notes.append("closing lines carry " + "/".join(sorted(hot)) + " signals")
        if DEMAND.search(txt) and not txt.endswith("?"):
            s += 2.5
            notes.append("final line is a demand/ultimatum nobody has answered")
        if last["tier"] >= 2:
            s += 1.0
            notes.append("final line is delivered at high intensity")
    elif last:
        s -= 1.0
        notes.append(f"{end_t - last['end']:.1f}s of tail after the last line (energy may sag)")
    arrival = next((e for e in tail if e["kind"] in ("sound", "action", "camera") and e["start"] >= end_t - 4.0
                    and ARRIVAL.search(e["text"])), None)
    if arrival:
        s += 2.5
        notes.append("ends on an arrival/ominous sound or action")
    if any(REACT.search(e["text"]) for e in tail if e["kind"] in ("action", "sound")):
        s += 1.5
        notes.append("ends on a reaction shot")
    if c["tail_features"].get("episode_end_cut"):
        s += 1.5
        notes.append("cut lands at the episode's end (the series ends episodes on a hook)")
    if c["tail_features"].get("resolution_language"):
        s -= 3.0
        notes.append("last lines use resolving language")
    s = max(0.0, min(10.0, s))
    return s, {"last_line": quote(last) if last else None, "tail_features": sorted(c["tail_features"]),
               "arrival_event": quote(arrival) if arrival else None, "notes": notes, "rule_component": round(s, 2)}


def addressed_or_introduced(name: str, win, speaker_events) -> bool:
    toks = [t.lower() for t in re.findall(r"[A-Za-z]{4,}", name) if t.lower() not in STOP]
    for e in win:
        if e["kind"] in ("action", "camera") or (e["kind"] == "dialogue" and e.get("speaker") != name):
            low = e["text"].lower()
            if any(t in low for t in toks):
                return True
    return False


def score_character_clarity(c, win, ctx) -> Tuple[float, Dict[str, Any]]:
    dl = [e for e in win if e["kind"] == "dialogue" and e.get("speaker")]
    cnt = Counter(e["speaker"] for e in dl)
    detail, vals = {}, []
    for s, n in cnt.items():
        base = 1.0 if s in ctx["proper"] else 0.4 if GENERIC_SPEAKER.search(s) else 0.8
        if base < 0.9 and addressed_or_introduced(s, win, dl):
            base = 0.9
        detail[s] = {"lines": n, "clarity": base}
        vals.append(base * (1 if n >= 2 else 0.7))
    ident = sum(vals) / len(vals) if vals else 0.0
    k = len(cnt)
    size = 1.0 if k in (2, 3) else 0.6 if k == 4 else 0.3 if k in (1, 5) else 0.1
    lead = 1.0 if set(cnt) & set(ctx["leads"]) else 0.5
    voice = sum(1 for n in cnt.values() if n >= 2) / k if k else 0.0
    score = 10 * (0.45 * ident + 0.25 * size + 0.15 * lead + 0.15 * voice)
    off = sorted({e["speaker"] for e in dl if "O.S." in (e.get("delivery") or "")})
    return score, {"speakers": detail, "n_speakers": k, "lead_present": lead == 1.0, "off_screen": off}


def score_standalone(c, win, ctx, cc_score) -> Tuple[float, Dict[str, Any]]:
    dl = [e for e in win if e["kind"] == "dialogue"]
    text_all = " ".join(e["text"] for e in win if e["kind"] != "camera")
    stake_hits = sorted({m.group(0).lower() for m in STAKES.finditer(text_all)})
    stake_line = max((e for e in dl if STAKES.search(e["text"])),
                     key=lambda e: (len({m.group(0).lower() for m in STAKES.finditer(e["text"])}), e["words"]), default=None)
    first10 = [e for e in dl if e["start"] < c["segments"][0]["start_sec"] + 10]
    backrefs = [e for e in first10 if BACKREF.search(e["text"])]
    s = 4.5
    s += 0.35 * (cc_score - 5.0) / 5.0 * 2.0                       # who-is-who feeds situation clarity
    s += 0.0 if not stake_hits else 1.0 if len(stake_hits) == 1 else 1.8 if len(stake_hits) <= 3 else 2.2
    s -= min(2.4, 1.0 * len(backrefs))
    cont = "continues_previous_episode" in c["flags"] or "continues_previous_episode" in c.get("flags_extra", [])
    if cont:
        s -= 1.5
    first = dl[0] if dl else None
    opener = bool(first and OPENER_CONT.search(first["text"]))
    if opener:
        s -= 1.0
    # setup visible in the clip itself: an action/camera line naming a speaker's character
    setup = [e for e in win if e["kind"] in ("action", "camera") and
             sum(1 for sp in {x["speaker"] for x in dl if x.get("speaker")}
                 if any(t.lower() in e["text"].lower() for t in re.findall(r"[A-Za-z]{4,}", sp))) >= 2]
    if setup:
        s += 0.6
    pron = sum(1 for e in first10 if re.match(r"\s*(he|she|they|it|that|this)\b", e["text"], re.I))
    s -= min(1.0, 0.5 * pron)
    rule = s + 1.0 * (1 if stake_hits else 0) * 0                 # (explicit no-op: stakes already counted)
    info = {"stakes_terms": stake_hits[:6], "stakes_line": quote(stake_line) if stake_line else None,
            "back_references": [quote(e) for e in backrefs], "continues_previous_episode": cont,
            "opens_mid_sentence": opener, "setup_in_clip": bool(setup),
            "rule_component": round(rule, 2)}
    return rule, info


def score_intensity(c, win) -> Tuple[float, Dict[str, Any]]:
    dl = [e for e in win if e["kind"] == "dialogue"]
    hi = sum(1 for e in dl if e["tier"] >= 2) / len(dl) if dl else 0.0
    exc = sum(1 for e in dl if "!" in e["text"]) / len(dl) if dl else 0.0
    snd = 1.0 if any(e["kind"] == "sound" and e["signals"] for e in win) else 0.0
    rapid = 1.0 if any(s["tag"] == "rapid_exchange" for e in win for s in e["signals"]) else 0.0
    feat = 0.45 * hi + 0.2 * exc + 0.1 * snd + 0.25 * rapid
    score = 10 * (0.55 * c["rule_scores"]["intensity"] + 0.45 * feat)
    top = max(dl, key=lambda e: (e["tier"], len(e["signals"])), default=None)
    return score, {"high_intensity_share": round(hi, 2), "rapid_exchange": bool(rapid), "sound_fx": bool(snd),
                   "peak_line": quote(top) if top else None}


def content_words(s: str) -> set:
    return {w for w in WORD.findall(s.lower()) if w not in STOP}


def score_coherence(c, win) -> Tuple[float, Dict[str, Any]]:
    dl = [e for e in win if e["kind"] == "dialogue"]
    scenes = {e["scene_id"] for e in win}
    s = 0.0
    s += 3.0 if len(c["segments"]) == 1 and len(scenes) == 1 else 1.5 if len(scenes) == 2 else 0.5
    jac = []
    for a, b in zip(dl, dl[1:]):
        A, B = content_words(a["text"]), content_words(b["text"])
        jac.append(len(A & B) / max(1, len(A | B)))
    topic = min(1.0, (sum(jac) / len(jac) if jac else 0.0) / 0.12)
    s += 2.0 * topic
    changes = sum(1 for a, b in zip(dl, dl[1:]) if a.get("speaker") != b.get("speaker"))
    recip = changes / max(1, len(dl) - 1)
    s += 1.5 * min(1.0, recip / 0.6)
    half = c["segments"][0]["start_sec"] + (c["segments"][-1]["end_sec"] - c["segments"][0]["start_sec"]) / 2
    a1 = score_interval([e for e in win if e["start"] < half])["total"]
    a2 = score_interval([e for e in win if e["start"] >= half])["total"]
    rise = 1.0 if a2 >= a1 else 0.5 if a2 >= 0.6 * a1 else 0.1
    s += 1.5 * rise
    gaps = [b["start"] - a["end"] for a, b in zip(dl, dl[1:])]
    dead = max(gaps) if gaps else 0.0
    s += 1.0 if dead <= 4.0 else 0.0
    first = dl[0] if dl else None
    clean_open = not (first and OPENER_CONT.search(first["text"]))
    s += 1.0 if clean_open else 0.0
    return s, {"single_scene": len(scenes) == 1, "topic_continuity": round(topic, 2),
               "speaker_reciprocity": round(recip, 2), "escalates": rise == 1.0,
               "longest_silence_sec": round(dead, 1), "clean_opening": clean_open}


# --------------------------------------------------------------------------- LLM blend
def blend(rule: float, llm: Optional[float]) -> float:
    return rule if llm is None else 0.4 * rule + 0.6 * llm


def llm_view(c) -> Optional[Dict[str, Any]]:
    sem = c.get("semantic") or {}
    return sem if sem.get("status") == "ok" else None


# --------------------------------------------------------------------------- explanations (grounded)
def explain_hook(score, info, sem) -> str:
    fl = info["first_line"]
    if fl is None:
        return "No one speaks in the first 3 seconds, so there is nothing to stop a scroller."
    parts = [f"Opens with {fmt_q(fl)}."]
    if info["latency_sec"] is not None:
        parts.append(f"First line lands {info['latency_sec']}s in" + (" (immediate)." if info["latency_sec"] <= 0.8 else "."))
    if info["zone_signals"]:
        parts.append("Opening seconds carry " + ", ".join(info["zone_signals"]) + " signals.")
    if info["mid_conversation_opener"]:
        parts.append("The first line reads as a continuation (starts with a conjunction or back-reference), which weakens it.")
    if sem and sem.get("hook_strength") is not None:
        parts.append(f"Model read of the first 3s: {sem['hook_strength']}/10.")
    parts.append("Verdict: " + ("strong scroll-stopper." if score >= 7 else "workable opening." if score >= 5 else "weak opening."))
    return " ".join(parts)


def explain_cliff(score, info, sem) -> str:
    ll = info["last_line"]
    parts = [f"Ends on {fmt_q(ll)}." if ll else "No spoken line near the end."]
    parts += [n.capitalize() + "." for n in info["notes"]]
    if sem and sem.get("cliffhanger_strength") is not None:
        parts.append(f"Model read of the ending: {sem['cliffhanger_strength']}/10.")
    parts.append("Verdict: " + ("leaves a clear open question." if score >= 7 else "some tension left open." if score >= 5
                                else "ending feels resolved or inert."))
    return " ".join(parts)


def explain_standalone(score, info, cc, sem) -> str:
    parts = []
    if info["stakes_line"]:
        parts.append(f"Stakes are stated in-clip: {fmt_q(info['stakes_line'])} (terms: {', '.join(info['stakes_terms'])}).")
    else:
        parts.append("No stakes vocabulary (family, law, danger, status) appears, so a newcomer may not see what is at risk.")
    ids = [f"{s} ({'named' if v['clarity'] >= 1.0 else 'identified in clip' if v['clarity'] >= 0.9 else 'role only' if v['clarity'] >= 0.7 else 'unclear'})"
           for s, v in cc["speakers"].items()]
    parts.append("Speakers: " + ", ".join(ids) + ".")
    for q in info["back_references"][:2]:
        parts.append(f"Leans on earlier events: {fmt_q(q)}.")
    if info["continues_previous_episode"]:
        parts.append("The episode's continuity notes say it picks up directly from the previous episode.")
    if info["opens_mid_sentence"]:
        parts.append("Opens mid-thought.")
    if sem and sem.get("needs_context"):
        parts.append("A new viewer would be missing: " + "; ".join(sem["needs_context"]) + ".")
    parts.append("Verdict: " + ("a stranger can follow this." if score >= 7 else "mostly followable with a caption or title card."
                                if score >= 5 else "needs too much prior knowledge."))
    return " ".join(parts)


# --------------------------------------------------------------------------- one candidate
def evaluate(c: Dict[str, Any], evs: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, Any]:
    win = window_events(c, evs)
    sem = llm_view(c)
    genuine = bool(sem and sem["is_genuine_promo_moment"] and not sem["keyword_only_false_positive"])

    hook_r, hook_i = score_hook(c, win)
    cliff_r, cliff_i = score_cliff(c, win)
    cc_r, cc_i = score_character_clarity(c, win, ctx)
    sa_r, sa_i = score_standalone(c, win, ctx, cc_r)
    int_r, int_i = score_intensity(c, win)
    coh_r, coh_i = score_coherence(c, win)

    llm = (lambda k: sem.get(k) if sem else None)
    scores = {
        "hook": r1(blend(hook_r, llm("hook_strength"))),
        "cliffhanger": r1(blend(cliff_r, llm("cliffhanger_strength"))),
        "standalone": r1(blend(sa_r, llm("standalone_clarity"))),
        "intensity": r1(int_r),
        "coherence": r1(coh_r),
        "character_clarity": r1(cc_r),
    }
    weighted = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    low = min(scores["hook"], scores["cliffhanger"], scores["standalone"])
    balance = BALANCE_FLOOR + (1 - BALANCE_FLOOR) * low / 10.0
    overall = round(weighted * balance, 2)

    # ---- hard gates
    dur = sum(s["duration_sec"] for s in c["segments"])
    dl = [e for e in win if e["kind"] == "dialogue"]
    nspk = len({e["speaker"] for e in dl if e.get("speaker")})
    rej: List[Dict[str, str]] = []

    def gate(code, failed, why):
        if failed:
            rej.append({"code": code, "reason": why})

    gate("duration", not (GATES["min_total_sec"] <= dur <= GATES["max_total_sec"]), f"total duration {dur:.1f}s is outside 20-60s")
    gate("no_hook", scores["hook"] < GATES["min_hook"], f"hook {scores['hook']} < {GATES['min_hook']}: nothing compelling in the first 3s")
    gate("no_cliffhanger", scores["cliffhanger"] < GATES["min_cliffhanger"], f"cliffhanger {scores['cliffhanger']} < {GATES['min_cliffhanger']}: the ending resolves or fizzles")
    gate("not_standalone", scores["standalone"] < GATES["min_standalone"], f"standalone {scores['standalone']} < {GATES['min_standalone']}: needs too much prior context")
    gate("incoherent", scores["coherence"] < GATES["min_coherence"], f"coherence {scores['coherence']} < {GATES['min_coherence']}: not a clean single beat")
    gate("too_thin", len(dl) < GATES["min_dialogue_lines"], f"only {len(dl)} spoken line(s): not enough dialogue to sell")
    if sem and not genuine:
        rej.append({"code": "llm_not_genuine", "reason": "model judged it a keyword false positive or not a real promo moment in context"})

    # ---- rules cannot reject confidently near a threshold (cliffhanger rule vs LLM rank agreement is only ~0.3)
    soft = {"no_hook": ("hook", "min_hook"), "no_cliffhanger": ("cliffhanger", "min_cliffhanger"),
            "not_standalone": ("standalone", "min_standalone")}
    borderline = (not sem and rej and all(r["code"] in soft and scores[soft[r["code"]][0]] >= GATES[soft[r["code"]][1]] - REVIEW_MARGIN
                                          for r in rej))
    # ---- recommendation
    weak = [k for k in ("hook", "cliffhanger", "standalone") if scores[k] < 5.5]
    if borderline:
        rec = "needs_llm_review"
        action = ("Rules-only score fails " + "; ".join(r["code"] for r in rej) +
                  f" by less than {REVIEW_MARGIN} points. Rules are not reliable enough to reject here: run the LLM judge.")
    elif rej:
        rec, action = "reject", "Do not use: " + "; ".join(r["reason"] for r in rej) + "."
    elif overall >= BANDS["top_pick"]:
        rec, action = "top_pick", "Use as is."
    elif overall >= BANDS["recommend"]:
        rec, action = "recommend", "Use; " + (f"consider fixing weaker dimension(s): {', '.join(weak)}." if weak else "no changes needed.")
    elif overall >= BANDS["maybe"]:
        rec, action = "maybe", "Borderline; only if a better option is not available" + (f" (weak: {', '.join(weak)})." if weak else ".")
    else:
        rec, action = "reject", f"Overall {overall} is below the {BANDS['maybe']} floor."
        rej.append({"code": "low_overall", "reason": f"balance-adjusted score {overall} < {BANDS['maybe']}"})
    confidence = "rules+llm" if sem else "rules_only"
    if rec in ("top_pick", "recommend") and not sem:
        action += " (Rules-only score: confirm with the LLM judge before publishing.)"

    explanations = {
        "hook": explain_hook(scores["hook"], hook_i, sem),
        "cliffhanger": explain_cliff(scores["cliffhanger"], cliff_i, sem),
        "standalone": explain_standalone(scores["standalone"], sa_i, cc_i, sem),
    }
    evidence = [q for q in (hook_i["first_line"], cliff_i["last_line"], sa_i["stakes_line"], int_i["peak_line"]) if q]
    evidence += sa_i["back_references"][:2]
    seen, ev_unique = set(), []
    for q in evidence:
        if q["item_id"] not in seen:
            seen.add(q["item_id"]); ev_unique.append(q)
    return {
        "candidate_id": c["candidate_id"], "episode_id": c["episode_id"], "episode_label": c["episode_label"],
        "scene_number": c["primary_scene_number"], "start_ts": c["start_ts"], "end_ts": c["end_ts"],
        "duration_sec": round(dur, 2), "segments": c["segments"], "main_characters": c["main_characters"],
        "moment_types": c.get("moment_types_final") or [m["type"] for m in c["moment_types"]],
        "scores": scores, "weighted_score": round(weighted, 2), "balance_factor": round(balance, 3),
        "overall_score": overall, "promotional_potential": overall,
        "gates": {"passed": not [r for r in rej if r["code"] != "low_overall"], "failures": rej, "borderline_unjudged": bool(borderline)},
        "recommendation": rec, "recommended_action": action, "confidence": confidence,
        "explanations": explanations, "why_it_attracts": (c.get("why_it_attracts") or {}).get("text"),
        "why_source": (c.get("why_it_attracts") or {}).get("source"),
        "needs_context": list(sem.get("needs_context") or []) if sem else [],
        "llm_model": sem.get("model") if sem else None,
        "evidence": ev_unique,
        "details": {"hook": hook_i, "cliffhanger": cliff_i, "standalone": sa_i, "intensity": int_i,
                    "coherence": coh_i, "character_clarity": cc_i},
        "rule_vs_llm": {k: {"rules": r1(v), "llm": (llm(lk))} for k, v, lk in
                        (("hook", hook_r, "hook_strength"), ("cliffhanger", cliff_r, "cliffhanger_strength"),
                         ("standalone", sa_r, "standalone_clarity"))} if sem else None,
        "flags": c.get("flags", []),
    }


# --------------------------------------------------------------------------- run
def evaluate_all(doc: Dict[str, Any], cdoc: Dict[str, Any]) -> List[Dict[str, Any]]:
    ctx = pack_context(doc)
    eps = {str(e["episode_id"]): e for e in doc["episodes"]}
    cache: Dict[str, List[Dict[str, Any]]] = {}
    out = []
    for c in cdoc["candidates"]:
        k = str(c["episode_id"])
        if k not in cache:
            cache[k] = episode_events(eps[k])
        out.append(evaluate(c, cache[k], ctx))
    return out


def rank(evals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    order = {"top_pick": 0, "recommend": 1, "maybe": 2, "needs_llm_review": 3, "reject": 4}
    return sorted(evals, key=lambda e: (order[e["recommendation"]], -e["overall_score"]))


def summarize(evals: List[Dict[str, Any]]) -> Dict[str, Any]:
    rec = Counter(e["recommendation"] for e in evals)
    codes = Counter(f["code"] for e in evals if e["recommendation"] == "reject" for f in e["gates"]["failures"])
    dims = {k: [e["scores"][k] for e in evals] for k in WEIGHTS}
    return {"n_candidates": len(evals), "recommendations": dict(rec), "rejection_reasons": dict(codes),
            "judged_by_llm": sum(1 for e in evals if e["confidence"] == "rules+llm"),
            "dimension_means": {k: (round(sum(v) / len(v), 2) if v else None) for k, v in dims.items()},
            "dimension_ranges": {k: ([min(v), max(v)] if v else None) for k, v in dims.items()},
            "episodes_with_a_recommended_promo": len({e["episode_id"] for e in evals
                                                       if e["recommendation"] in ("top_pick", "recommend")})}


def write_csv(path: str, evals: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "candidate_id", "episode", "scene", "start", "end", "dur_s", "recommendation", "overall"] +
                   list(WEIGHTS) + ["confidence", "characters", "hook_line", "last_line"])
        for i, e in enumerate(rank(evals), 1):
            hl, ll = e["details"]["hook"]["first_line"], e["details"]["cliffhanger"]["last_line"]
            w.writerow([i, e["candidate_id"], e["episode_label"], e["scene_number"], e["start_ts"], e["end_ts"],
                        e["duration_sec"], e["recommendation"], e["overall_score"]] + [e["scores"][k] for k in WEIGHTS] +
                       [e["confidence"], "; ".join(e["main_characters"]), hl["quote"] if hl else "", ll["quote"] if ll else ""])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Promo Evaluation Engine")
    ap.add_argument("extracted"); ap.add_argument("candidates")
    ap.add_argument("-o", "--out", default="evaluated_promos.json")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--llm", default="none", help="none | anthropic | openai-compat | replay:FILE (judges candidates lacking a verdict)")
    ap.add_argument("--llm-cache", default=None)
    a = ap.parse_args(argv)
    doc = json.load(open(a.extracted, encoding="utf-8"))
    cdoc = json.load(open(a.candidates, encoding="utf-8"))
    if a.llm != "none":
        from llm_client import make_client
        from semantic_judge import judge_candidate, merge_semantic
        client = make_client(a.llm, a.llm_cache)
        eps = {str(e["episode_id"]): e for e in doc["episodes"]}
        for c in cdoc["candidates"]:
            if (c.get("semantic") or {}).get("status") == "ok":
                continue
            if hasattr(client, "has") and not client.has(c["candidate_id"]):
                continue
            ep = eps[str(c["episode_id"])]
            merge_semantic(c, judge_candidate(c, ep, [it for sc in ep["scenes"] for it in sc["items"]], client))
    evals = rank(evaluate_all(doc, cdoc))
    result = {"evaluator_version": EVAL_VERSION, "source_candidates": os.path.basename(a.candidates),
              "weights": WEIGHTS, "weight_rationale": WEIGHT_RATIONALE, "gates": GATES, "bands": BANDS,
              "balance_rule": f"overall = weighted * ({BALANCE_FLOOR} + {1 - BALANCE_FLOOR:.2f} * min(hook, cliffhanger, standalone)/10)",
              "summary": summarize(evals), "evaluations": evals}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if a.csv:
        write_csv(a.csv, evals)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

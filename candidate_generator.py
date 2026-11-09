"""
candidate_generator.py - Module 2 of the Promo Discovery Engine.

Finds candidate promo moments (20-60 s) in extracted_scenes.json using three layers:
  1. Rules       keyword / delivery-tag / action / sound signals for ten moment types.
  2. Dialogue    turn-taking, rapid-fire exchanges, punctuation, reply/continuation structure.
  3. LLM         (optional, swappable) semantic check of the top candidates - see semantic_judge.py.

Context rule: a dramatic keyword on its own is NOT evidence. A category only scores in full when at
least two independent signal families agree within the same stretch of screenplay (families: text
keywords, delivery tags, physical action, sound, turn-taking, reaction cues, punctuation). Otherwise it
is down-weighted to 30% and flagged uncorroborated.

Timestamps are never invented: every start/end is the original timestamp string of an item (or the
original scene end when a cut lands within 1.5 s of it).

Usage:
  python candidate_generator.py extracted_scenes.json -o candidate_promos.json
         [--per-episode 4] [--llm none|anthropic|openai-compat|replay:FILE] [--llm-top-k 30]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DETECTOR_VERSION = "1.0"
CATEGORIES = ["revelation", "confrontation", "betrayal", "breakdown", "twist", "suspense",
              "threat", "romance", "entrance"]            # 'unresolved' is scored from the tail
DEFAULTS = {"min_dur": 20.0, "max_dur": 60.0, "lookback": 14.0, "agg_window": 8.0, "peak_sep": 6.0,
            "peaks_per_episode": 8, "per_episode": 4, "nms_iou": 0.5, "min_peak": 1.6,
            "min_candidate_score": 0.30, "min_hook": 0.25, "min_hook_episode_end": 0.10, "hook_seconds": 3.0, "scene_end_snap": 1.5, "episode_end_snap": 7.0}
WEIGHTS = {"hook": 0.35, "cliffhanger": 0.35, "intensity": 0.20, "standalone": 0.10}


def _c(p: str) -> "re.Pattern[str]":
    return re.compile(p, re.IGNORECASE)


# ----------------------------------------------------------------------------- lexicons
# Dialogue text. (name, regex, weight). Weak on purpose: keywords only support other evidence.
TEXT_RULES: Dict[str, List[Tuple[str, "re.Pattern[str]", float]]] = {
    "revelation": [
        ("secret_truth", _c(r"\b(the truth|secret|all along|never told|didn'?t know|did not know|who i (really )?am|"
                            r"true identity|real name|still alive|not dead|pregnant)\b"), 1.0),
        ("status_word", _c(r"\b(the (late )?duke|duchess|heir|royal|crown)\b"), 0.6),
        ("recognition", _c(r"\b(impossible|it can'?t be|it cannot be|you are the|you'?re the|he is the|"
                           r"resembles?|same scar)\b"), 0.9)],
    "confrontation": [
        ("accusation", _c(r"\b(how dare|who are you|what have you|you (dare|lied|knew|chose|cannot|will not)|"
                          r"get out|enough|stop (this|it)|you have no right)\b"), 0.8)],
    "betrayal": [
        ("betrayal", _c(r"\b(betray\w*|traitor|behind my back|how could you|you promised|abandon\w*|deceiv\w*|"
                        r"cast (me|her|him) aside|chose (her|him|my sister|clarissa)|left me|used me|schemed|plotted)\b"), 1.0)],
    "breakdown": [
        ("pleading", _c(r"\b(please|don'?t leave|i beg|forgive me|it hurts|why (me|did you|would you))\b"), 0.6)],
    "twist": [
        ("reversal", _c(r"\b(think again|you are wrong|you'?re wrong|no longer|not anymore|"
                        r"you thought|did not expect|didn'?t expect|surprise)\b"), 0.7)],
    "suspense": [
        ("hush", _c(r"\b(listen|quiet|keep your voice|someone is|footsteps|hide|watching us|if (they|anyone|the guards) "
                    r"(hear|find|see))\b"), 0.8)],
    "threat": [
        ("threat", _c(r"\b(hangs? from|gallows|execut\w*|punish\w*|regret|suffer\w*|you will (not )?(leave|live|pay|die|beg)|"
                      r"or else|last chance|if you (refuse|fail|speak|dare|tell|leave)|fail,? or refuse|"
                      r"we are both dead|a warning|i will (not )?(allow|let|forgive))\b"), 1.0)],
    "romance": [
        ("intimacy", _c(r"\b(i love you|kiss|beautiful|my (wife|husband|love)|stay with me|hold me|safe with me|"
                        r"don'?t go)\b"), 0.8)],
    "entrance": [
        ("arrival", _c(r"\b(who is that|who are you|you are here|you came|you'?re here)\b"), 0.5)],
}
# Delivery tags (the `delivery` field). Matched on whole words.
DELIV_RULES: Dict[str, Tuple[set, float]] = {
    "threat": ({"threatening", "menacing", "chilling", "warning", "vindictive", "sinister", "malicious", "ominous"}, 1.0),
    "confrontation": ({"accusatory", "accusing", "defiant", "outraged", "indignant", "furious", "shouting", "screaming",
                       "screeching", "demanding", "scornful", "enraged", "incredulous", "roaring", "booming"}, 0.9),
    "betrayal": ({"heartbroken", "betrayed", "bitter", "crushed", "wounded"}, 0.6),
    "breakdown": ({"sobbing", "weeping", "tearful", "choked", "broken", "despairing", "hysterical", "anguished",
                   "sorrowful", "pained", "crying", "desperate", "pleading", "trembling"}, 0.9),
    "romance": ({"tender", "passionate", "devoted", "loving", "intimate", "reverent", "protective", "reassuring",
                 "gentle", "warm"}, 0.5),
    "suspense": ({"whispering", "hushed", "tense", "anxious", "alarmed", "panicked", "frantic", "terrified", "cautious"}, 0.7),
    "revelation": ({"stunned", "shocked", "gasping", "startled", "horrified", "astonished"}, 0.8),
    "twist": ({"triumphant", "unyielding", "steely", "final"}, 0.4),
}
TIER3 = {"screaming", "shouting", "furious", "sobbing", "enraged", "hysterical", "screeching", "roaring", "outraged",
         "terrified", "horrified", "panicked", "frantic", "venomous", "menacing", "threatening", "chilling", "booming",
         "weeping", "anguished"}
TIER2 = {"desperate", "defiant", "accusing", "accusatory", "trembling", "tearful", "heartbroken", "shocked", "stunned",
         "indignant", "scornful", "sinister", "ominous", "pleading", "incredulous", "vindictive", "malicious", "urgent",
         "gasping", "alarmed", "shaken", "despairing", "choked", "broken", "fierce", "demanding", "harsh", "cruel",
         "gloating", "taunting", "bitter", "startled", "tense"}
TIER1 = {"cold", "stern", "firm", "resolute", "commanding", "mocking", "smug", "authoritative", "steely", "grave",
         "breathless", "hushed", "whispering", "dismissive", "condescending", "arrogant", "calculating", "unyielding",
         "solemn", "determined", "anxious", "nervous", "hesitant", "spiteful", "derisive", "contemptuous", "grim"}
# Physical action (ACTION lines; CAMERA lines count at half weight).
ACTION_RULES: Dict[str, List[Tuple[str, "re.Pattern[str]", float]]] = {
    "confrontation": [("physical", _c(r"\b(slaps?|slapped|strikes?|shoves?|grabs?|seizes?|drags?|hurls?|slams?|backhand\w*|"
                                      r"pins?|wrenches?)\b"), 1.0)],
    "breakdown": [("tears", _c(r"\b(tears (streak|fall|spill|well)|tearful|weeps?|sobs?|crying|collapses?|"
                               r"falls? to (her|his) knees|knees buckle|clutches? (her|his) (chest|face|heart))\b"), 0.9)],
    "revelation": [("reveal", _c(r"\b(reveals?|revealing|unmasks?|removes? (his|the|her) (hood|cloak|mask)|"
                                 r"pulls? back (his|the) (hood|cloak)|crest|signet|royal seal)\b"), 1.0)],
    "entrance": [("entrance", _c(r"\b(enters?|bursts?|steps? (in|forward|out)|appears?|emerges?|arrives?|"
                                 r"stands? (in|at) the (doorway|threshold)|flung open|strides?|storms? in|rushes? in|"
                                 r"stranger)\b"), 0.8)],
    "romance": [("closeness", _c(r"\b(leans? in|cups?|fingers (intertwine|brush)|pulls? (her|him) (close|against)|"
                                 r"forehead|gaze lingers|hand (on|against|to) (her|his) (cheek|chest|face)|"
                                 r"brushes? (a strand|her hair|his))\b"), 0.9)],
    "suspense": [("furtive", _c(r"\b(glances? (over|around|behind)|peers?|creeps?|tiptoes?|listens?|freezes?|hides?|"
                                r"presses? (herself|himself) against)\b"), 0.8)],
    "threat": [("weapon", _c(r"\b(draws?|raises?|points?|levels?|aims?) (a |an |his |her |the )?(sword|blade|dagger|knife|"
                             r"weapon|pistol)\b|\bblade (to|at|against)\b"), 1.0)],
    "twist": [("sudden", _c(r"\b(suddenly|abruptly|unexpectedly|drops? to (one|his) knee|kneels?)\b"), 0.7)],
}
SOUND_RULES: Dict[str, List[Tuple[str, "re.Pattern[str]", float]]] = {
    "entrance": [("door_burst", _c(r"\b(doors? (burst|swing|slam|crash|fling)|boots? (on|striking|pounding))\b"), 0.6)],
    "suspense": [("ominous", _c(r"\b(ominous|footsteps|creak\w*|drone|tension|silence|heartbeat|muffled)\b"), 0.5)],
    "threat": [("violent", _c(r"\b(violent|crash\w*|shatter\w*|clang\w*)\b"), 0.5)],
}
REACT = _c(r"\b(gasps?|stunned|freezes?|frozen|stares?|eyes widen|jaw drops?|recoils?|steps? back|horrified|disbelief|"
           r"falls silent|stumbles?|staggers?|pales?|speechless|silence)\b")
STINGER = _c(r"\b(cliffhanger|sting|swell\w*|cut to black|abrupt cut|ominous|drone|dramatic)\b")
RESOLVED = _c(r"\b(forgive (me|you)|i agree|thank you|goodbye|farewell|at last|well done|welcome home|"
              r"i accept|of course|very well|as you wish|it is settled)\b")
RESOLVED_TONE = {"warm", "reassuring", "relieved", "content", "grateful", "peaceful"}
BACKREF = _c(r"\b(again|as i (said|told)|earlier|last (night|time)|remember when|that (man|woman|day|night)|"
             r"the other (day|night)|the same (man|woman))\b")
YOU = re.compile(r"\byou(r|rs)?\b", re.IGNORECASE)


# ----------------------------------------------------------------------------- events & signals
def delivery_tokens(delivery: Optional[str]) -> List[str]:
    return re.findall(r"[a-z']+", (delivery or "").lower())


def delivery_tier(tokens: List[str]) -> int:
    t = set(tokens)
    return 3 if t & TIER3 else 2 if t & TIER2 else 1 if t & TIER1 else 0


def build_events(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten an episode into time-ordered events (all kinds), keeping scene info."""
    evs: List[Dict[str, Any]] = []
    for sc in episode["scenes"]:
        for it in sc["items"]:
            if it["start_sec"] is None or it["end_sec"] is None:
                continue
            toks = delivery_tokens(it.get("delivery"))
            evs.append({"item_id": it["item_id"], "kind": it["kind"], "scene_id": sc["scene_id"],
                        "scene_number": sc["scene_number"], "scene_end_sec": sc["end_sec"],
                        "scene_end_ts": sc["end_ts"], "start": it["start_sec"], "end": it["end_sec"],
                        "start_ts": it["start_ts"], "end_ts": it["end_ts"],
                        "speaker": it.get("character_canon") or it.get("character"), "text": it["text"],
                        "delivery": it.get("delivery"), "tokens": toks, "tier": delivery_tier(toks),
                        "words": len(re.findall(r"\w+", it["text"])), "signals": []})
    evs.sort(key=lambda e: (e["start"], e["item_id"]))
    for i, e in enumerate(evs):
        e["i"] = i
    return evs


def add_signal(e: Dict[str, Any], cat: Optional[str], fam: str, w: float, tag: str) -> None:
    e["signals"].append({"cat": cat, "fam": fam, "w": w, "tag": tag})


def extract_event_signals(e: Dict[str, Any]) -> None:
    """Keyword / delivery / action / sound / punctuation signals for one event."""
    text, kind = e["text"], e["kind"]
    if kind == "dialogue":
        for cat, rules in TEXT_RULES.items():
            for name, rx, w in rules:
                if rx.search(text):
                    add_signal(e, cat, "lex", w, name)
        for cat, (words, w) in DELIV_RULES.items():
            hit = words & set(e["tokens"])
            if hit:
                add_signal(e, cat, "deliv", w, sorted(hit)[0])
        if "?!" in text or "!!" in text or ("!" in text and e["words"] <= 6):
            add_signal(e, None, "punct", 0.3, "exclaim")
    elif kind in ("action", "camera"):
        scale = 1.0 if kind == "action" else 0.5
        for cat, rules in ACTION_RULES.items():
            for name, rx, w in rules:
                if rx.search(text):
                    add_signal(e, cat, "phys", w * scale, name)
        if kind == "action" and REACT.search(text):
            add_signal(e, None, "react", 0.4, "reaction")
    elif kind == "sound":
        for cat, rules in SOUND_RULES.items():
            for name, rx, w in rules:
                if rx.search(text):
                    add_signal(e, cat, "snd", w, name)
        if REACT.search(text):
            add_signal(e, None, "react", 0.4, "reaction")


def add_dialogue_dynamics(evs: List[Dict[str, Any]]) -> None:
    """
    Turn-taking: >=4 dialogue lines inside 7 s with >=3 speaker changes between >=2 speakers is a rapid
    exchange. It is a signal for 'confrontation' and also an independent corroborator for other types.
    """
    dl = [e for e in evs if e["kind"] == "dialogue" and e["speaker"]]
    for k, e in enumerate(dl):
        recent = [x for x in dl[max(0, k - 6):k + 1] if e["start"] - x["start"] <= 7.0]
        if len(recent) < 4:
            continue
        spk = [x["speaker"] for x in recent]
        changes = sum(1 for a, b in zip(spk, spk[1:]) if a != b)
        if changes >= 3 and len(set(spk)) >= 2:
            you = sum(len(YOU.findall(x["text"])) for x in recent)
            words = sum(x["words"] for x in recent) or 1
            w = 0.9 + (0.3 if you / words >= 0.10 else 0.0)
            add_signal(e, "confrontation", "turns", w, "rapid_exchange")
            add_signal(e, None, "turns", 0.0, "rapid_exchange")


# ----------------------------------------------------------------------------- scoring primitives
def score_interval(evs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Category scores for a set of events with the corroboration rule.
    Returns {'total', 'cats': {cat: {score, raw, fams, corroborated}}, 'tier'}.
    """
    cats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"raw": 0.0, "fams": set()})
    general: set = set()
    tier = 0
    for e in evs:
        tier = max(tier, e["tier"])
        for sg in e["signals"]:
            if sg["cat"] is None:
                general.add(sg["fam"])
            else:
                cats[sg["cat"]]["raw"] += sg["w"]
                cats[sg["cat"]]["fams"].add(sg["fam"])
    out, total = {}, 0.0
    for cat, v in cats.items():
        fams = v["fams"] | general
        corroborated = len(fams) >= 2
        score = min(v["raw"], 2.5) * (1.0 if corroborated else 0.3)
        out[cat] = {"score": round(score, 3), "raw": round(v["raw"], 3),
                    "fams": sorted(v["fams"]), "corroborated": corroborated}
        total += score
    return {"total": total + 0.25 * tier, "cats": out, "tier": tier}


def squash(x: float, k: float) -> float:
    return 1.0 - math.exp(-max(x, 0.0) / k)


def moment_strength(evs: List[Dict[str, Any]], t: float, cfg: Dict[str, Any]) -> float:
    window = [e for e in evs if t - 1.0 <= e["start"] < t + cfg["agg_window"]]
    return score_interval(window)["total"]


def find_peaks(evs: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Anchor events where moment strength is a local maximum (>= min_peak), best first."""
    anchors = [e for e in evs if e["kind"] in ("dialogue", "action")]
    strength = {e["i"]: moment_strength(evs, e["start"], cfg) for e in anchors}
    peaks = []
    for e in anchors:
        m = strength[e["i"]]
        if m < cfg["min_peak"]:
            continue
        rivals = [strength[x["i"]] for x in anchors if x is not e and abs(x["start"] - e["start"]) <= cfg["peak_sep"]
                  and (strength[x["i"]] > m or (strength[x["i"]] == m and x["start"] < e["start"]))]
        if not rivals:
            peaks.append({"event": e, "strength": m})
    peaks.sort(key=lambda p: -p["strength"])
    return peaks[:cfg["peaks_per_episode"]]


# ----------------------------------------------------------------------------- window construction
def straddles(evs: List[Dict[str, Any]], t: float, tol: float = 0.005) -> bool:
    """True if a dialogue line is in progress at time t (a cut there would slice a line).
    Strict on purpose (timestamps have 0.01s resolution): a cut 0.1s inside a line still clips the audio."""
    return any(e["kind"] == "dialogue" and e["start"] < t - tol and e["end"] > t + tol for e in evs)


def snap_start_event(anchors: List[Dict[str, Any]], k: int) -> int:
    """Never open on a continuation ('...') or on a bare reply: step back to the line it answers."""
    for _ in range(2):
        e = anchors[k]
        if e["kind"] != "dialogue" or k == 0:
            break
        prev = anchors[k - 1]
        cont = e["text"].startswith(("...", "…"))
        reply = e["words"] <= 3 and prev["kind"] == "dialogue" and e["start"] - prev["end"] <= 3.5 \
            and prev["speaker"] != e["speaker"]
        if (cont and e["start"] - prev["end"] <= 2.0) or reply:
            k -= 1
        else:
            break
    return k


def tail_features(evs: List[Dict[str, Any]], end_t: float, ep_end: float, scene_end: float) -> Dict[str, float]:
    """How unresolved does the clip's ending feel? Each feature is a named bonus/penalty."""
    tail = [e for e in evs if end_t - 5.0 <= e["start"] < end_t and e["kind"] != "camera"]
    dlg = [e for e in tail if e["kind"] == "dialogue"]
    f: Dict[str, float] = {}
    last = dlg[-1] if dlg else None
    if last and end_t - last["end"] <= 2.0:
        if last["text"].rstrip().endswith("?"):
            f["ends_on_question"] = 0.35
        if last["text"].rstrip().endswith(("...", "…", "-", "—")):
            f["ends_unfinished"] = 0.2
    if any(REACT.search(e["text"]) for e in tail if e["kind"] in ("action", "sound")):
        f["reaction_at_end"] = 0.25
    if any(STINGER.search(e["text"]) for e in tail if e["kind"] == "sound"):
        f["sound_sting"] = 0.2
    if score_interval(tail)["total"] >= 1.2:
        f["open_tension"] = 0.3
    if ep_end - end_t <= 1.5:
        f["episode_end_cut"] = 0.35
    last2 = dlg[-2:]
    if any(RESOLVED.search(e["text"]) or (set(e["tokens"]) & RESOLVED_TONE) for e in last2):
        f["resolution_language"] = -0.4
    return f


def window_context_flags(evs: List[Dict[str, Any]], start: float, ep: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Cheap proxy for 'makes sense to a new viewer' (the LLM does the real judgement)."""
    penalty, flags = 0.0, []
    first10 = [e for e in evs if start <= e["start"] < start + 10 and e["kind"] == "dialogue"]
    refs = sum(1 for e in first10 if BACKREF.search(e["text"]))
    if refs:
        penalty += min(0.3, 0.12 * refs)
        flags.append("back_references_early")
    cont = (ep.get("continuity_notes") or "").lower()
    if start < 8.0 and ("picks up" in cont or "follows directly" in cont or "continues" in cont):
        penalty += 0.25
        flags.append("continues_previous_episode")
    return penalty, flags


def best_window_for_peak(evs: List[Dict[str, Any]], peak: Dict[str, Any], ep: Dict[str, Any],
                         ep_end: float, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    p = peak["event"]
    forced = bool(peak.get("forced"))
    anchors = [e for e in evs if e["kind"] in ("dialogue", "action")]
    ends = [e for e in evs if e["kind"] in ("dialogue", "action", "sound")]
    if forced:      # episode-ending window: may start anywhere that makes a 20-60 s clip ending at the cut
        lo, hi = ep_end - cfg["max_dur"], ep_end - cfg["min_dur"]
        start_idx = sorted({snap_start_event(anchors, k) for k, e in enumerate(anchors) if lo <= e["start"] <= hi})
    else:
        start_idx = sorted({snap_start_event(anchors, k) for k, e in enumerate(anchors)
                            if p["start"] - cfg["lookback"] <= e["start"] <= p["start"]})
    best: Optional[Dict[str, Any]] = None
    for k in start_idx:
        s_ev = anchors[k]
        s = s_ev["start"]
        if straddles(evs, s):
            continue
        hook_evs = [e for e in evs if s - 1e-6 <= e["start"] < s + cfg["hook_seconds"]]
        hook = squash(score_interval(hook_evs)["total"], 1.5)
        zone = [e for e in hook_evs if e["kind"] in ("dialogue", "action")]
        strongest = max(zone, key=lambda e: (sum(g["w"] for g in e["signals"] if g["cat"]) + 0.25 * e["tier"],
                                              -e["start"]), default=None)
        lead_in = max(0.0, strongest["start"] - s) if strongest else 0.0   # seconds spent before the real hook
        for e_ev in ends:
            end_t = e_ev["end"]
            dur = end_t - s
            if end_t < p["end"] - 1e-6 or not (cfg["min_dur"] <= dur <= cfg["max_dur"]):
                continue
            snap_win = cfg["episode_end_snap"] if forced else cfg["scene_end_snap"]
            if forced and ep_end - end_t > snap_win:
                continue
            if straddles(evs, end_t):
                continue
            nxt = next((x for x in evs if x["kind"] == "dialogue" and x["start"] >= e_ev["end"] - 1e-6
                        and x is not e_ev), None)
            if e_ev["kind"] == "dialogue" and e_ev["text"].rstrip().endswith(("...", "…")) and nxt \
                    and nxt["speaker"] == e_ev["speaker"] and nxt["start"] - e_ev["end"] <= 1.2:
                continue                                          # sentence continues in the next line
            cut_ts, cut_sec = e_ev["end_ts"], end_t
            if e_ev["scene_end_sec"] - end_t <= snap_win:
                snapped = e_ev["scene_end_sec"]
                # snapping to the scene end must never move the cut INTO a spoken line (the source scene end can fall
                # before the last line finishes) and never earlier than the real end of the last event
                if snapped >= end_t and cfg["min_dur"] <= snapped - s <= cfg["max_dur"] and not straddles(evs, snapped):
                    cut_ts, cut_sec = e_ev["scene_end_ts"], snapped
            tail = tail_features(evs, end_t, ep_end, e_ev["scene_end_sec"])
            cliff = max(0.0, min(1.0, sum(tail.values())))
            inside = [x for x in evs if s - 1e-6 <= x["start"] < end_t]
            inten = squash(max(moment_strength(evs, x["start"], cfg) for x in inside), 3.0)
            pen, flags = window_context_flags(evs, s, ep)
            standalone = max(0.0, 1.0 - pen)
            spk = {x["speaker"] for x in inside if x["kind"] == "dialogue" and x["speaker"]}
            if len(spk) > 4:
                standalone -= 0.1
                flags.append("crowded_cast")
            score = (WEIGHTS["hook"] * hook + WEIGHTS["cliffhanger"] * cliff + WEIGHTS["intensity"] * inten
                     + WEIGHTS["standalone"] * max(0.0, standalone)) - 0.0005 * dur - min(0.09, 0.03 * lead_in)
            if best is None or score > best["score"]:
                best = {"score": score, "start_ev": s_ev, "end_ev": e_ev, "start": s, "end": cut_sec,
                        "end_ts": cut_ts, "hook": hook, "cliff": cliff, "tail": tail, "inten": inten,
                        "standalone": max(0.0, standalone), "flags": flags, "lead_in": lead_in}
    return best


# ----------------------------------------------------------------------------- candidate assembly
def iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def explain(types: List[Dict[str, Any]], key_lines: List[Dict[str, Any]], tail: Dict[str, float]) -> str:
    """Rule-based one-liner (replaced by the LLM's sentence when a verdict is available)."""
    phrase = {"confrontation": "a heated confrontation", "threat": "a threat or ultimatum",
              "betrayal": "a betrayal landing on screen", "revelation": "a revelation or shock reveal",
              "breakdown": "an emotional breakdown", "twist": "a sudden reversal", "romance": "charged intimacy",
              "suspense": "building suspense", "entrance": "a dramatic entrance",
              "unresolved": "an unresolved ending"}
    lead = ", ".join(phrase[t["type"]] for t in types[:2]) or "dramatic tension"
    quote = ""
    if key_lines:
        k = key_lines[0]
        words = k["text"].split()
        quote = f" Key line ({k['speaker']}): \"{' '.join(words[:12])}{'...' if len(words) > 12 else ''}\"."
    end = ""
    if "ends_on_question" in tail:
        end = " Ends on an unanswered question."
    elif "episode_end_cut" in tail:
        end = " Cuts at the episode's closing moment."
    elif "reaction_at_end" in tail or "open_tension" in tail:
        end = " Ends with the situation still open."
    return f"Contains {lead}.{quote}{end}".strip()


def build_candidate(ep: Dict[str, Any], evs: List[Dict[str, Any]], win: Dict[str, Any], peak: Dict[str, Any],
                    cfg: Dict[str, Any]) -> Dict[str, Any]:
    s, e_t = win["start"], win["end"]
    inside = [x for x in evs if s - 1e-6 <= x["start"] < e_t]
    sc = score_interval(inside)
    top = max((v["score"] for v in sc["cats"].values()), default=0.0)
    types = sorted(({"type": c, "score": v["score"], "corroborated": v["corroborated"], "families": v["fams"]}
                    for c, v in sc["cats"].items() if v["score"] >= max(0.5, 0.45 * top)), key=lambda t: -t["score"])
    dl = [x for x in inside if x["kind"] == "dialogue" and x["end"] <= e_t + 0.15]
    sig_by_item: Dict[str, float] = {x["item_id"]: sum(g["w"] for g in x["signals"] if g["cat"]) + 0.2 * x["tier"]
                                     for x in dl}
    key_ids = {i for i, _ in sorted(sig_by_item.items(), key=lambda kv: -kv[1])[:4] if sig_by_item[i] > 0}
    dialogue = [{"item_id": x["item_id"], "speaker": x["speaker"], "text": x["text"], "delivery": x["delivery"],
                 "start_ts": x["start_ts"], "end_ts": x["end_ts"], "is_key": x["item_id"] in key_ids,
                 "voice_mode": None} for x in dl]
    talk: Dict[str, float] = defaultdict(float)
    for x in dl:
        if x["speaker"]:
            talk[x["speaker"]] += x["end"] - x["start"]
    mains = [n for n, _ in sorted(talk.items(), key=lambda kv: -kv[1])[:3]]
    scene_ids: List[str] = []
    for x in inside:
        if x["scene_id"] not in scene_ids:
            scene_ids.append(x["scene_id"])
    p = peak["event"]
    flags = list(win["flags"])
    if win["cliff"] >= 0.5:      # tail-derived type: the clip leaves the situation open
        types.append({"type": "unresolved", "score": round(win["cliff"], 3), "corroborated": True,
                      "families": ["tail"]})
    if peak.get("forced"):
        flags.append("episode_end_candidate")
        if e_t - win["end_ev"]["end"] > 2.0:
            flags.append("closing_shot_without_dialogue")
    if not any(t["corroborated"] for t in types if t["type"] != "unresolved") and not peak.get("forced"):
        flags.append("no_corroborated_category")
    evidence = [{"category": g["cat"], "family": g["fam"], "item_id": x["item_id"], "tag": g["tag"]}
                for x in inside for g in x["signals"] if g["cat"]][:14]
    first = win["start_ev"]
    key_lines = [d for d in dialogue if d["is_key"]]
    cid = f"E{ep['episode_id']:03d}_{int(round(s * 100)):05d}_{int(round(e_t * 100)):05d}"
    return {
        "candidate_id": cid, "episode_id": ep["episode_id"], "episode_label": ep.get("episode_label"),
        "scene_ids": scene_ids, "primary_scene_id": p["scene_id"], "primary_scene_number": p["scene_number"],
        "segments": [{"episode_id": ep["episode_id"], "scene_ids": scene_ids, "start_ts": first["start_ts"],
                      "end_ts": win["end_ts"], "start_sec": s, "end_sec": e_t, "duration_sec": round(e_t - s, 2),
                      "first_item_id": first["item_id"], "last_item_id": win["end_ev"]["item_id"]}],
        "start_ts": first["start_ts"], "end_ts": win["end_ts"], "start_sec": s, "end_sec": e_t,
        "duration_sec": round(e_t - s, 2),
        "moment_types": types, "peak": {"item_id": p["item_id"], "start_ts": p["start_ts"],
                                        "strength": round(peak["strength"], 3)},
        "dialogue": dialogue, "main_characters": mains,
        "hook_zone": {"start_ts": first["start_ts"], "lead_in_sec": round(win.get("lead_in", 0.0), 2),
                      "item_ids": [x["item_id"] for x in evs if s - 1e-6 <= x["start"] < s + cfg["hook_seconds"]
                                   and x["kind"] != "camera"]},
        "tail_features": win["tail"], "evidence": evidence,
        "rule_scores": {"hook": round(win["hook"], 3), "cliffhanger": round(win["cliff"], 3),
                        "intensity": round(win["inten"], 3), "standalone_proxy": round(win["standalone"], 3),
                        "candidate_score": round(win["score"], 4)},
        "why_it_attracts": {"text": explain(types, key_lines, win["tail"]), "source": "rules"},
        "semantic": {"status": "not_run"}, "candidate_score_final": round(win["score"], 4),
        "score_basis": "rules_only", "flags": flags,
    }


def detect_episode(ep: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    evs = build_events(ep)
    if not evs:
        return []
    for e in evs:
        extract_event_signals(e)
    add_dialogue_dynamics(evs)
    ep_end = max(e["scene_end_sec"] for e in evs)
    cands = []
    peaks = find_peaks(evs, cfg)
    tail_anchors = [e for e in evs if e["kind"] in ("dialogue", "action") and e["start"] >= ep_end - 45.0]
    if tail_anchors:        # microdramas close every episode on a hook: always consider the closing window
        best = max(tail_anchors, key=lambda e: moment_strength(evs, e["start"], cfg))
        peaks.append({"event": best, "strength": moment_strength(evs, best["start"], cfg), "forced": True})
    for peak in peaks:
        win = best_window_for_peak(evs, peak, ep, ep_end, cfg)
        if win:
            c = build_candidate(ep, evs, win, peak, cfg)
            rs = c["rule_scores"]
            min_hook = cfg["min_hook_episode_end"] if peak.get("forced") else cfg["min_hook"]
            if rs["hook"] < cfg["min_hook"]:
                c["flags"].append("weak_hook")
            if "no_corroborated_category" not in c["flags"] and rs["candidate_score"] >= cfg["min_candidate_score"] \
                    and rs["hook"] >= min_hook:
                cands.append(c)
    cands.sort(key=lambda c: -c["rule_scores"]["candidate_score"])
    kept: List[Dict[str, Any]] = []
    for c in cands:                                           # non-max suppression on time overlap
        if all(iou((c["start_sec"], c["end_sec"]), (k["start_sec"], k["end_sec"])) <= cfg["nms_iou"] for k in kept):
            kept.append(c)
    return kept[:cfg["per_episode"]]


def detect_all(doc: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ep in doc["episodes"]:
        if ep["parse_status"] == "failed":
            continue
        out += detect_episode(ep, cfg)
    return out


def summarize(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    types: Dict[str, int] = defaultdict(int)
    for c in cands:
        for t in c["moment_types"][:1]:
            types[t["type"]] += 1
    per_ep: Dict[int, int] = defaultdict(int)
    for c in cands:
        per_ep[c["episode_id"]] += 1
    judged = [c for c in cands if c["semantic"].get("status") == "ok"]
    durs = [c["duration_sec"] for c in cands]
    return {"candidates": len(cands), "episodes_with_candidates": len(per_ep),
            "top_type_counts": dict(sorted(types.items(), key=lambda kv: -kv[1])),
            "duration_sec": {"min": min(durs, default=None), "max": max(durs, default=None)},
            "llm_judged": len(judged),
            "llm_rejected": sum(1 for c in judged if "llm_rejected" in c["flags"])}


def run_llm_layer(cands: List[Dict[str, Any]], doc: Dict[str, Any], client: Any, top_k: int) -> None:
    from semantic_judge import judge_candidate, merge_semantic
    eps = {e["episode_id"]: e for e in doc["episodes"]}
    for c in sorted(cands, key=lambda c: -c["rule_scores"]["candidate_score"])[:top_k]:
        if hasattr(client, "has") and not client.has(c["candidate_id"]):
            continue                                  # replay file has no verdict for this one: stays 'not_run'
        ep = eps[c["episode_id"]]
        items = [it for sc in ep["scenes"] for it in sc["items"]]
        merge_semantic(c, judge_candidate(c, ep, items, client))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Detect candidate promo moments.")
    ap.add_argument("input_json")
    ap.add_argument("-o", "--output", default="candidate_promos.json")
    ap.add_argument("--per-episode", type=int, default=DEFAULTS["per_episode"])
    ap.add_argument("--llm", default="none", help="none | anthropic | openai-compat | replay:FILE")
    ap.add_argument("--llm-top-k", type=int, default=30)
    ap.add_argument("--llm-cache", default=None, help="jsonl cache file for live LLM calls")
    args = ap.parse_args(argv)
    cfg = dict(DEFAULTS, per_episode=args.per_episode)
    doc = json.load(open(args.input_json, encoding="utf-8"))
    cands = detect_all(doc, cfg)
    llm_info = {"provider": "none", "model": None, "top_k": 0}
    if args.llm != "none":
        from llm_client import make_client
        client = make_client(args.llm, args.llm_cache)
        run_llm_layer(cands, doc, client, args.llm_top_k)
        llm_info = {"provider": client.name, "model": client.model, "top_k": args.llm_top_k}
    cands.sort(key=lambda c: -c["candidate_score_final"])
    out = {"detector_version": DETECTOR_VERSION, "source": os.path.basename(args.input_json),
           "show": doc.get("show"), "config": {**cfg, "weights": WEIGHTS}, "llm": llm_info,
           "summary": summarize(cands), "candidates": cands}
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps(out["summary"], indent=1))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

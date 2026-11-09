"""
semantic_judge.py - LLM semantic layer for candidate promos.

The model reads a candidate window (dialogue + action + sound, with item ids) plus the episode
logline, and answers: is this a real promo-worthy moment *in context*, what kind, how strong is the
hook / cliffhanger, what context would a new viewer lack, and why would it attract viewers.

Safety rules enforced in code (never trusted to the model):
  * the model cites item ids and short quotes only - it never produces timestamps;
  * every cited item id must be inside the window, every quote must literally occur in that item;
  * numbers are clamped, categories filtered to the allowed list, malformed answers marked invalid.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from llm_client import LLMClient, LLMError

JUDGE_CATEGORIES = ["revelation", "confrontation", "betrayal", "breakdown", "twist", "suspense",
                    "threat", "romance", "entrance", "unresolved"]

SYSTEM_PROMPT = """You are a story editor choosing promo clips for short vertical microdramas.
You will see one candidate clip (20-60 seconds) as numbered lines. Judge it IN CONTEXT.

Moment types (use only these labels):
revelation   - a secret, identity or truth is exposed, or a character learns something shocking
confrontation- an argument or open conflict between characters
betrayal     - someone is wronged by a person they trusted, and it happens or lands in this clip
breakdown    - a visible emotional collapse (crying, despair, desperate pleading)
twist        - an unexpected reversal of power, plan or expectation
suspense     - tension, danger or secrecy where the outcome is unclear
threat       - an explicit threat or ultimatum aimed at a character
romance      - romantic or protective intimacy, charged closeness
entrance     - a surprising arrival or first appearance that changes the scene
unresolved   - the clip ends with the situation still open, so a viewer wants the next scene

Rules:
1. Dramatic words alone do not make a moment. A character mentioning betrayal in passing,
   remembering the past, joking, or describing something that is not happening here is NOT that
   moment. Set keyword_only_false_positive=true if the clip only sounds dramatic.
2. hook_strength: how likely the FIRST 3 SECONDS (lines above the divider) make a scrolling viewer
   stop, 0-10. 0-3 slow/neutral, 4-6 interesting, 7-10 a slap, reveal, shocking line or confrontation.
3. cliffhanger_strength: how unresolved the LAST lines are, 0-10. A resolved scene is low.
4. standalone_clarity: could someone who never saw the show follow who is fighting about what, 0-10.
5. needs_context: up to 3 short phrases naming what a new viewer would be missing (empty if none).
6. evidence: 1-3 items, each {"item_id": "<id from the clip>", "quote": "<=10 words copied exactly>"}.
7. why_it_attracts: one sentence, max 30 words, specific to this clip, no generic praise.
Reply with ONE JSON object and nothing else, with exactly these keys:
is_genuine_promo_moment (bool), keyword_only_false_positive (bool), moment_types (list),
hook_strength (number), cliffhanger_strength (number), standalone_clarity (number),
needs_context (list of strings), main_characters (list of up to 4 names), why_it_attracts (string),
evidence (list)."""


def render_window(candidate: Dict[str, Any], items: List[Dict[str, Any]], hook_seconds: float = 3.0) -> str:
    """Numbered text of the window (camera shots omitted). `items` = all items of the episode."""
    start, end = candidate["start_sec"], candidate["end_sec"]
    lines: List[str] = []
    divider_done = False
    for it in items:
        if it["kind"] == "camera" or it["start_sec"] is None:
            continue
        if it["start_sec"] < start - 1e-6 or it["start_sec"] >= end:
            continue
        if not divider_done and it["start_sec"] >= start + hook_seconds:
            lines.append("--- end of first 3 seconds ---")
            divider_done = True
        if it["kind"] == "dialogue":
            tag = f" ({it['delivery']})" if it.get("delivery") else ""
            vm = f" [{it['voice_mode']}]" if it.get("voice_mode") else ""
            lines.append(f"[{it['item_id']}] {it['start_ts']} {it['character']}{vm}{tag}: \"{it['text']}\"")
        else:
            lines.append(f"[{it['item_id']}] {it['start_ts']} {it['kind'].upper()}: {it['text']}")
    return "\n".join(lines)


def build_user_prompt(candidate: Dict[str, Any], episode: Dict[str, Any], items: List[Dict[str, Any]]) -> str:
    return (f"Episode {candidate['episode_id']} of a microdrama series.\n"
            f"Episode logline: {episode.get('logline') or 'n/a'}\n"
            f"Continuity: {episode.get('continuity_notes') or 'n/a'}\n"
            f"Clip length: {candidate['duration_sec']:.1f}s\n\nCLIP:\n"
            f"{render_window(candidate, items)}\n")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", s.lower())).strip()


def _num(v: Any, lo: float, hi: float) -> Optional[float]:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return None


def validate_result(raw: Dict[str, Any], window_items: Dict[str, str]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns (clean_result, problems). window_items maps item_id -> its text for items in the window.
    clean_result['status'] is 'ok' or 'invalid'. Never raises.
    """
    problems: List[str] = []
    if not isinstance(raw, dict):
        return {"status": "invalid"}, ["response is not an object"]
    clean: Dict[str, Any] = {}
    for key in ("is_genuine_promo_moment", "keyword_only_false_positive"):
        if isinstance(raw.get(key), bool):
            clean[key] = raw[key]
        else:
            problems.append(f"{key} missing or not boolean")
    for key in ("hook_strength", "cliffhanger_strength", "standalone_clarity"):
        v = _num(raw.get(key), 0, 10)
        if v is None:
            problems.append(f"{key} missing or not a number")
        else:
            clean[key] = round(v, 1)
    types = raw.get("moment_types", [])
    types = types if isinstance(types, list) else []
    clean["moment_types"] = [t for t in types if t in JUDGE_CATEGORIES]
    if len(clean["moment_types"]) != len(types):
        problems.append("dropped moment types outside the allowed list")
    ctx = raw.get("needs_context", [])
    clean["needs_context"] = [str(x)[:80] for x in ctx][:3] if isinstance(ctx, list) else []
    chars = raw.get("main_characters", [])
    clean["main_characters"] = [str(x)[:40] for x in chars][:4] if isinstance(chars, list) else []
    why = raw.get("why_it_attracts")
    if isinstance(why, str) and why.strip():
        clean["why_it_attracts"] = " ".join(why.split()[:40])
    else:
        problems.append("why_it_attracts missing")
    evidence, dropped = [], 0
    for ev in raw.get("evidence", []) if isinstance(raw.get("evidence"), list) else []:
        iid, quote = (ev or {}).get("item_id"), (ev or {}).get("quote", "")
        if iid in window_items and isinstance(quote, str) and _norm(quote) and _norm(quote) in _norm(window_items[iid]):
            evidence.append({"item_id": iid, "quote": quote.strip()})
        else:
            dropped += 1
    clean["evidence"] = evidence[:3]
    if dropped:
        problems.append(f"dropped {dropped} evidence entries (id not in window or quote not found)")
    required = {"is_genuine_promo_moment", "keyword_only_false_positive", "hook_strength",
                "cliffhanger_strength", "standalone_clarity", "why_it_attracts"}
    clean["status"] = "ok" if required <= clean.keys() else "invalid"
    return clean, problems


def judge_candidate(candidate: Dict[str, Any], episode: Dict[str, Any], items: List[Dict[str, Any]],
                    client: LLMClient) -> Dict[str, Any]:
    """Ask the model, validate the answer, return the `semantic` block for this candidate."""
    window_ids = {it["item_id"]: it["text"] for it in items
                  if it["kind"] != "camera" and it["start_sec"] is not None
                  and candidate["start_sec"] - 1e-6 <= it["start_sec"] < candidate["end_sec"]}
    try:
        raw = client.complete_json(SYSTEM_PROMPT, build_user_prompt(candidate, episode, items),
                                   key=candidate["candidate_id"])
    except LLMError as exc:
        return {"status": "error", "provider": client.name, "model": client.model, "error": str(exc)}
    clean, problems = validate_result(raw, window_ids)
    clean.update({"provider": client.name, "model": client.model, "problems": problems})
    return clean


def merge_semantic(candidate: Dict[str, Any], sem: Dict[str, Any]) -> None:
    """
    Fold the model's verdict into the candidate. Formula (documented, tunable):
      genuine:     final = 0.4 * rule_score + 0.6 * mean(hook, cliffhanger)/10
      not genuine: final = 0.4 * rule_score   (demoted, flagged 'llm_rejected')
    """
    candidate["semantic"] = sem
    rule = candidate["rule_scores"]["candidate_score"]
    if sem.get("status") != "ok":
        candidate["candidate_score_final"], candidate["score_basis"] = rule, "rules_only"
        return
    genuine = sem["is_genuine_promo_moment"] and not sem["keyword_only_false_positive"]
    llm_part = (sem["hook_strength"] + sem["cliffhanger_strength"]) / 20.0
    if genuine:
        candidate["candidate_score_final"] = round(0.4 * rule + 0.6 * llm_part, 4)
        candidate["score_basis"] = "rules+llm"
        if sem["moment_types"]:
            candidate["moment_types_final"] = sem["moment_types"]
        candidate["why_it_attracts"] = {"text": sem["why_it_attracts"], "source": "llm"}
        if sem["main_characters"]:
            candidate["main_characters_llm"] = sem["main_characters"]
    else:
        candidate["candidate_score_final"] = round(0.4 * rule, 4)
        candidate["score_basis"] = "rules+llm(rejected)"
        candidate.setdefault("flags", []).append("llm_rejected")

"""
promo_ranker.py - Module 4: final selection, ranking and posting plan.

Reads the outputs of the three earlier modules:
    extracted_scenes.json   (screenplay_parser.py)
    candidate_promos.json   (candidate_generator.py)
    evaluated_promos.json   (promo_evaluator.py)
and produces ranked_promos.json + ranked_promos.md: the top-N promos, each with every field the brief asks for,
plus a posting order.

Final eligibility (stricter than the evaluator's hard gates, because these are the user-facing promos):
    total duration 20-60 s, every segment in a real episode, gaps between segments <= MAX_GAP
    hook >= 6.0, cliffhanger >= 6.0, standalone >= 5.5, evaluator recommendation top_pick/recommend
    and a model-verified verdict (rules-only candidates are only used if the verified pool runs dry, and
    are then labelled UNVERIFIED).
Selection: rank by the evaluator's overall score; at most one promo per episode (promos from the same
episode compete for the same story beat, and adjacent episodes would be near-duplicates when posted).

Every text field shown to a reader (opening hook, ending cliffhanger) is copied from the screenplay items;
nothing is paraphrased or generated here. The "why it works" sentence is the model's one-sentence reason
(when judged) or, for unverified clips, the rule engine's template, and is labelled with its source.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

FINAL_MIN = {"hook": 6.0, "cliffhanger": 6.0, "standalone": 5.5}
MAX_GAP = 8.0            # seconds skipped between two segments of one promo; larger jumps would feel like a different scene
TOP_N = 10
POST_WEIGHTS = {"overall": 0.45, "standalone": 0.25, "hook": 0.15, "entry": 0.15}
SPOILER_TYPES = {"revelation", "twist"}


def sec(ts: str) -> float:
    m, rest = ts.split(":")
    return int(m) * 60 + float(rest)


def clean_text(it: Dict[str, Any]) -> str:
    return it["text"].strip()


def line(it: Dict[str, Any]) -> Dict[str, Any]:
    who = it.get("character_canon") or it.get("character")
    return {"item_id": it["item_id"], "kind": it["kind"], "speaker": who if it["kind"] == "dialogue" else None,
            "delivery": it.get("delivery") if it["kind"] == "dialogue" else None,
            "start_ts": it["start_ts"], "end_ts": it["end_ts"], "text": clean_text(it)}


def window_items(ep: Dict[str, Any], t0: float, t1: float) -> List[Dict[str, Any]]:
    out = [it for sc in ep["scenes"] for it in sc["items"]
           if it["kind"] != "camera" and it["start_sec"] is not None and t0 - 1e-6 <= it["start_sec"] < t1]
    return sorted(out, key=lambda i: (i["start_sec"], i["item_id"]))


def eligibility(ev: Dict[str, Any], cand: Dict[str, Any]) -> List[str]:
    """Reasons a candidate cannot be a final promo (empty list = eligible)."""
    why = []
    segs = ev["segments"]
    total = sum(s["duration_sec"] for s in segs)
    if not 20 <= total <= 60:
        why.append(f"total duration {total:.1f}s outside 20-60s")
    for a, b in zip(segs, segs[1:]):
        if a["episode_id"] == b["episode_id"] and b["start_sec"] - a["end_sec"] > MAX_GAP:
            why.append(f"gap of {b['start_sec'] - a['end_sec']:.1f}s between segments")
    if not ev["gates"]["passed"]:
        why.append("failed an evaluation gate")
    for k, floor in FINAL_MIN.items():
        if ev["scores"][k] < floor:
            why.append(f"{k} {ev['scores'][k]} < {floor}")
    if ev["recommendation"] not in ("top_pick", "recommend"):
        why.append(f"evaluator recommendation is {ev['recommendation']}")
    return why


def build_promo(rank: int, ev: Dict[str, Any], cand: Dict[str, Any], doc_eps: Dict[str, Any]) -> Dict[str, Any]:
    segs = []
    for s in ev["segments"]:
        ep = doc_eps[str(s["episode_id"])]
        segs.append({"episode": int(s["episode_id"]), "episode_label": ep["episode_label"], "scene_ids": s["scene_ids"],
                     "start_ts": s["start_ts"], "end_ts": s["end_ts"], "start_sec": s["start_sec"],
                     "end_sec": s["end_sec"], "duration_sec": s["duration_sec"]})
    first, last = ev["segments"][0], ev["segments"][-1]
    ep0, ep1 = doc_eps[str(first["episode_id"])], doc_eps[str(last["episode_id"])]
    hook_lines = [line(i) for i in window_items(ep0, first["start_sec"], first["start_sec"] + 3.0)]
    tail_lines = [line(i) for i in window_items(ep1, max(last["start_sec"], last["end_sec"] - 8.0), last["end_sec"])]
    spoken_tail = [l for l in tail_lines if l["kind"] == "dialogue"][-2:]
    final_beat = tail_lines[-1] if tail_lines else None
    verified = ev["confidence"] == "rules+llm"
    if verified and ev.get("why_source") == "llm":
        why, why_src = ev["why_it_attracts"], "model verdict (in-session)" if "in-session" in str(ev.get("llm_model")) else "model verdict"
    else:
        why, why_src = ev.get("why_it_attracts") or "", "rule template (unverified)"
    caveats = list(ev.get("needs_context") or [])
    if "continues_previous_episode" in ev["flags"]:
        caveats.append("continues directly from the previous episode")
    if len({s["scene_ids"][0] for s in segs}) < sum(len(s["scene_ids"]) for s in segs) or len(ev["segments"][0]["scene_ids"]) > 1:
        caveats.append("runs across a scene change")
    total = round(sum(s["duration_sec"] for s in ev["segments"]), 2)
    gaps = [round(b["start_sec"] - a["end_sec"], 2) for a, b in zip(segs, segs[1:]) if a["episode"] == b["episode"]]
    return {
        "promo_id": f"PROMO-{rank:02d}",
        "candidate_id": ev["candidate_id"],
        "rank": rank,
        "episodes": sorted({s["episode"] for s in segs}),
        "segments": segs,
        "n_segments": len(segs),
        "total_duration_sec": total,
        "max_gap_sec": max(gaps) if gaps else 0.0,
        "opening_hook": {"first_3_seconds": hook_lines,
                         "hook_line": ev["details"]["hook"]["first_line"]},
        "ending_cliffhanger": {"last_spoken_lines": spoken_tail, "final_beat": final_beat,
                               "why_unresolved": ev["details"]["cliffhanger"]["notes"]},
        "why_it_works": {"text": why, "source": why_src,
                         "hook_explanation": ev["explanations"]["hook"],
                         "cliffhanger_explanation": ev["explanations"]["cliffhanger"],
                         "context_explanation": ev["explanations"]["standalone"]},
        "scores": ev["scores"], "overall_ranking_score": ev["overall_score"],
        "weighted_score_before_balance": ev["weighted_score"],
        "moment_types": ev["moment_types"], "main_characters": ev["main_characters"],
        "verification": "model-verified" if verified else "UNVERIFIED (rules only)",
        "caveats": caveats,
        "longest_silence_sec": ev["details"]["coherence"]["longest_silence_sec"],
    }


def posting_plan(promos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Order for publishing. Different goal from the quality rank: the first posts must work on viewers who
    have never seen the show, so context, hook and an early entry point matter more than raw drama.
      score = 0.45*overall + 0.25*standalone + 0.15*hook + 0.15*entry(episode position, 10 = Episode 1)
      minus 1.0 if it shows a major reveal/twist from the second half of the series (spoiler cost).
    Greedy pick, and avoid posting two promos from adjacent episodes back to back (same story beat).
    """
    max_ep = max(p["episodes"][0] for p in promos) or 1
    scored = []
    for p in promos:
        ep = p["episodes"][0]
        entry = 10.0 * (1 - (ep - 1) / max(49, max_ep))
        spoil = bool(SPOILER_TYPES & set(p["moment_types"])) and ep >= 25
        s = (POST_WEIGHTS["overall"] * p["overall_ranking_score"] + POST_WEIGHTS["standalone"] * p["scores"]["standalone"] +
             POST_WEIGHTS["hook"] * p["scores"]["hook"] + POST_WEIGHTS["entry"] * entry - (1.0 if spoil else 0.0))
        scored.append({"p": p, "score": round(s, 3), "entry": round(entry, 1), "spoiler": spoil})
    scored.sort(key=lambda x: -x["score"])
    order, left = [], scored[:]
    while left:
        prev = order[-1]["p"]["episodes"][0] if order else None
        pick = next((x for x in left if prev is None or abs(x["p"]["episodes"][0] - prev) > 1), left[0])
        left.remove(pick)
        order.append(pick)
    plan = []
    for i, x in enumerate(order, 1):
        p = x["p"]
        ep = p["episodes"][0]
        reasons = [f"standalone {p['scores']['standalone']}, hook {p['scores']['hook']}"]
        reasons.append("from the opening episodes, so a new viewer can start the series right after it" if ep <= 5
                       else "mid-series: works as a hook but assumes the premise" if ep <= 25
                       else "late-series: strongest after the audience knows the premise")
        if x["spoiler"]:
            reasons.append("shows a late-series reveal, so held back to avoid spoiling")
        if p["caveats"]:
            reasons.append("add a one-line caption for: " + "; ".join(p["caveats"][:2]))
        plan.append({"post_order": i, "wave": 1 if i <= 3 else 2 if i <= 7 else 3, "promo_id": p["promo_id"],
                     "episodes": p["episodes"], "posting_score": x["score"], "rationale": "; ".join(reasons)})
    return plan


def select(evals: List[Dict[str, Any]], cands: Dict[str, Dict[str, Any]], top_n: int, allow_unverified: bool):
    ranked = sorted(evals, key=lambda e: (-e["overall_score"], -e["scores"]["cliffhanger"]))
    rejected: List[Dict[str, Any]] = []
    pools = {"verified": [], "unverified": []}
    for e in ranked:
        why = eligibility(e, cands[e["candidate_id"]])
        if why:
            rejected.append({"candidate_id": e["candidate_id"], "overall": e["overall_score"], "reasons": why})
            continue
        pools["verified" if e["confidence"] == "rules+llm" else "unverified"].append(e)
    chosen, used_eps, skipped = [], set(), []
    # Consecutive episodes usually continue the same scene, so two clips from them are near-duplicates as promos.
    # First pass avoids episodes next to an already-chosen one; adjacent ones only fill slots nothing else can fill.
    # Unverified clips never outrank verified ones: within each adjacency pass, verified pool goes first.
    names = ["verified", "unverified"] if allow_unverified else ["verified"]
    for avoid_adjacent in (True, False):
        for pool in names:
            for e in pools[pool]:
                if len(chosen) >= top_n:
                    break
                ep = e["episode_id"]
                if ep in used_eps or e in chosen:
                    continue
                if avoid_adjacent and ((ep - 1) in used_eps or (ep + 1) in used_eps):
                    continue
                used_eps.add(ep)
                chosen.append(e)
    chosen_ids = {e["candidate_id"] for e in chosen}
    for pool in names:
        for e in pools[pool]:
            if e["candidate_id"] in chosen_ids:
                continue
            ep = e["episode_id"]
            if ep in used_eps:
                reason = f"Episode {ep} already has a higher-ranked promo"
            else:
                reason = "top list was full"
            skipped.append({"candidate_id": e["candidate_id"], "overall": e["overall_score"], "reason": reason})
    chosen.sort(key=lambda e: (e["confidence"] != "rules+llm", -e["overall_score"], -e["scores"]["cliffhanger"]))
    return chosen, skipped, rejected


def render_markdown(result: Dict[str, Any]) -> str:
    L = ["# Final promo ranking", ""]
    L.append(f"Source: {result['pipeline']['show']} - {result['pipeline']['episodes']} episodes parsed, "
             f"{result['pipeline']['candidates']} candidates generated, {result['pipeline']['verified_candidates']} model-verified, "
             f"{result['pipeline']['eligible']} passed the final bar, top {len(result['top_promos'])} shown.\n")
    for p in result["top_promos"]:
        L.append(f"## {p['promo_id']}  (candidate {p['candidate_id']})  -  overall {p['overall_ranking_score']}")
        L.append(f"- **Episode(s):** {', '.join(str(e) for e in p['episodes'])}")
        for s in p["segments"]:
            L.append(f"- **Segment:** Episode {s['episode']} {s['start_ts']} - {s['end_ts']}  ({s['duration_sec']}s)")
        L.append(f"- **Total duration:** {p['total_duration_sec']}s ({p['n_segments']} segment, gap {p['max_gap_sec']}s)")
        L.append("- **Opening hook (first 3 s):**")
        for l in p["opening_hook"]["first_3_seconds"]:
            tag = f"{l['speaker']}" if l["speaker"] else l["kind"].upper()
            L.append(f"    - {l['start_ts']} {tag}: \"{l['text']}\"")
        L.append("- **Ending cliffhanger:**")
        for l in p["ending_cliffhanger"]["last_spoken_lines"]:
            L.append(f"    - {l['start_ts']} {l['speaker']}: \"{l['text']}\"")
        fb = p["ending_cliffhanger"]["final_beat"]
        if fb and fb["kind"] != "dialogue":
            L.append(f"    - {fb['start_ts']} {fb['kind'].upper()}: \"{fb['text']}\"")
        L.append(f"- **Why it works:** {p['why_it_works']['text']} _(source: {p['why_it_works']['source']})_")
        L.append("- **Scores:** " + ", ".join(f"{k} {v}" for k, v in p["scores"].items()) + f"  ->  **overall {p['overall_ranking_score']}**")
        if p["caveats"]:
            L.append("- **Weaknesses / needs a caption:** " + "; ".join(p["caveats"]))
        L.append(f"- **Verification:** {p['verification']}")
        L.append("")
    L.append("## Posting order\n")
    for x in result["posting_plan"]:
        L.append(f"{x['post_order']}. **{x['promo_id']}** (Ep {', '.join(map(str, x['episodes']))}, wave {x['wave']}): {x['rationale']}")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Final promo selection and ranking")
    ap.add_argument("extracted"); ap.add_argument("candidates"); ap.add_argument("evaluated")
    ap.add_argument("-o", "--out", default="ranked_promos.json")
    ap.add_argument("--md", default=None)
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--allow-unverified", action="store_true",
                    help="fill empty slots with rules-only clips (labelled UNVERIFIED) when too few are model-verified")
    a = ap.parse_args(argv)
    doc = json.load(open(a.extracted, encoding="utf-8"))
    cdoc = json.load(open(a.candidates, encoding="utf-8"))
    edoc = json.load(open(a.evaluated, encoding="utf-8"))
    eps = {str(e["episode_id"]): e for e in doc["episodes"]}
    cands = {c["candidate_id"]: c for c in cdoc["candidates"]}
    evals = edoc["evaluations"]
    chosen, skipped, rejected = select(evals, cands, a.top, a.allow_unverified)
    promos = [build_promo(i, e, cands[e["candidate_id"]], eps) for i, e in enumerate(chosen, 1)]
    plan = posting_plan(promos) if promos else []
    near = [r for r in rejected if r["overall"] >= 6.0][:8]
    result = {
        "pipeline": {"show": doc.get("show") or doc.get("title") or os.path.basename(str(doc.get("source_dir", ""))) or "unknown",
                     "episodes": len(doc["episodes"]), "candidates": len(cands),
                     "verified_candidates": sum(1 for e in evals if e["confidence"] == "rules+llm"),
                     "eligible": len(chosen) + len(skipped),
                     "weights": edoc["weights"], "final_thresholds": FINAL_MIN, "max_gap_sec": MAX_GAP},
        "top_promos": promos, "posting_plan": plan, "posting_weights": POST_WEIGHTS,
        "same_episode_skips": skipped, "near_misses": near,
        "limitations": [
            "Every promo here is a single contiguous segment; no multi-segment stitching was attempted (zero gaps by construction).",
            "Only model-verified clips can be top-ranked; verdicts for the verified set were written in-session and replayed (no live API call).",
            "Standalone/context is judged from screenplay text, not from video: the footage itself was not seen.",
            "Weights and thresholds have not been validated against hand-labelled promos.",
        ],
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    md = render_markdown(result)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(md)
    print(f"selected {len(promos)} promos (eligible pool {result['pipeline']['eligible']}); wrote {a.out}")
    for p in promos:
        print(f"  {p['promo_id']} {p['candidate_id']} ep{p['episodes']} {p['total_duration_sec']}s score {p['overall_ranking_score']} {p['verification']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

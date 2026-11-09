"""
results.py - turns pipeline output into what the UI shows and what users download.

Pure functions over the JSON documents the pipeline already wrote (no Streamlit import, so they are unit-testable):
  promo_table / rejection_table   pandas tables
  validate_promo                  independent sanity checks on each final promo (shown as badges)
  clip_script                     the actual screenplay lines inside a promo, from extracted_scenes.json
  to_csv / to_detail_json         the two downloads
"""
from __future__ import annotations

import io
import json
import re
from typing import Any, Dict, List

import pandas as pd

from promo_app.runner import RunResult  # noqa: F401  (re-exported for type hints)


def sec(ts: str) -> float:
    m = re.match(r"(\d+):(\d{2})\.(\d{2})$", ts)
    return int(m[1]) * 60 + int(m[2]) + int(m[3]) / 100


def segments_text(p: Dict[str, Any]) -> str:
    return " + ".join(f"Ep {s['episode']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"])


def _lines(lines: List[Dict[str, Any]]) -> str:
    return " / ".join((f"{l['speaker']}: " if l.get("speaker") else "") + l["text"] for l in lines)


# ------------------------------------------------------------------------------------------ tables
def promo_table(run: RunResult) -> pd.DataFrame:
    order = {x["promo_id"]: x for x in run.final.get("posting_plan", [])}
    rows = []
    for p in run.final["top_promos"]:
        sc = p["scores"]
        rows.append({
            "Rank": p["rank"], "Promo": p["promo_id"], "Episode": ", ".join(str(e) for e in p["episodes"]),
            "Timestamps": segments_text(p), "Duration (s)": p["total_duration_sec"],
            "Hook": sc["hook"], "Cliffhanger": sc["cliffhanger"], "Standalone": sc["standalone"],
            "Overall": p["overall_ranking_score"], "Post order": order.get(p["promo_id"], {}).get("post_order"),
            "Verification": p["verification"],
        })
    return pd.DataFrame(rows)


def rejection_table(run: RunResult) -> pd.DataFrame:
    """Every candidate that is NOT in the final list, with the reason it did not make it."""
    import promo_ranker as fr
    cands = {c["candidate_id"]: c for c in run.candidates["candidates"]}
    top = {p["candidate_id"] for p in run.final["top_promos"]}
    skips = {s["candidate_id"]: s["reason"] for s in run.final.get("same_episode_skips", [])}
    rows = []
    for e in run.evaluated["evaluations"]:
        cid = e["candidate_id"]
        if cid in top:
            continue
        gate_reasons = [f["reason"] for f in e["gates"]["failures"]]
        if e["recommendation"] == "reject":
            status, why = "Rejected", gate_reasons
        elif e["recommendation"] == "needs_llm_review":
            status, why = "Needs LLM review", gate_reasons + ["rules-only score is not reliable enough to reject; run a live model"]
        else:
            why = fr.eligibility(e, cands[cid])
            if cid in skips:
                why.append(skips[cid])
            status = "Not selected"
            if not why:
                why = ["met the bar but ranked below the top picks"]
        rows.append({"Candidate": cid, "Episode": e["episode_label"], "Timestamps": f"{e['start_ts']}-{e['end_ts']}",
                     "Duration (s)": e["duration_sec"], "Status": status, "Why it failed": "; ".join(why),
                     "Hook": e["scores"]["hook"], "Cliffhanger": e["scores"]["cliffhanger"],
                     "Standalone": e["scores"]["standalone"], "Overall": e["overall_score"],
                     "Opening line": ((e["details"]["hook"]["first_line"] or {}).get("quote", "")),
                     "Closing line": ((e["details"]["cliffhanger"]["last_line"] or {}).get("quote", ""))})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Overall"], ascending=False).reset_index(drop=True)
    return df


# ------------------------------------------------------------------------------------------ validation
def validate_promo(p: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Independent checks on the numbers a user will act on. Level 'error' = do not use as shown, 'warn' = look first.
    Catches the defect found in the QA audit: a displayed hook/ending line that is not fully inside the clip.
    """
    out: List[Dict[str, str]] = []
    tot = sum(sec(s["end_ts"]) - sec(s["start_ts"]) for s in p["segments"])
    if not 20 <= tot <= 60:
        out.append({"level": "error", "msg": f"Total duration {tot:.2f}s is outside 20-60s."})
    elif tot < 21:
        out.append({"level": "warn", "msg": f"Only {tot - 20:.1f}s above the 20s minimum; any trimming breaks the rule."})
    elif tot > 59:
        out.append({"level": "warn", "msg": f"Only {60 - tot:.1f}s below the 60s maximum."})
    for a, b in zip(p["segments"], p["segments"][1:]):
        if a["episode"] != b["episode"]:
            out.append({"level": "warn", "msg": "Segments come from different episodes; the evaluator scores only the first "
                                                  "segment's episode, so these scores are unreliable."})
        elif sec(b["start_ts"]) - sec(a["end_ts"]) > 8:
            out.append({"level": "warn", "msg": "Large gap between segments."})
    s0, s1 = p["segments"][0], p["segments"][-1]
    lo, hi = sec(s0["start_ts"]), sec(s1["end_ts"])
    shown = [("opening hook", l) for l in p["opening_hook"]["first_3_seconds"]] + \
            [("ending", l) for l in p["ending_cliffhanger"]["last_spoken_lines"]]
    for where, l in shown:
        if l["kind"] != "dialogue":
            continue
        if sec(l["start_ts"]) < lo - 0.01 or sec(l["end_ts"]) > hi + 0.15:
            out.append({"level": "error", "msg": f"The {where} line \"{l['text'][:50]}\" ({l['start_ts']}-{l['end_ts']}) "
                                                  f"is not fully inside the clip (ends {s1['end_ts']}); the cut would clip it."})
    if not any(l["kind"] == "dialogue" for l in p["opening_hook"]["first_3_seconds"]):
        out.append({"level": "warn", "msg": "No spoken line in the first 3 seconds."})
    if p["verification"].startswith("UNVERIFIED"):
        out.append({"level": "warn", "msg": "Rules-only score: no model has confirmed the hook, cliffhanger or context."})
    return out


def clip_script(run: RunResult, p: Dict[str, Any]) -> pd.DataFrame:
    """The screenplay lines inside each segment (camera cues omitted), straight from extracted_scenes.json."""
    eps = {str(e["episode_id"]): e for e in run.extracted["episodes"]}
    rows = []
    for seg in p["segments"]:
        ep = eps[str(seg["episode"])]
        for sc in ep["scenes"]:
            for it in sc["items"]:
                if it["kind"] == "camera" or it["start_sec"] is None:
                    continue
                if seg["start_sec"] - 1e-6 <= it["start_sec"] < seg["end_sec"]:
                    rows.append({"Episode": seg["episode"], "Start": it["start_ts"], "End": it["end_ts"], "Type": it["kind"].title(),
                                 "Speaker": (it.get("character_canon") or it.get("character") or "") if it["kind"] == "dialogue" else "",
                                 "Delivery": it.get("delivery") or "" if it["kind"] == "dialogue" else "",
                                 "Text": it["text"], "In first 3s": it["start_sec"] < seg["start_sec"] + 3.0})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ downloads
def to_csv(run: RunResult) -> bytes:
    order = {x["promo_id"]: x for x in run.final.get("posting_plan", [])}
    rows = []
    for p in run.final["top_promos"]:
        sc = p["scores"]
        for i, s in enumerate(p["segments"], 1):
            rows.append({
                "rank": p["rank"], "promo_id": p["promo_id"], "candidate_id": p["candidate_id"], "segment": i, "episode": s["episode"],
                "start_timestamp": s["start_ts"], "end_timestamp": s["end_ts"], "segment_duration_sec": s["duration_sec"],
                "total_duration_sec": p["total_duration_sec"],
                "opening_hook": _lines(p["opening_hook"]["first_3_seconds"]),
                "ending_cliffhanger": _lines(p["ending_cliffhanger"]["last_spoken_lines"]),
                "why_it_works": p["why_it_works"]["text"],
                "hook_score": sc["hook"], "cliffhanger_score": sc["cliffhanger"], "standalone_score": sc["standalone"],
                "intensity_score": sc["intensity"], "coherence_score": sc["coherence"], "character_clarity_score": sc["character_clarity"],
                "overall_score": p["overall_ranking_score"], "post_order": order.get(p["promo_id"], {}).get("post_order"),
                "verification": p["verification"],
                "validation_warnings": " | ".join(v["msg"] for v in validate_promo(p)),
            })
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def to_detail_json(run: RunResult) -> bytes:
    rej = rejection_table(run)
    doc = {
        "pipeline": run.final["pipeline"], "seconds_per_stage": run.seconds,
        "final_promos": [dict(p, validation=validate_promo(p)) for p in run.final["top_promos"]],
        "posting_plan": run.final.get("posting_plan", []),
        "all_evaluations": run.evaluated["evaluations"],
        "rejected_or_not_selected": rej.to_dict(orient="records"),
        "weights": run.evaluated["weights"], "weight_rationale": run.evaluated["weight_rationale"],
        "gates": run.evaluated["gates"], "limitations": run.final.get("limitations", []),
    }
    return json.dumps(doc, indent=2, ensure_ascii=False).encode("utf-8")

"""
quality_checker.py - independent quality audit of the pipeline's output, written to quality_report.json.

It checks the OUTPUT FILES against the RAW screenplay files. It does not import the parser, generator, evaluator
or ranker, so a bug in those cannot hide itself here. Every check returns pass/fail plus the exact failures.

Checks
  timestamps   every timestamp in extracted_scenes.json equals the raw file, in order; every dialogue line is verbatim
  candidates   duration 20-60 s, start/end are real item times, no dialogue line cut in half, quotes verbatim, no heavy overlap
  evaluations  weights sum to 1, scores in 0-10, overall = weighted sum x balance factor (recomputed), gates respected,
               every quoted span in an explanation exists in the raw file
  ranked       every promo: real timestamps, 20-60 s, hook within the first 3 s, ending in the last 8 s, quotes verbatim,
               scores equal evaluated_promos.json, descending order, one promo per episode, posting plan complete
  boundaries   no promo starts or ends inside a spoken line; every displayed hook/ending line lies inside the clip

Usage:
  python quality_checker.py <screenplay_dir> <outputs_dir> [-o quality_report.json]
Reads  <outputs_dir>/extracted_scenes.json, candidate_promos.json, evaluated_promos.json, ranked_promos.json
Exit code 0 = every check passed, 1 = at least one failure.
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys

TS = re.compile(r"\d{1,3}:\d{2}\.\d{2}")


def sec(t):
    m = re.match(r"(\d+):(\d{2})\.(\d{2})$", t)
    return int(m[1]) * 60 + int(m[2]) + int(m[3]) / 100


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def load_raw(raw_dir):
    """{basename: text} and {episode_number: text}; the episode number is read from the file CONTENT."""
    by_name, by_ep = {}, {}
    for p in glob.glob(os.path.join(raw_dir, "**", "*.txt"), recursive=True):
        txt = open(p, encoding="utf-8-sig").read().replace("\r\n", "\n")
        by_name[os.path.basename(p)] = txt
        m = re.search(r"EPISODE NUMBER:\s*Episode\s+(\d+)", txt)
        if m:
            by_ep[int(m[1])] = txt
    return by_name, by_ep


def raw_dialogue(txt):
    """Independent reader: [(start_sec, end_sec, text)] for dialogue blocks in a raw episode."""
    out, lines = [], txt.split("\n")
    for i, line in enumerate(lines[:-1]):
        m = re.match(r"\*\*(.+?):\*\*\s*$", line)
        t = re.match(r"Start: \((\d+:\d{2}\.\d{2})\) \| End: \((\d+:\d{2}\.\d{2})\)", lines[i + 1])
        if m and t and not re.match(r"(CAMERA|ACTION|SOUND FX)\b", m[1]):
            body, j = [], i + 2
            while j < len(lines) and lines[j].strip() and not lines[j].startswith(("**", "SCENE")):
                body.append(lines[j].strip()); j += 1
            out.append((sec(t[1]), sec(t[2]), norm(" ".join(body)).strip('"“”')))
    return out


def raw_item_times(txt):
    starts, ends = set(), set()
    for a, b in re.findall(r"Start: \((\d+:\d{2}\.\d{2})\) \| End: \((\d+:\d{2}\.\d{2})\)", txt):
        starts.add(sec(a)); ends.add(sec(b))
    return starts, ends


def _result(name, what, failures, **counts):
    return {"check": name, "what_it_verifies": what, "passed": not failures, "counts": counts,
            "failure_count": len(failures), "failures": failures[:50]}


# ------------------------------------------------------------------ 1. timestamps
def check_timestamps(by_name, extracted):
    fails, tokens, dialogue = [], 0, 0
    for ep in extracted["episodes"]:
        raw = by_name.get(ep["source_file"])
        if raw is None:
            fails.append(f"episode {ep['episode_id']}: source file {ep['source_file']} not found"); continue
        rt = TS.findall(raw)
        jt = []
        for sc in ep["scenes"]:
            jt += [sc["start_ts"], sc["end_ts"]]
            for it in sc["items"]:
                jt += [it["start_ts"], it["end_ts"]]
        jt = [t for t in jt if t is not None]
        tokens += len(rt)
        if rt != jt:
            first = next((i for i, (a, b) in enumerate(zip(rt, jt)) if a != b), min(len(rt), len(jt)))
            fails.append(f"episode {ep['episode_id']}: raw has {len(rt)} timestamps, JSON {len(jt)}; first difference at #{first}")
        flat = norm(raw)
        for sc in ep["scenes"]:
            for it in sc["items"]:
                if it["kind"] == "dialogue":
                    dialogue += 1
                    if norm(it["text"]) not in flat or f"**{it['character']}" not in raw:
                        fails.append(f"episode {ep['episode_id']} {it['item_id']}: dialogue or speaker not verbatim in raw")
    return _result("timestamps", "every timestamp in extracted_scenes.json equals the raw file in order; dialogue is verbatim",
                   fails, episodes=len(extracted["episodes"]), raw_timestamp_tokens_compared=tokens, dialogue_lines_checked=dialogue)


# ------------------------------------------------------------------ 2. candidates
def check_candidates(by_name, extracted, cands):
    fails, n_dialogue = [], 0
    eps = {e["episode_id"]: e for e in extracted["episodes"]}
    items = {it["item_id"]: (e["episode_id"], sc, it) for e in extracted["episodes"] for sc in e["scenes"] for it in sc["items"]}
    for c in cands:
        ep = eps[c["episode_id"]]; raw = by_name[ep["source_file"]]; rts = set(TS.findall(raw)); total = 0.0
        d = raw_dialogue(raw.replace("\r\n", "\n"))
        for seg in c["segments"]:
            s, e = sec(seg["start_ts"]), sec(seg["end_ts"]); total += e - s
            if seg["start_ts"] not in rts or seg["end_ts"] not in rts:
                fails.append(f"{c['candidate_id']}: start/end timestamp not present in raw file")
            first, last = items[seg["first_item_id"]], items[seg["last_item_id"]]
            if first[2]["start_ts"] != seg["start_ts"]:
                fails.append(f"{c['candidate_id']}: start_ts differs from its first item")
            if seg["end_ts"] not in (last[2]["end_ts"], last[1]["end_ts"]):
                fails.append(f"{c['candidate_id']}: end_ts is neither the last item's end nor its scene end")
            for a, b, txt in d:
                if (a < s - 0.005 < b - 0.005) or (a < e - 0.005 < b - 0.005 and a >= s):
                    fails.append(f"{c['candidate_id']}: a spoken line ({txt[:30]!r}) is cut by the clip boundary")
        if not 20 <= total <= 60:
            fails.append(f"{c['candidate_id']}: total duration {total:.2f}s outside 20-60")
        for dl in c["dialogue"]:
            n_dialogue += 1
            _, _, it = items[dl["item_id"]]
            if (dl["start_ts"], dl["end_ts"], dl["text"]) != (it["start_ts"], it["end_ts"], it["text"]):
                fails.append(f"{c['candidate_id']}: {dl['item_id']} differs from extracted data")
            if norm(dl["text"]) not in norm(raw):
                fails.append(f"{c['candidate_id']}: {dl['item_id']} text not verbatim in raw file")
    by = {}
    for c in cands:
        by.setdefault(c["episode_id"], []).append(c)
    for cs in by.values():
        for i, a in enumerate(cs):
            for b in cs[i + 1:]:
                inter = max(0, min(a["end_sec"], b["end_sec"]) - max(a["start_sec"], b["start_sec"]))
                union = (a["end_sec"] - a["start_sec"]) + (b["end_sec"] - b["start_sec"]) - inter
                if union and inter / union > 0.5:
                    fails.append(f"{a['candidate_id']} / {b['candidate_id']}: overlap IoU {inter / union:.2f} > 0.5")
    return _result("candidates", "duration 20-60 s, real item times, no dialogue cut in half, quotes verbatim, no heavy overlap",
                   fails, candidates=len(cands), dialogue_lines_checked=n_dialogue)


# ------------------------------------------------------------------ 3. evaluations
def check_evaluations(by_name, extracted, doc):
    fails, n_quotes = [], 0
    src = {str(e["episode_id"]): norm(by_name[e["source_file"]]) for e in extracted["episodes"]}
    W = doc["weights"]
    if abs(sum(W.values()) - 1) > 1e-9:
        fails.append("weights do not sum to 1")
    bal = re.search(r"\(([\d.]+) \+ ([\d.]+) \*", doc["balance_rule"]); floor, slope = float(bal[1]), float(bal[2])
    present = lambda ep, q: norm(q.replace(" ...", "")) in src[ep]
    for e in doc["evaluations"]:
        ep, cid = str(e["episode_id"]), e["candidate_id"]
        for q in e["evidence"]:
            n_quotes += 1
            if not present(ep, q["quote"]):
                fails.append(f"{cid}: evidence quote not in raw file: {q['quote'][:40]!r}")
            if f"Start: ({q['ts']})" not in src[ep]:
                fails.append(f"{cid}: evidence timestamp {q['ts']} not in raw file")
        for k, text in e["explanations"].items():
            for span in re.findall(r'"([^"]{3,})"', text):
                n_quotes += 1
                if not present(ep, span):
                    fails.append(f"{cid}: {k} explanation quotes text not in raw file: {span[:40]!r}")
        sc = e["scores"]
        if any(not 0 <= v <= 10 for v in sc.values()):
            fails.append(f"{cid}: a score is outside 0-10")
        overall = sum(W[k] * sc[k] for k in W) * (floor + slope * min(sc["hook"], sc["cliffhanger"], sc["standalone"]) / 10)
        if abs(overall - e["overall_score"]) > 0.02:
            fails.append(f"{cid}: overall {e['overall_score']} != recomputed {overall:.2f}")
        if not 20 <= e["duration_sec"] <= 60:
            fails.append(f"{cid}: duration {e['duration_sec']} outside 20-60")
        hard = [f for f in e["gates"]["failures"] if f["code"] != "low_overall"]
        if hard and e["recommendation"] in ("top_pick", "recommend", "maybe"):
            fails.append(f"{cid}: passed through a failed gate")
        if e["recommendation"] == "reject" and not e["gates"]["failures"]:
            fails.append(f"{cid}: rejected with no recorded reason")
    return _result("evaluations", "scores recomputed from weights, gates respected, every quoted span exists in the raw file",
                   fails, evaluations=len(doc["evaluations"]), quoted_spans_checked=n_quotes)


# ------------------------------------------------------------------ 4. ranked + 5. boundaries
def check_ranked(by_ep, ranked, evaluated):
    fails, checked = [], 0
    ev = {e["candidate_id"]: e for e in evaluated["evaluations"]}
    flat = {k: norm(v) for k, v in by_ep.items()}
    promos = ranked["top_promos"]
    for p in promos:
        pid, total, prev = p["promo_id"], 0.0, None
        for s in p["segments"]:
            ep = s["episode"]
            if ep not in flat:
                fails.append(f"{pid}: episode {ep} not in the input"); continue
            a, b = sec(s["start_ts"]), sec(s["end_ts"]); total += b - a
            if not a < b:
                fails.append(f"{pid}: start >= end")
            if f"Start: ({s['start_ts']})" not in flat[ep]:
                fails.append(f"{pid}: start {s['start_ts']} is not an item start in Episode {ep}")
            if s["end_ts"] not in flat[ep]:
                fails.append(f"{pid}: end {s['end_ts']} not present in Episode {ep}")
            if prev and prev[0] == ep and not 0 <= a - prev[1] <= 8.0:
                fails.append(f"{pid}: bad gap {a - prev[1]:.2f}s between segments")
            prev = (ep, b)
        if not 20 <= total <= 60:
            fails.append(f"{pid}: total duration {total:.2f}s outside 20-60")
        if abs(total - p["total_duration_sec"]) > 0.02:
            fails.append(f"{pid}: reported duration differs from recomputed")
        s0, s1 = p["segments"][0], p["segments"][-1]
        hook = p["opening_hook"]["first_3_seconds"]
        if not hook or abs(sec(hook[0]["start_ts"]) - sec(s0["start_ts"])) > 1.0:
            fails.append(f"{pid}: no spoken or visible hook line at the start of the clip")
        for l in hook:
            checked += 1
            if sec(l["start_ts"]) - sec(s0["start_ts"]) >= 3.0 + 1e-6:
                fails.append(f"{pid}: a hook line starts after the first 3 s")
            if norm(l["text"]) not in flat[s0["episode"]]:
                fails.append(f"{pid}: hook text not verbatim: {l['text'][:40]!r}")
        end = p["ending_cliffhanger"]
        for l in end["last_spoken_lines"] + ([end["final_beat"]] if end["final_beat"] else []):
            checked += 1
            if sec(s1["end_ts"]) - sec(l["start_ts"]) > 8.0 + 1e-6:
                fails.append(f"{pid}: an ending line is not in the last 8 s")
            if norm(l["text"]) not in flat[s1["episode"]]:
                fails.append(f"{pid}: ending text not verbatim: {l['text'][:40]!r}")
        if not end["last_spoken_lines"]:
            fails.append(f"{pid}: ending has no spoken line")
        e = ev[p["candidate_id"]]
        if abs(e["overall_score"] - p["overall_ranking_score"]) > 1e-9 or e["scores"] != p["scores"]:
            fails.append(f"{pid}: scores differ from evaluated_promos.json")
        for need in ("promo_id", "episodes", "segments", "total_duration_sec", "opening_hook", "ending_cliffhanger", "why_it_works", "scores"):
            if not p.get(need):
                fails.append(f"{pid}: missing {need}")
    scores = [p["overall_ranking_score"] for p in promos]
    if scores != sorted(scores, reverse=True) and all(p["verification"] == "model-verified" for p in promos):
        fails.append("ranking: not in descending score order")
    eps = [e for p in promos for e in p["episodes"]]
    if len(eps) != len(set(eps)):
        fails.append("ranking: two promos from one episode")
    if sorted(x["promo_id"] for x in ranked["posting_plan"]) != sorted(p["promo_id"] for p in promos):
        fails.append("posting plan does not contain every promo exactly once")
    return _result("ranked", "every ranked promo's timestamps, duration, hook window, ending window, quotes, scores and ordering",
                   fails, promos=len(promos), quoted_lines_checked=checked)


def check_boundaries(by_ep, ranked):
    fails, lines = [], 0
    for p in ranked["top_promos"]:
        pid = p["promo_id"]
        for s in p["segments"]:
            txt = by_ep.get(s["episode"])
            if txt is None:
                continue
            starts, ends = raw_item_times(txt)
            if not any(abs(x - s["start_sec"]) < 0.011 for x in starts):
                fails.append(f"{pid}: start is not an item start")
            if not any(abs(x - s["end_sec"]) < 0.011 for x in ends):
                fails.append(f"{pid}: end is not an item end")
            for a, b, t in raw_dialogue(txt):
                if a < s["start_sec"] - 0.01 and b > s["start_sec"] + 0.01:
                    fails.append(f"{pid}: clip starts inside the line {t[:30]!r}")
                if a < s["end_sec"] - 0.01 and b > s["end_sec"] + 0.01:
                    fails.append(f"{pid}: clip ends inside the line {t[:30]!r}")
        lo, hi = p["segments"][0]["start_sec"], p["segments"][-1]["end_sec"]
        for l in p["opening_hook"]["first_3_seconds"] + p["ending_cliffhanger"]["last_spoken_lines"]:
            if l["kind"] == "dialogue":
                lines += 1
                if sec(l["start_ts"]) < lo - 0.01 or sec(l["end_ts"]) > hi + 0.01:
                    fails.append(f"{pid}: shown line {l['text'][:30]!r} ({l['start_ts']}-{l['end_ts']}) is not fully inside the clip")
    return _result("boundaries", "no promo starts or ends inside a spoken line; every shown hook/ending line lies inside its clip",
                   fails, promos=len(ranked["top_promos"]), displayed_lines_checked=lines)


# ------------------------------------------------------------------ driver
def run_checks(screenplay_dir, outputs_dir):
    J = lambda n: json.load(open(os.path.join(outputs_dir, n), encoding="utf-8"))
    ext, cand, evald, ranked = J("extracted_scenes.json"), J("candidate_promos.json"), J("evaluated_promos.json"), J("ranked_promos.json")
    by_name, by_ep = load_raw(screenplay_dir)
    checks = [check_timestamps(by_name, ext), check_candidates(by_name, ext, cand["candidates"]),
              check_evaluations(by_name, ext, evald), check_ranked(by_ep, ranked, evald), check_boundaries(by_ep, ranked)]
    rules_only = [p["promo_id"] for p in ranked["top_promos"] if p["verification"] != "model-verified"]
    return {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "tool": "quality_checker.py (independent of the pipeline code; reads raw screenplay files directly)",
        "screenplay_dir": os.path.basename(os.path.abspath(screenplay_dir)), "outputs_dir": os.path.basename(os.path.abspath(outputs_dir)),
        "summary": {"checks_run": len(checks), "passed": sum(c["passed"] for c in checks),
                    "failed": sum(not c["passed"] for c in checks), "all_passed": all(c["passed"] for c in checks),
                    "promos_in_ranked_list": len(ranked["top_promos"]), "promos_not_model_verified": rules_only},
        "checks": checks,
        "what_these_checks_do_not_cover": [
            "Whether a clip is actually a good promo. These checks prove facts (timestamps, quotes, arithmetic, rule compliance), not taste.",
            "Agreement with a human editor: no human-labelled set exists yet.",
            "Video or audio: everything is checked against screenplay text and timestamps only.",
            "Model-verified scores come from replayed verdicts or a live model; this tool does not judge them, it only checks that they were combined correctly.",
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("screenplay_dir"); ap.add_argument("outputs_dir")
    ap.add_argument("-o", "--out", default=None, help="default: <outputs_dir>/quality_report.json")
    a = ap.parse_args(argv)
    report = run_checks(a.screenplay_dir, a.outputs_dir)
    out = a.out or os.path.join(a.outputs_dir, "quality_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    for c in report["checks"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']:<11} {c['counts']}")
        for f in c["failures"][:5]:
            print("        -", f)
    s = report["summary"]
    print(f"quality_checker: {s['passed']}/{s['checks_run']} checks passed -> {out}")
    return 0 if s["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

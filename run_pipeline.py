"""
run_pipeline.py - one command from raw screenplay folder to ranked promos.

  python run_pipeline.py <screenplay_dir> -o <out_dir> [--llm none|anthropic|openai-compat|replay:FILE]
                         [--top 10] [--allow-unverified]

Stages (each is its own module and can be run alone):
  1 screenplay_parser   -> extracted_scenes.json
  2 candidate_generator  -> candidate_promos.json     (rules + dialogue dynamics + optional LLM)
  3 promo_evaluator     -> evaluated_promos.json/.csv (6 scores, weighted overall, gates, explanations)
  4 promo_ranker        -> ranked_promos.json/.md     (top N, posting plan)
  5 quality_checker     -> quality_report.json        (independent audit of 1-4 against the raw files)
The LLM is swappable: set LLM_MODEL / ANTHROPIC_API_KEY, or LLM_API_KEY + LLM_BASE_URL + LLM_MODEL for any
OpenAI-compatible endpoint (e.g. Gemini). With --llm none the run is rules-only, and because only
model-verified clips may be top-ranked, add --allow-unverified to still get a list (labelled UNVERIFIED).
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

def run(script, *args, ok=(0,)):
    cmd = [sys.executable, "-I", os.path.join(HERE, script), *map(str, args)]
    print("\n$", " ".join(os.path.basename(c) if i < 3 else c for i, c in enumerate(cmd)))
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode not in ok:
        sys.exit(f"{script} failed with exit code {r.returncode}")

def main():
    ap = argparse.ArgumentParser(description="Promo Discovery Engine - full pipeline")
    ap.add_argument("screenplay_dir"); ap.add_argument("-o", "--out-dir", default="run_output")
    ap.add_argument("--llm", default="none"); ap.add_argument("--llm-cache", default=None)
    ap.add_argument("--top", type=int, default=10); ap.add_argument("--allow-unverified", action="store_true")
    a = ap.parse_args()
    out = os.path.abspath(a.out_dir); os.makedirs(out, exist_ok=True)
    p = lambda n: os.path.join(out, n)
    llm = ["--llm", a.llm] + (["--llm-cache", os.path.abspath(a.llm_cache)] if a.llm_cache else [])
    if a.llm.startswith("replay:"):
        llm = ["--llm", "replay:" + os.path.abspath(a.llm[7:])]
    run("screenplay_parser.py", os.path.abspath(a.screenplay_dir), "-o", p("extracted_scenes.json"), ok=(0, 1))   # 1 = some files unreadable
    import json
    doc = json.load(open(p("extracted_scenes.json"), encoding="utf-8"))
    if not any(e.get("parse_status") != "failed" and e.get("n_items", 0) > 0 for e in doc.get("episodes", [])):
        sys.exit("No usable screenplay episodes found in " + a.screenplay_dir)
    run("candidate_generator.py", p("extracted_scenes.json"), "-o", p("candidate_promos.json"), "--llm-top-k", 200, *llm)
    run("promo_evaluator.py", p("extracted_scenes.json"), p("candidate_promos.json"), "-o", p("evaluated_promos.json"),
        "--csv", p("evaluated_promos.csv"), *llm)
    run("promo_ranker.py", p("extracted_scenes.json"), p("candidate_promos.json"), p("evaluated_promos.json"),
        "-o", p("ranked_promos.json"), "--md", p("ranked_promos.md"), "--top", a.top, *(["--allow-unverified"] if a.allow_unverified else []))
    run("quality_checker.py", os.path.abspath(a.screenplay_dir), out, ok=(0, 1))     # 1 = a check failed; report is still written
    print(f"\nDone. Results in {out}")

if __name__ == "__main__":
    main()

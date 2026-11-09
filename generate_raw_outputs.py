"""
generate_raw_outputs.py - executes the final application pipeline on a screenplay folder and saves everything it produces.

Nothing is edited after generation: stage files are written directly by the pipeline stages; the CSV/JSON/tables are
produced by the app's own export functions (promo_app.results); logs are captured from the run.

  python generate_raw_outputs.py <screenplay_dir> <out_dir>
"""
import datetime, hashlib, json, os, platform, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from promo_app import runner, results
import contextlib, io
import quality_checker

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def execute(src, out, mode, label):
    os.makedirs(out, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc)
    events = []
    def on_stage(key, msg, frac):
        events.append(f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}  [{frac:5.0%}] {key}: {msg}")
    llm = runner.LLMChoice(mode=mode)
    t0 = time.time()
    run = runner.run_pipeline(src, llm, on_stage, top_n=10, workdir=out)          # stage files are written into `out` by the stages
    wall = round(time.time() - t0, 2)
    W = lambda name, data: open(os.path.join(out, name), "wb").write(data if isinstance(data, bytes) else data.encode("utf-8"))
    # outputs of the app's own export / table functions (same code the Download buttons call)
    W("ranked_promos_export.csv", results.to_csv(run))
    W("detailed_results_export.json", results.to_detail_json(run))
    rej = results.rejection_table(run)
    W("rejected_candidates.csv", rej.to_csv(index=False))
    W("rejected_candidates.json", json.dumps(rej.to_dict(orient="records"), indent=2, ensure_ascii=False))
    val = [{"promo_id": p["promo_id"], "candidate_id": p["candidate_id"], "episode": p["episodes"], "verification": p["verification"],
            "checks": results.validate_promo(p)} for p in run.final["top_promos"]]
    W("app_validator_results.json", json.dumps({"validator": "promo_app.results.validate_promo", "promos_checked": len(val),
        "errors": sum(1 for v in val for c in v["checks"] if c["level"] == "error"),
        "warnings": sum(1 for v in val for c in v["checks"] if c["level"] == "warn"), "per_promo": val}, indent=2, ensure_ascii=False))
    buf = io.StringIO()                                                           # independent audit, same code as `python quality_checker.py`
    with contextlib.redirect_stdout(buf):
        quality_checker.main([src, out])
    W("quality_checker_log.txt", buf.getvalue())
    W("execution_log.txt", "\n".join(events) + "\n\n--- stage output (captured stdout/stderr) ---\n" + run.log)
    return run, started, wall

def main():
    src, root = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    files = sorted(f for f in os.listdir(src) if f.endswith(".txt"))
    manifest = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "command": "python generate_raw_outputs.py data/screenplay_input raw_outputs", "python": platform.python_version(),
                "input_files": len(files), "input_sha256": {f: sha(os.path.join(src, f)) for f in files},
                "code_sha256": {f: sha(os.path.join(HERE, f)) for f in sorted(os.listdir(HERE)) if f.endswith(".py") and not f.startswith("test")},
                "replay_verdicts_sha256": sha(runner.DEFAULT_REPLAY), "runs": {}}
    for mode, label in (("replay", "run_A_saved_model_verdicts"), ("none", "run_B_rules_only")):
        run, started, wall = execute(src, os.path.join(root, label), mode, label)
        manifest["runs"][label] = {"llm_mode": mode, "started_utc": started.isoformat(timespec="seconds"), "wall_seconds": wall,
                                   "stage_seconds": run.seconds, "pipeline_summary": run.final["pipeline"],
                                   "promos": [(p["promo_id"], p["candidate_id"]) for p in run.final["top_promos"]]}
    json.dump(manifest, open(os.path.join(root, "manifest.json"), "w"), indent=2)

if __name__ == "__main__":
    main()

# Promo Discovery Engine

Reads microdrama screenplays and finds the best 20–60 second promo clips. For every clip it reports the episode,
exact start/end timestamps, the opening hook (first 3 s), the ending cliffhanger, whether a new viewer can follow it,
six scores, a rank and a one-line reason. It also gives a posting order and lists every rejected candidate with the rule
it failed. Timestamps and dialogue are never invented: they are copied from the input and re-checked against it by
`quality_checker.py` and the tests.

The repository ships with three synthetic demo episodes in `data/demo_input/`, so everything below runs as-is.

## Project layout

```
.
├── app.py                  Streamlit web app (upload, process, ranked promos, inspector, rejected list, CSV/JSON download)
├── screenplay_parser.py    stage 1: raw .txt -> scenes and items with original timestamps
├── candidate_generator.py  stage 2: finds peak moments, builds 20-60 s windows, optional LLM judgement
├── promo_evaluator.py      stage 3: six scores, weighted overall, hard gates, explanations
├── promo_ranker.py         stage 4: eligibility, top N, posting plan
├── quality_checker.py      stage 5: independent audit of the outputs against the raw files -> quality_report.json
│
├── data/
│   └── demo_input/         3 synthetic episodes (invented characters and dialogue)
│
├── requirements.txt        runtime dependencies          requirements-dev.txt   adds pytest
└── README.md

Supporting files:  llm_client.py (swappable model: anthropic | openai-compat | replay | none), semantic_judge.py (prompt and
                   verdict handling), run_pipeline.py (command line for stages 1-5), promo_app/ (UI helpers for app.py),
                   tests/ (unit and end-to-end tests), generate_raw_outputs.py (helper that runs the pipeline in several judging
                   modes and saves everything with a manifest; it expects a saved-verdict file and is not needed for normal use)
```

Run results are written to a folder you choose (the examples use `outputs/`). Run output, logs and any private input are
excluded by `.gitignore` and are not part of the repository.

## Installation

Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt                          # streamlit, pandas  (tests: pip install -r requirements-dev.txt)
```

The pipeline and command line use only the Python standard library; `requirements.txt` is needed for the web app.

## Execution commands

All examples use the bundled synthetic demo. Replace `data/demo_input` with the folder holding your own episode files.

| Goal | Command |
|---|---|
| Web app | `streamlit run app.py` (opens http://localhost:8501; the folder field defaults to `data/demo_input/`) |
| Whole pipeline, rules only | `python run_pipeline.py data/demo_input -o outputs --allow-unverified` |
| Quality audit of a run | `python quality_checker.py data/demo_input outputs`  (writes `outputs/quality_report.json`, exit code 1 if any check fails) |
| One stage | `python screenplay_parser.py <dir> -o x.json`, then `candidate_generator.py`, `promo_evaluator.py`, `promo_ranker.py` (each has `--help`) |
| Tests | `python -m pytest tests -q`  (tests that need a private screenplay pack skip themselves when it is absent) |
| With a live model (bash) | `ANTHROPIC_API_KEY=... python run_pipeline.py data/demo_input -o outputs --llm anthropic`  or  `LLM_API_KEY=... LLM_BASE_URL=... LLM_MODEL=... python run_pipeline.py data/demo_input -o outputs --llm openai-compat` |
| With a live model (PowerShell) | `$env:ANTHROPIC_API_KEY = "..."; python run_pipeline.py data/demo_input -o outputs --llm anthropic`  or set `$env:LLM_API_KEY`, `$env:LLM_BASE_URL`, `$env:LLM_MODEL` and use `--llm openai-compat` |
| With saved verdicts | `python run_pipeline.py <dir> -o outputs --llm replay:<verdicts.jsonl> --allow-unverified` (see below) |

`run_pipeline.py` writes into the folder given with `-o`, replacing files of the same name, so point it at a new folder to keep earlier results.
`--llm-cache <file>` caches live model answers on disk, so repeated runs do not pay twice.

**Model modes (`--llm`).** `none` (default) is rules only. `anthropic` needs `ANTHROPIC_API_KEY`; `LLM_MODEL` optionally overrides the model name.
`openai-compat` talks to any OpenAI-compatible endpoint and needs `LLM_API_KEY` and `LLM_MODEL` (`LLM_BASE_URL` for non-OpenAI providers).
`replay:<file>` serves saved answers from a JSONL file with one record per line, `{"key": <candidate_id>, "result": {...}, "model": "..."}`, with no network and no key.
Keys are read from the process environment and are never written to disk. A failed or malformed model answer makes that
candidate fall back to rules only (visible as `semantic.status` and `confidence` in the outputs); it does not stop the run.

## Architecture

```
 raw .txt files (upload / folder / zip)
   ▼
 screenplay_parser ──► extracted_scenes.json   episodes → scenes → items(kind, speaker, delivery, start/end, text)
   ▼
 candidate_generator ─► candidate_promos.json  20–60 s windows that start and end on item boundaries, never inside a spoken line
   ▼
 promo_evaluator ─────► evaluated_promos.json  hook, cliffhanger, standalone, intensity, coherence, character clarity;
   ▼                                           weighted overall x balance factor; hard gates; rules blended with LLM if present
 promo_ranker ────────► ranked_promos.json     eligibility bar, one promo per episode, no adjacent-episode duplicates, posting plan
   ▼
 quality_checker ─────► quality_report.json    re-checks 1-4 against the raw files; imports none of the pipeline code
```

Weights: hook 0.25, cliffhanger 0.25, standalone 0.20, intensity 0.12, coherence 0.10, character clarity 0.08; overall =
weighted sum × (0.80 + 0.20 × min(hook, cliffhanger, standalone) / 10). Eligible for the final list: hook ≥ 6, cliffhanger ≥ 6,
standalone ≥ 5.5, evaluator verdict top_pick or recommend, duration 20–60 s. Model-verified clips outrank rules-only ones;
rules-only results are labelled `UNVERIFIED`. Rule and model scores are blended 0.4 / 0.6 where a model judged the clip.
Code does everything that must be exact (parsing, timestamps, windows, scoring arithmetic, ranking, checks); a model is used only to
judge meaning, and never writes timestamps or dialogue (it cites item ids and short quotes, which the code verifies against the input).

## Input format

One `.txt` file per episode (several files or a `.zip`). File names do not matter; the episode number comes from the
`EPISODE NUMBER:` line. UTF-8 with or without BOM, LF or CRLF. A complete synthetic example is in `data/demo_input/`.

```
EPISODE NUMBER: Episode 101
EPISODE LOGLINE: A test logline.
CONTINUITY NOTES: Synthetic test episode.
--- SCRIPT START ---
SCENE 01
Scene heading: INT. HARBOR OFFICE – NIGHT
Scene start: 00:00.00
Scene end: 00:58.00
Actions and dialogue:
**CAMERA:**
Start: (00:00.00) | End: (00:00.50)
Wide shot of an empty office...
**CAPTAIN MARA (Furious):**
Start: (00:00.50) | End: (00:02.50)
"You sold the harbor to the man who killed my brother!"
```

Block types: `**CAMERA:**`, `**ACTION:**`, `**SOUND FX:**`, and `**CHARACTER (delivery):**` for dialogue; times are `mm:ss.cc`.
Unreadable files are skipped and listed in the app; a run fails with a clear message only if no usable episode remains.

## Output format

* **`ranked_promos.json`**: `top_promos[]`, each with `promo_id`, `rank`, `episodes`, `segments[]` (`episode`, `start_ts`, `end_ts`,
  `start_sec`, `end_sec`, `duration_sec`), `total_duration_sec`, `opening_hook.first_3_seconds[]`, `ending_cliffhanger.last_spoken_lines[]`
  and `why_unresolved`, `why_it_works.text`, `scores`, `overall_ranking_score`, `verification` (`model-verified` or `UNVERIFIED (rules only)`),
  `caveats` (what a new viewer would miss). Also `posting_plan[]`, `same_episode_skips`, `near_misses`, `limitations`, `pipeline` (run summary).
* **`candidate_promos.json`**: every candidate window with its dialogue, characters and the reason it was proposed.
* **`evaluated_promos.json` / `.csv`**: every candidate's six scores, overall, gates that failed, recommendation and explanations quoting the script.
* **`quality_report.json`**: five checks (`timestamps`, `candidates`, `evaluations`, `ranked`, `boundaries`), each with pass/fail, counts and failures,
  plus `what_these_checks_do_not_cover`.
* **App downloads**: ranked promos CSV (one row per segment: rank, promo_id, episode, start/end timestamp, durations, hook, cliffhanger, why,
  six scores, overall, post order, verification, validation warnings) and detailed results JSON (final promos with validation, posting plan,
  every evaluation, rejected candidates with reasons, weights, gates, limitations).

## What you can verify with the demo

`python run_pipeline.py data/demo_input -o outputs --allow-unverified` processes the 3 synthetic episodes, finds 4 candidates and
selects 2 promos, both labelled `UNVERIFIED (rules only)` because no model was used. `quality_checker.py` then reports 5 of 5 checks passing
(timestamps, candidates, evaluations, ranked list, clip boundaries), and no promo starts or ends inside a spoken line.
These checks prove facts (timestamps, quotes, arithmetic, rule compliance), not that a clip is a *good* promo.

## Known limitations

* **Rule scores for cliffhanger and standalone context are weak heuristics.** Rules-only output is labelled `UNVERIFIED` and needs review;
  a model verdict is expected to confirm hook, cliffhanger and standalone clarity before a clip is published.
* **Model adapters.** The `anthropic` and `openai-compat` adapters are implemented but have not been exercised against a real endpoint in this repository.
  A saved-verdict (`replay`) file only matches the exact input and clip windows it was written for; any other input falls back to rules.
* **Borderline candidates.** Without a model, borderline candidates stay "needs LLM review" instead of being rejected or recommended.
* **Cost.** A live run sends one request per judged candidate. `--llm-top-k` limits the generator's calls, but the evaluator also judges any candidate
  that still has no verdict, so a full run can approach one call per candidate; use `--llm-cache` for repeat runs.
* Single contiguous clip per promo; no joins across moments or episodes. Text only: no video or audio.
* One promo per episode and neighbouring episodes avoided, so a second strong moment in an episode is listed as "not selected".
* Scores and weights were set by reasoning, not tuned on human labels; there is no editor-labelled set and no baseline comparison.
* The parser expects the file layout shown under Input format.
* **Confidentiality.** Private screenplay input, saved verdicts, run outputs and logs are kept out of this repository (see `.gitignore`);
  only the synthetic demo is included.

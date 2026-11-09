"""
runner.py - runs the REAL Promo Discovery Engine pipeline for the web app.

No logic is duplicated here: each stage calls the main() of the existing module
(screenplay_parser -> candidate_generator -> promo_evaluator -> promo_ranker) in-process and reports progress
through a callback, so the UI can show a progress bar. Output of each stage is captured into a log.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

DEFAULT_REPLAY = os.path.join(ENGINE_DIR, "data", "llm_judgements_all.jsonl")
STAGES = [("parse", "Reading and parsing screenplay files"),
          ("detect", "Finding candidate moments"),
          ("evaluate", "Scoring hook, cliffhanger and context"),
          ("rank", "Ranking the final promos")]

MAX_UPLOAD_BYTES = 60 * 1024 * 1024


class PipelineError(RuntimeError):
    """A stage failed in a way the user can act on (message is shown in the UI)."""


@dataclass
class LLMChoice:
    mode: str = "none"                     # none | replay | anthropic | openai-compat
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    replay_file: str = DEFAULT_REPLAY
    top_k: int = 40

    def spec(self) -> str:
        if self.mode == "replay":
            return "replay:" + self.replay_file
        return self.mode

    def apply_env(self) -> None:
        """Hand credentials to llm_client through the environment of this process only (never written to disk)."""
        if self.mode == "anthropic":
            if not self.api_key:
                raise PipelineError("Enter an Anthropic API key, or choose another model option.")
            os.environ["ANTHROPIC_API_KEY"] = self.api_key
        elif self.mode == "openai-compat":
            if not (self.api_key and self.base_url and self.model):
                raise PipelineError("OpenAI-compatible mode needs an API key, base URL and model name.")
            os.environ.update(LLM_API_KEY=self.api_key, LLM_BASE_URL=self.base_url)
        if self.model:
            os.environ["LLM_MODEL"] = self.model


@dataclass
class RunResult:
    workdir: str
    paths: Dict[str, str]
    extracted: Dict[str, Any]
    candidates: Dict[str, Any]
    evaluated: Dict[str, Any]
    final: Dict[str, Any]
    log: str
    seconds: Dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------------------------------ uploads
def ingest_uploads(files: Sequence[Any], dest: str) -> List[str]:
    """
    Save uploaded .txt files (or the .txt files inside uploaded .zip archives) into `dest`.
    Only base names are used, archive paths are never trusted (no path traversal), and size is capped.
    `files` are Streamlit UploadedFile-like objects (.name, .getvalue()).
    """
    os.makedirs(dest, exist_ok=True)
    saved: List[str] = []
    total = 0

    def put(name: str, data: bytes) -> None:
        nonlocal total
        total += len(data)
        if total > MAX_UPLOAD_BYTES:
            raise PipelineError("Upload is larger than 60 MB; screenplay text files should be far smaller.")
        base = re.sub(r"[^\w.\- ]", "_", os.path.basename(name))
        if not base.lower().endswith(".txt"):
            return
        stem, ext = os.path.splitext(base)
        n = 1
        while base in saved:                       # same name twice (e.g. two folders in a zip): keep both
            n += 1
            base = f"{stem}__{n}{ext}"
        with open(os.path.join(dest, base), "wb") as fh:
            fh.write(data)
        saved.append(base)

    for f in files:
        name, data = f.name, f.getvalue()
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for info in zf.infolist():
                        if info.is_dir() or info.filename.startswith("__MACOSX") or "/." in info.filename:
                            continue
                        if info.file_size > 5 * 1024 * 1024:
                            continue
                        put(info.filename, zf.read(info))
            except zipfile.BadZipFile as exc:
                raise PipelineError(f"{name} is not a valid zip file.") from exc
        else:
            put(name, data)
    if not saved:
        raise PipelineError("No .txt screenplay files found in the upload.")
    return saved


def copy_folder(src: str, dest: str) -> List[str]:
    os.makedirs(dest, exist_ok=True)
    out = []
    for root, _, names in os.walk(src):
        for n in names:
            if n.lower().endswith(".txt"):
                shutil.copy(os.path.join(root, n), os.path.join(dest, n))
                out.append(n)
    if not out:
        raise PipelineError(f"No .txt files found in {src}.")
    return out


# ------------------------------------------------------------------------------------------ stages
def _call(module_name: str, argv: List[str], log: io.StringIO) -> int:
    import importlib
    mod = importlib.import_module(module_name)
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        return mod.main(argv)


def run_pipeline(screenplay_dir: str, llm: LLMChoice, on_stage: Optional[Callable[[str, str, float], None]] = None,
                 top_n: int = 10, workdir: Optional[str] = None, keep_files: bool = False) -> RunResult:
    """Run all four stages on a folder of screenplay .txt files. on_stage(key, message, fraction_done).
    Intermediate files live in a temp dir that is deleted afterwards unless keep_files=True (or workdir is given)."""
    import time
    on_stage = on_stage or (lambda *_: None)
    own_dir = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="promo_run_")
    try:
        P = {n: os.path.join(workdir, n) for n in ("extracted_scenes.json", "candidate_promos.json", "evaluated_promos.json",
                                                   "evaluated_promos.csv", "ranked_promos.json", "ranked_promos.md")}
        log = io.StringIO()
        llm.apply_env()
        if llm.mode == "replay" and not os.path.exists(llm.replay_file):
            raise PipelineError(f"Saved verdict file not found: {llm.replay_file}")
        llm_args = ["--llm", llm.spec()] if llm.mode != "none" else []
        secs: Dict[str, float] = {}

        def stage(i: int, key: str, argv: List[str], module: str) -> None:
            on_stage(key, STAGES[i][1], i / len(STAGES))
            t0 = time.time()
            try:
                code = _call(module, argv, log)
            except PipelineError:
                raise
            except Exception as exc:                                    # a stage crashed: tell the user which one
                raise PipelineError(f"Stage '{STAGES[i][1]}' failed: {type(exc).__name__}: {exc}") from exc
            # the parser exits 1 when some FILES failed; that is fine as long as other episodes parsed (checked below)
            if code not in (0, None) and not (key == "parse" and code == 1):
                raise PipelineError(f"Stage '{STAGES[i][1]}' reported errors (exit code {code}). See the run log.")
            secs[key] = round(time.time() - t0, 2)

        stage(0, "parse", [screenplay_dir, "-o", P["extracted_scenes.json"]], "screenplay_parser")
        with open(P["extracted_scenes.json"], encoding="utf-8") as fh:
            ext = json.load(fh)
        usable = [e for e in ext.get("episodes", []) if e.get("parse_status") != "failed" and e.get("n_items", 0) > 0]
        if not usable or ext["summary"].get("items", 0) == 0:
            raise PipelineError("No usable episodes were found. Files should be named like 'episode-001-Episode 123.txt' "
                                "and contain SCENE blocks with 'Start: (mm:ss.cc) | End: (mm:ss.cc)' lines.")
        stage(1, "detect", [P["extracted_scenes.json"], "-o", P["candidate_promos.json"], "--llm-top-k", str(llm.top_k)] + llm_args,
              "candidate_generator")
        stage(2, "evaluate", [P["extracted_scenes.json"], P["candidate_promos.json"], "-o", P["evaluated_promos.json"],
                              "--csv", P["evaluated_promos.csv"]] + llm_args, "promo_evaluator")
        stage(3, "rank", [P["extracted_scenes.json"], P["candidate_promos.json"], P["evaluated_promos.json"], "-o", P["ranked_promos.json"],
                          "--md", P["ranked_promos.md"], "--top", str(top_n), "--allow-unverified"], "promo_ranker")
        on_stage("done", "Finished", 1.0)
        def J(n):
            with open(P[n], encoding="utf-8") as fh:
                return json.load(fh)
        return RunResult(workdir, P, ext, J("candidate_promos.json"), J("evaluated_promos.json"), J("ranked_promos.json"),
                         log.getvalue(), secs)
    finally:
        if own_dir and not keep_files:             # the JSON is already in memory; do not leak temp dirs
            shutil.rmtree(workdir, ignore_errors=True)

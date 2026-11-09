"""
testdata.py - fixtures for the unit tests, built by running the REAL pipeline (never read from stale files).

Uses the real 50-episode pack (with saved verdicts) when it is available (data/screenplay_input/ or $PROMO_SAMPLE_DIR);
otherwise the synthetic data/demo_input/ that ships with the project, so the tests also run on a fresh checkout.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from promo_app import runner  # noqa: E402

PACK = os.environ.get("PROMO_SAMPLE_DIR", os.path.join(ROOT, "data", "screenplay_input"))
HAVE_PACK = os.path.isdir(PACK) and any(f.endswith(".txt") for f in os.listdir(PACK))
DEMO = os.path.join(ROOT, "data", "demo_input")
_cache = {}


def run(kind="judged"):
    """kind='judged': pack + saved verdicts if possible.  kind='rules': rules only."""
    key = (kind, HAVE_PACK)
    if key not in _cache:
        src = PACK if HAVE_PACK else DEMO
        mode = "replay" if (kind == "judged" and HAVE_PACK and os.path.exists(runner.DEFAULT_REPLAY)) else "none"
        _cache[key] = runner.run_pipeline(src, runner.LLMChoice(mode=mode))
    return _cache[key]


def fixtures(kind="judged"):
    r = run(kind)
    return r.extracted, r.candidates, r.evaluated


STRONG_VERDICT = {"is_genuine_promo_moment": True, "keyword_only_false_positive": False, "hook_strength": 9.0,
                  "cliffhanger_strength": 8.0, "standalone_clarity": 6.0, "moment_types": ["confrontation", "unresolved"],
                  "needs_context": ["who the Hooded Man is"], "main_characters": [], "why_it_attracts": "test verdict",
                  "evidence": [], "status": "ok"}

"""
test_e2e.py - end-to-end tests of the whole Promo Discovery Engine, through the same entry points the app uses.

Ground truth comes from an INDEPENDENT reader of the raw .txt files (written here, sharing no code with
screenplay_parser.py), so a parser bug cannot hide itself.

Areas: 1 upload, 2 multi-episode parsing, 3 candidates, 4 hook, 5 cliffhanger, 6 standalone, 7 ranking,
8 timestamp accuracy, 9 CSV/JSON exports, 10 a new screenplay input (synthetic show, renumbered subset, zip).

Run:  python -m pytest test_e2e.py -q        (the real-pack tests are skipped if data/screenplay_input/ is absent;
                                              the synthetic-input tests always run)
"""
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from promo_app import results, runner                                   # noqa: E402

PACK = os.environ.get("PROMO_SAMPLE_DIR", os.path.join(ROOT, "data", "screenplay_input"))
HAVE_PACK = os.path.isdir(PACK) and any(f.endswith(".txt") for f in os.listdir(PACK))
need_pack = unittest.skipUnless(HAVE_PACK, "data/screenplay_input/ not present")


# ------------------------------------------------------------------ independent ground-truth reader
def ts_sec(t):
    m = re.match(r"(\d+):(\d{2})\.(\d{2})$", t)
    return int(m[1]) * 60 + int(m[2]) + int(m[3]) / 100


def raw_items(path):
    """Return (episode_number, [items]) read straight from a raw file. item = dict(kind, speaker, start, end, text)."""
    txt = open(path, encoding="utf-8-sig").read().replace("\r\n", "\n")
    num = int(re.search(r"EPISODE NUMBER:\s*Episode\s+(\d+)", txt)[1])
    items = []
    lines = txt.split("\n")
    i = 0
    while i < len(lines):
        m = re.match(r"\*\*(.+?):\*\*\s*$", lines[i])
        if m and i + 1 < len(lines):
            t = re.match(r"Start: \((\d+:\d{2}\.\d{2})\) \| End: \((\d+:\d{2}\.\d{2})\)", lines[i + 1])
            if t:
                head = m[1].strip()
                body, j = [], i + 2
                while j < len(lines) and lines[j].strip() and not lines[j].startswith("**") and not lines[j].startswith("SCENE"):
                    body.append(lines[j].strip()); j += 1
                name = re.sub(r"\s*\(.*\)\s*$", "", head).strip()
                kind = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sound"}.get(name, "dialogue")
                items.append({"kind": kind, "speaker": name if kind == "dialogue" else None,
                              "start": t[1], "end": t[2], "text": " ".join(body).strip().strip('"“”')})
                i = j
                continue
        i += 1
    return num, items


# ------------------------------------------------------------------ synthetic screenplay generator (a different show)
def _blk(kind, start, end, text, speaker=None, delivery=None):
    if kind == "dialogue":
        return f"**{speaker} ({delivery}):**\nStart: ({start}) | End: ({end})\n\"{text}\"\n"
    return f"**{kind.upper()}:**\nStart: ({start}) | End: ({end})\n{text}\n"


def make_episode(num, scenes, logline="A test logline.", crlf=False, bom=False):
    out = [f"EPISODE NUMBER: Episode {num}", f"EPISODE LOGLINE: {logline}", "CONTINUITY NOTES: Synthetic test episode.",
           "", "--- SCRIPT START ---", ""]
    for n, (head, s, e, blocks) in enumerate(scenes, 1):
        out += [f"SCENE {n:02d}", f"Scene heading: {head}", f"Scene start: {s}", f"Scene end: {e}", "", "Actions and dialogue:", ""]
        out += blocks
    txt = "\n".join(out)
    if crlf:
        txt = txt.replace("\n", "\r\n")
    return ("﻿" if bom else "") + txt


def strong_scene(offset=0):
    d = lambda s, e, who, dl, t: _blk("dialogue", f"00:{s:05.2f}", f"00:{e:05.2f}", t, who, dl)
    a = lambda s, e, t: _blk("action", f"00:{s:05.2f}", f"00:{e:05.2f}", t)
    return ("INT. HARBOR OFFICE – NIGHT", "00:00.00", "00:58.00", [
        d(0.5, 2.5, "CAPTAIN MARA", "Furious", "You sold the harbor to the man who killed my brother!"),
        d(3.0, 5.5, "TOBIAS", "Panicked", "I had no choice, Mara. He has the ledger."),
        a(5.5, 8.0, "Mara slams the ledger onto the desk, scattering sealed letters across the floor."),
        d(8.5, 11.0, "CAPTAIN MARA", "Cold", "Then tell me who signed the order, Tobias."),
        d(11.5, 14.0, "TOBIAS", "Trembling", "If I say his name, we both die tonight."),
        a(14.5, 17.0, "A lantern shatters outside; footsteps cross the deck above them."),
        d(17.5, 20.0, "CAPTAIN MARA", "Desperate", "Say it. Say the name!"),
        d(20.5, 23.0, "TOBIAS", "Whispering", "It was never him. It was you, Mara."),
        d(24.0, 27.0, "CAPTAIN MARA", "Shocked", "What did you just say to me?"),
        d(28.0, 31.0, "TOBIAS", "Terrified", "Your own father signed the order. He is still alive."),
        a(31.5, 34.0, "The door bursts open. A hooded figure stands in the lamplight."),
        d(34.5, 37.5, "HOODED MAN", "Menacing", "Hello, my daughter. Did you miss me?"),
        d(38.0, 40.0, "CAPTAIN MARA", "Breathless", "Father...?"),
    ])


def weak_scene():
    d = lambda s, e, who, t: _blk("dialogue", f"00:{s:05.2f}", f"00:{e:05.2f}", t, who, "Calm")
    return ("INT. KITCHEN – DAY", "00:00.00", "00:50.00", [
        d(1.0, 3.0, "ANNA", "Good morning."), d(3.5, 5.5, "BEN", "Good morning. Did you sleep well?"),
        d(6.0, 8.0, "ANNA", "Yes, thank you. Would you like some tea?"), d(8.5, 10.5, "BEN", "Tea would be lovely."),
        d(11.0, 13.0, "ANNA", "I will put the kettle on."), d(13.5, 15.5, "BEN", "The weather is fine today."),
        d(16.0, 18.0, "ANNA", "It is. I think I will walk later."), d(18.5, 20.5, "BEN", "That sounds pleasant."),
        d(21.0, 23.0, "ANNA", "The tea is ready."), d(23.5, 25.5, "BEN", "Thank you, Anna."),
        d(26.0, 28.0, "ANNA", "You are welcome."), d(28.5, 30.5, "BEN", "I will see you at dinner."),
    ])


def write_synthetic(dest, style="plain"):
    os.makedirs(dest, exist_ok=True)
    files = {
        "ep-A.txt": make_episode(101, [strong_scene()], crlf=(style == "crlf"), bom=(style == "crlf")),
        "ep-B.txt": make_episode(102, [weak_scene()]),
        "ep-C.txt": make_episode(103, [strong_scene()]),
    }
    for n, t in files.items():
        with open(os.path.join(dest, n), "w", encoding="utf-8", newline="") as fh:
            fh.write(t)
    return files


class Up:                       # UploadedFile stand-in
    def __init__(self, name, data):
        self.name, self._d = name, data if isinstance(data, bytes) else data.encode("utf-8")

    def getvalue(self):
        return self._d


def run_on(folder, mode="none", top=10):
    return runner.run_pipeline(folder, runner.LLMChoice(mode=mode), top_n=top)


# ------------------------------------------------------------------ shared pipeline runs (computed once)
_CACHE = {}


def full_run(mode):
    if mode not in _CACHE:
        _CACHE[mode] = run_on(PACK, mode)
    return _CACHE[mode]


# ================================================================== 1. uploading
class T01_Upload(unittest.TestCase):
    def test_multi_file_upload_roundtrip(self):
        d = tempfile.mkdtemp()
        saved = runner.ingest_uploads([Up("a.txt", "x"), Up("b.txt", "y"), Up("notes.pdf", "z"), Up("c.TXT", "w")], d)
        self.assertEqual(sorted(saved), ["a.txt", "b.txt", "c.TXT"])
        self.assertEqual(sorted(os.listdir(d)), sorted(saved))

    def test_zip_upload_with_folders_and_junk(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("series/ep1.txt", "a"); z.writestr("__MACOSX/._ep1.txt", "junk"); z.writestr("series/readme.md", "n")
            z.writestr("../../evil.txt", "e")
        d = tempfile.mkdtemp()
        saved = runner.ingest_uploads([Up("pack.zip", buf.getvalue())], d)
        self.assertIn("ep1.txt", saved)
        self.assertNotIn("readme.md", saved)
        self.assertFalse(os.path.exists(os.path.join(d, "..", "evil.txt")))
        self.assertFalse(any(n.startswith("._") for n in os.listdir(d)))

    def test_unsafe_names_are_sanitised(self):
        d = tempfile.mkdtemp()
        saved = runner.ingest_uploads([Up("../../etc/passwd.txt", "x"), Up("we ird;$(rm -rf).txt", "y")], d)
        for n in saved:
            self.assertTrue(os.path.exists(os.path.join(d, n)))
            self.assertNotIn("/", n); self.assertNotIn("$", n); self.assertNotIn(";", n)

    def test_same_name_twice_is_not_silently_lost(self):
        d = tempfile.mkdtemp()
        saved = runner.ingest_uploads([Up("ep.txt", "first"), Up("ep.txt", "second")], d)
        self.assertEqual(len(set(saved)), 2, "two different uploads with the same name must both be kept")

    def test_errors_are_user_readable(self):
        d = tempfile.mkdtemp()
        with self.assertRaises(runner.PipelineError):
            runner.ingest_uploads([Up("a.pdf", "x")], d)
        with self.assertRaises(runner.PipelineError):
            runner.ingest_uploads([Up("bad.zip", b"not a zip")], d)

    def test_size_cap(self):
        old = runner.MAX_UPLOAD_BYTES
        runner.MAX_UPLOAD_BYTES = 100
        try:
            with self.assertRaises(runner.PipelineError):
                runner.ingest_uploads([Up("big.txt", "x" * 500)], tempfile.mkdtemp())
        finally:
            runner.MAX_UPLOAD_BYTES = old

    @need_pack
    def test_upload_real_pack_then_process(self):
        d = tempfile.mkdtemp()
        ups = [Up(n, open(os.path.join(PACK, n), "rb").read()) for n in sorted(os.listdir(PACK)) if n.endswith(".txt")]
        runner.ingest_uploads(ups, d)
        run = run_on(d)
        self.assertEqual(run.final["pipeline"]["episodes"], 50)

    def test_app_ui_on_shipped_demo_input(self):
        """The Streamlit page end to end on the small synthetic data/demo_input/ that ships with the project (no pack needed)."""
        from streamlit.testing.v1 import AppTest
        demo = os.path.join(ROOT, "data", "demo_input")
        at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120).run()
        self.assertFalse(at.exception)
        at.radio(key="judge_mode").set_value("Rules only (no model)").run()
        at.radio(key="src_kind").set_value("Folder on this computer").run()
        at.text_input(key="folder_path").set_value(demo).run()
        at.button[0].click().run()
        self.assertFalse(at.exception)
        run = at.session_state["run"]
        self.assertEqual(run.final["pipeline"]["episodes"], 3)
        self.assertEqual(sorted({p["episodes"][0] for p in run.final["top_promos"]}), [101, 103])

    @need_pack
    def test_app_ui_upload_path_via_apptest(self):
        """Drive the actual Streamlit page: folder mode -> Process -> results shown, no exception."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=240).run()
        at.radio(key="judge_mode").set_value("Rules only (no model)").run()
        at.radio(key="src_kind").set_value("Folder on this computer").run()
        at.text_input(key="folder_path").set_value(PACK).run()
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.tabs), 4)


# ================================================================== 2. parsing multiple episodes
@need_pack
class T02_Parsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R = full_run("none")
        cls.files = sorted(f for f in os.listdir(PACK) if f.endswith(".txt"))
        cls.truth = {}
        for f in cls.files:
            n, items = raw_items(os.path.join(PACK, f))
            cls.truth[n] = items
        cls.parsed = {int(e["episode_id"]): e for e in cls.R.extracted["episodes"]}

    def test_every_episode_parsed_in_order(self):
        self.assertEqual(sorted(self.parsed), sorted(self.truth))
        self.assertEqual(len(self.parsed), len(self.files))
        self.assertEqual(sorted(self.parsed), list(range(1, 51)))

    def test_item_counts_match_independent_reader(self):
        for n, items in self.truth.items():
            got = [it for sc in self.parsed[n]["scenes"] for it in sc["items"]]
            self.assertEqual(len(got), len(items), f"episode {n}: item count")
            for kind in ("camera", "action", "sound", "dialogue"):
                self.assertEqual(sum(i["kind"] == kind for i in got), sum(i["kind"] == kind for i in items), f"ep {n} {kind}")

    def test_dialogue_text_and_times_identical_to_raw(self):
        for n, items in self.truth.items():
            got = [it for sc in self.parsed[n]["scenes"] for it in sc["items"]]
            for g, t in zip(got, items):
                self.assertEqual(g["start_ts"], t["start"], f"ep {n} start")
                self.assertEqual(g["end_ts"], t["end"], f"ep {n} end")
                if t["kind"] == "dialogue":
                    self.assertEqual(re.sub(r"\s+", " ", g["text"]).strip().strip('"“”'), t["text"], f"ep {n} text")

    def test_no_parse_errors(self):
        self.assertEqual(self.R.extracted["summary"]["issues_by_severity"]["error"], 0)

    def test_episodes_do_not_leak_into_each_other(self):
        for n, e in self.parsed.items():
            self.assertTrue(e["source_file"].lower().startswith(f"episode-{n:03d}"), (n, e["source_file"]))


# ================================================================== 3. candidate generation
@need_pack
class T03_Candidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R = full_run("none")
        cls.cands = cls.R.candidates["candidates"]
        cls.truth = {raw_items(os.path.join(PACK, f))[0]: raw_items(os.path.join(PACK, f))[1]
                     for f in os.listdir(PACK) if f.endswith(".txt")}

    def test_candidates_exist_for_most_episodes(self):
        eps = {c["episode_id"] for c in self.cands}
        self.assertGreaterEqual(len(self.cands), 50)
        self.assertGreaterEqual(len(eps), 35)

    def test_ids_unique_and_durations_legal(self):
        ids = [c["candidate_id"] for c in self.cands]
        self.assertEqual(len(ids), len(set(ids)))
        for c in self.cands:
            self.assertTrue(20 <= c["duration_sec"] <= 60, (c["candidate_id"], c["duration_sec"]))

    def test_windows_sit_on_real_item_boundaries(self):
        for c in self.cands:
            ep = int(re.match(r"E(\d+)", c["candidate_id"])[1])
            starts = {ts_sec(i["start"]) for i in self.truth[ep]}
            ends = {ts_sec(i["end"]) for i in self.truth[ep]}
            self.assertTrue(any(abs(ts_sec(c["start_ts"]) - s) < 0.011 for s in starts), (c["candidate_id"], "start not on an item start"))
            self.assertTrue(any(abs(ts_sec(c["end_ts"]) - e) < 0.011 for e in ends), (c["candidate_id"], "end not on an item end"))

    def test_candidates_are_not_just_keyword_hits(self):
        """A window of pure small-talk must never be a candidate (context matters, not keywords)."""
        d = tempfile.mkdtemp(); write_synthetic(d)
        run = run_on(d)
        weak_ep = [c for c in run.candidates["candidates"] if c["candidate_id"].startswith("E102")]
        for c in weak_ep:
            ev = next(e for e in run.evaluated["evaluations"] if e["candidate_id"] == c["candidate_id"])
            self.assertNotEqual(ev["recommendation"], "top_pick")


# ================================================================== 4-6. hook / cliffhanger / standalone
class T04_06_Scoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(); write_synthetic(cls.d)
        cls.R = run_on(cls.d)
        cls.ev = cls.R.evaluated["evaluations"]

    def best(self, ep):
        c = [e for e in self.ev if e["candidate_id"].startswith(f"E{ep}")]
        return max(c, key=lambda e: e["overall_score"]) if c else None

    def test_scores_in_range_and_complete(self):
        for e in self.ev:
            for k in ("hook", "cliffhanger", "standalone", "intensity", "coherence", "character_clarity"):
                self.assertTrue(0 <= e["scores"][k] <= 10, (e["candidate_id"], k, e["scores"][k]))
            self.assertTrue(0 <= e["overall_score"] <= 10)

    def test_hook_strong_beats_weak(self):
        s, w = self.best(101), self.best(102)
        self.assertIsNotNone(s)
        if w:
            self.assertGreater(s["scores"]["hook"], w["scores"]["hook"])

    def test_cliffhanger_strong_beats_weak(self):
        s, w = self.best(101), self.best(102)
        if w:
            self.assertGreater(s["scores"]["cliffhanger"], w["scores"]["cliffhanger"])

    def test_hook_and_cliffhanger_quote_real_lines(self):
        for e in self.ev:
            hq = (e["details"]["hook"]["first_line"] or {}).get("quote")
            cq = (e["details"]["cliffhanger"]["last_line"] or {}).get("quote")
            raw = " ".join(l.strip() for f in os.listdir(self.d) for l in open(os.path.join(self.d, f), encoding="utf-8-sig"))
            for q in (hq, cq):
                if q:
                    self.assertIn(re.sub(r"\s+", " ", q).strip(), re.sub(r"\s+", " ", raw), f"{e['candidate_id']}: invented quote {q!r}")

    def test_standalone_flags_missing_context(self):
        """Episode 101's clip names 'Father' only at the end; a clip that starts mid-scene must not score perfect standalone."""
        s = self.best(101)
        self.assertLess(s["scores"]["standalone"], 10.0)

    def test_weak_dialogue_is_not_recommended(self):
        w = [e for e in self.ev if e["candidate_id"].startswith("E102")]
        for e in w:
            self.assertNotIn(e["recommendation"], ("top_pick", "recommend"), e["candidate_id"])

    def test_every_evaluation_explains_itself(self):
        for e in self.ev:
            self.assertTrue(e["explanations"]["hook"] and e["explanations"]["cliffhanger"] and e["explanations"]["standalone"])

    @need_pack
    def test_real_pack_replay_scores_blend_model_and_rules(self):
        run = full_run("replay")
        n = sum(1 for e in run.evaluated["evaluations"] if e.get("llm_model"))
        self.assertGreaterEqual(n, 40)
        for e in run.evaluated["evaluations"]:
            if e.get("llm_model"):
                self.assertTrue(0 <= e["scores"]["hook"] <= 10)


# ================================================================== 7. ranking
@need_pack
class T07_Ranking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = {"none": full_run("none"), "replay": full_run("replay")}

    def test_sorted_consecutive_and_one_per_episode(self):
        for mode, run in self.runs.items():
            tp = run.final["top_promos"]
            self.assertEqual([p["rank"] for p in tp], list(range(1, len(tp) + 1)), mode)
            sc = [p["overall_ranking_score"] for p in tp]
            self.assertEqual(sc, sorted(sc, reverse=True), mode)
            eps = [p["episodes"][0] for p in tp]
            self.assertEqual(len(eps), len(set(eps)), f"{mode}: two promos from the same episode")

    def test_every_promo_meets_the_gates(self):
        for mode, run in self.runs.items():
            for p in run.final["top_promos"]:
                s = p["scores"]
                self.assertGreaterEqual(s["hook"], 6.0, (mode, p["promo_id"]))
                self.assertGreaterEqual(s["cliffhanger"], 6.0, (mode, p["promo_id"]))
                self.assertGreaterEqual(s["standalone"], 5.5, (mode, p["promo_id"]))
                self.assertTrue(20 <= p["total_duration_sec"] <= 60)

    def test_posting_plan_is_a_permutation(self):
        for run in self.runs.values():
            ids = {p["promo_id"] for p in run.final["top_promos"]}
            plan = run.final["posting_plan"]
            self.assertEqual({x["promo_id"] for x in plan}, ids)
            self.assertEqual(sorted(x["post_order"] for x in plan), list(range(1, len(ids) + 1)))

    def test_top_n_respected_and_deterministic(self):
        a = run_on(PACK, "none", top=5); b = run_on(PACK, "none", top=5)
        self.assertEqual(len(a.final["top_promos"]), 5)
        self.assertEqual([p["candidate_id"] for p in a.final["top_promos"]], [p["candidate_id"] for p in b.final["top_promos"]])

    def test_no_adjacent_episode_duplicates_in_top10(self):
        """QA audit finding: three adjacent-episode near-duplicate pairs sat in the top 10."""
        for mode, run in self.runs.items():
            eps = sorted(p["episodes"][0] for p in run.final["top_promos"])
            adj = [(a, b) for a, b in zip(eps, eps[1:]) if b - a == 1]
            self.assertLessEqual(len(adj), 1, f"{mode}: adjacent-episode pairs {adj}")


# ================================================================== 8. timestamp accuracy (against raw files only)
@need_pack
class T08_Timestamps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth = {}
        for f in os.listdir(PACK):
            if f.endswith(".txt"):
                n, items = raw_items(os.path.join(PACK, f)); cls.truth[n] = items
        cls.runs = {"none": full_run("none"), "replay": full_run("replay")}

    def _dialogue(self, ep):
        return [i for i in self.truth[ep] if i["kind"] == "dialogue"]

    def test_segment_bounds_are_real_timestamps(self):
        for mode, run in self.runs.items():
            for p in run.final["top_promos"]:
                for s in p["segments"]:
                    items = self.truth[s["episode"]]
                    self.assertTrue(any(abs(ts_sec(i["start"]) - s["start_sec"]) < 0.011 for i in items), (mode, p["promo_id"], "start"))
                    self.assertTrue(any(abs(ts_sec(i["end"]) - s["end_sec"]) < 0.011 for i in items), (mode, p["promo_id"], "end"))
                    self.assertAlmostEqual(s["end_sec"] - s["start_sec"], s["duration_sec"], places=1)
                    self.assertAlmostEqual(ts_sec(s["start_ts"]), s["start_sec"], places=2)
                    self.assertAlmostEqual(ts_sec(s["end_ts"]), s["end_sec"], places=2)

    def test_total_duration_is_sum_of_segments(self):
        for run in self.runs.values():
            for p in run.final["top_promos"]:
                self.assertAlmostEqual(sum(s["duration_sec"] for s in p["segments"]), p["total_duration_sec"], places=1)

    def test_shown_dialogue_exists_verbatim_with_same_times(self):
        for mode, run in self.runs.items():
            for p in run.final["top_promos"]:
                ep = p["segments"][0]["episode"]
                raw = self._dialogue(ep)
                shown = p["opening_hook"]["first_3_seconds"] + p["ending_cliffhanger"]["last_spoken_lines"]
                for l in shown:
                    if l["kind"] != "dialogue":
                        continue
                    hit = [r for r in raw if r["start"] == l["start_ts"] and r["end"] == l["end_ts"]
                           and re.sub(r"\s+", " ", r["text"]) == re.sub(r"\s+", " ", l["text"].strip('"“”'))]
                    self.assertTrue(hit, f"{mode} {p['promo_id']}: {l['text']!r} not in raw ep {ep} at {l['start_ts']}")

    def test_every_displayed_line_is_inside_the_clip(self):
        """Regression test for QA finding F1 (an ending line that finishes after the clip ends)."""
        for mode, run in self.runs.items():
            for p in run.final["top_promos"]:
                lo, hi = p["segments"][0]["start_sec"], p["segments"][-1]["end_sec"]
                for l in p["opening_hook"]["first_3_seconds"] + p["ending_cliffhanger"]["last_spoken_lines"]:
                    if l["kind"] == "dialogue":
                        self.assertGreaterEqual(ts_sec(l["start_ts"]), lo - 0.01, (mode, p["promo_id"]))
                        self.assertLessEqual(ts_sec(l["end_ts"]), hi + 0.01, f"{mode} {p['promo_id']}: line ends {l['end_ts']} after clip end")

    def test_validator_reports_no_errors_on_final_list(self):
        for mode, run in self.runs.items():
            for p in run.final["top_promos"]:
                errs = [v["msg"] for v in results.validate_promo(p) if v["level"] == "error"]
                self.assertEqual(errs, [], f"{mode} {p['promo_id']}")

    def test_no_clip_cuts_through_a_spoken_line(self):
        """A promo must neither start nor end in the middle of a line of dialogue."""
        for mode, run in self.runs.items():
            for p in run.final["top_promos"]:
                for s in p["segments"]:
                    for r in self._dialogue(s["episode"]):
                        a, b = ts_sec(r["start"]), ts_sec(r["end"])
                        self.assertFalse(a < s["start_sec"] - 0.01 and b > s["start_sec"] + 0.01, f"{mode} {p['promo_id']} cuts the start of {r['text'][:30]!r}")
                        self.assertFalse(a < s["end_sec"] - 0.01 and b > s["end_sec"] + 0.01, f"{mode} {p['promo_id']} cuts the end of {r['text'][:30]!r}")

    def test_clip_script_matches_raw(self):
        run = self.runs["replay"]
        for p in run.final["top_promos"]:
            df = results.clip_script(run, p)
            for s in p["segments"]:
                raw = [r for r in self.truth[s["episode"]] if r["kind"] != "camera" and s["start_sec"] - 1e-6 <= ts_sec(r["start"]) < s["end_sec"]]
                self.assertEqual(len(df[df["Episode"] == s["episode"]]), len(raw), p["promo_id"])


# ================================================================== 9. exports
@need_pack
class T09_Exports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R = full_run("replay")
        cls.csv = results.to_csv(cls.R)
        cls.json = results.to_detail_json(cls.R)

    def test_csv_parses_and_matches_final(self):
        df = pd.read_csv(io.BytesIO(self.csv), dtype=str, keep_default_na=False)
        self.assertEqual(df["promo_id"].nunique(), len(self.R.final["top_promos"]))
        for p in self.R.final["top_promos"]:
            rows = df[df["promo_id"] == p["promo_id"]]
            self.assertEqual(len(rows), len(p["segments"]))
            self.assertEqual(list(rows["start_timestamp"]), [s["start_ts"] for s in p["segments"]])
            self.assertEqual(list(rows["end_timestamp"]), [s["end_ts"] for s in p["segments"]])
            self.assertEqual(float(rows["overall_score"].iloc[0]), p["overall_ranking_score"])
            self.assertEqual(float(rows["total_duration_sec"].iloc[0]), p["total_duration_sec"])

    def test_csv_timestamps_not_mangled_by_spreadsheet_style_parsing(self):
        df = pd.read_csv(io.BytesIO(self.csv), dtype=str)
        self.assertTrue(df["start_timestamp"].str.fullmatch(r"\d+:\d{2}\.\d{2}").all())
        self.assertTrue(df["end_timestamp"].str.fullmatch(r"\d+:\d{2}\.\d{2}").all())

    def test_csv_utf8_and_quotes_survive(self):
        text = self.csv.decode("utf-8")
        df = pd.read_csv(io.StringIO(text))
        self.assertTrue(df["opening_hook"].astype(str).str.len().gt(0).all())
        self.assertTrue(df["why_it_works"].astype(str).str.len().gt(0).all())

    def test_csv_columns_cover_the_brief(self):
        need = {"rank", "promo_id", "episode", "start_timestamp", "end_timestamp", "total_duration_sec", "opening_hook",
                "ending_cliffhanger", "why_it_works", "hook_score", "cliffhanger_score", "standalone_score", "overall_score"}
        self.assertTrue(need <= set(pd.read_csv(io.BytesIO(self.csv)).columns))

    def test_json_valid_complete_and_consistent(self):
        d = json.loads(self.json)
        self.assertEqual(len(d["final_promos"]), len(self.R.final["top_promos"]))
        self.assertEqual(len(d["all_evaluations"]), len(self.R.evaluated["evaluations"]))
        self.assertEqual(d["pipeline"]["episodes"], 50)
        for p in d["final_promos"]:
            self.assertIn("validation", p)
        self.assertEqual(len(d["rejected_or_not_selected"]) + len(d["final_promos"]), len(d["all_evaluations"]))
        self.assertTrue(d["limitations"])

    def test_json_has_no_nan_or_infinity(self):
        self.assertNotRegex(self.json.decode("utf-8"), r"\bNaN\b|\bInfinity\b")

    def test_exports_for_empty_result_do_not_crash(self):
        """A screenplay that yields no eligible promos must still export valid (empty) files."""
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "ep.txt"), "w", encoding="utf-8") as fh:
            fh.write(make_episode(1, [weak_scene()]))
        run = run_on(d)
        pd.read_csv(io.BytesIO(results.to_csv(run))) if results.to_csv(run).strip() else None
        json.loads(results.to_detail_json(run))
        results.promo_table(run); results.rejection_table(run)


# ================================================================== CLI (the documented command line)
class T11_CLI(unittest.TestCase):
    def test_run_pipeline_cli_on_new_input(self):
        import subprocess
        src, out = tempfile.mkdtemp(), os.path.join(tempfile.mkdtemp(), "out")
        write_synthetic(src)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "run_pipeline.py"), src, "-o", out, "--allow-unverified"],
                           capture_output=True, text=True, cwd=tempfile.mkdtemp())
        self.assertEqual(r.returncode, 0, r.stderr[-800:] + r.stdout[-400:])
        for f in ("extracted_scenes.json", "candidate_promos.json", "evaluated_promos.json", "evaluated_promos.csv",
                  "ranked_promos.json", "ranked_promos.md"):
            self.assertTrue(os.path.getsize(os.path.join(out, f)) > 0, f)
        self.assertTrue(json.load(open(os.path.join(out, "ranked_promos.json")))["top_promos"])

    def test_cli_with_no_usable_input_exits_with_message(self):
        import subprocess
        src = tempfile.mkdtemp(); open(os.path.join(src, "x.txt"), "w").write("nothing")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "run_pipeline.py"), src, "-o", os.path.join(src, "o")],
                           capture_output=True, text=True, cwd=tempfile.mkdtemp())
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("Traceback", r.stderr)


# ================================================================== 10. a new screenplay input
class T10_NewInput(unittest.TestCase):
    def test_synthetic_show_end_to_end(self):
        d = tempfile.mkdtemp(); write_synthetic(d)
        run = run_on(d)
        self.assertEqual(run.final["pipeline"]["episodes"], 3)
        top = run.final["top_promos"]
        self.assertTrue(top, "the two strong episodes must yield promos")
        self.assertTrue({p["episodes"][0] for p in top} <= {101, 103}, "the small-talk episode must not be promoted")
        for p in top:                                          # every displayed line exists in the input we wrote
            raw = " ".join(open(os.path.join(d, f), encoding="utf-8").read() for f in os.listdir(d))
            for l in p["opening_hook"]["first_3_seconds"] + p["ending_cliffhanger"]["last_spoken_lines"]:
                if l["kind"] == "dialogue":
                    self.assertIn(l["text"].strip('"'), raw)
            self.assertTrue(20 <= p["total_duration_sec"] <= 60)
        self.assertIn("UNVERIFIED", top[0]["verification"].upper())

    def test_crlf_and_bom_input(self):
        d = tempfile.mkdtemp(); write_synthetic(d, "crlf")
        run = run_on(d)
        self.assertEqual(run.final["pipeline"]["episodes"], 3)
        self.assertTrue(run.final["top_promos"])

    def test_zip_upload_of_new_input_through_the_app_path(self):
        src = tempfile.mkdtemp(); write_synthetic(src)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for f in os.listdir(src):
                z.write(os.path.join(src, f), f"myshow/{f}")
        d = tempfile.mkdtemp()
        runner.ingest_uploads([Up("myshow.zip", buf.getvalue())], d)
        self.assertEqual(run_on(d).final["pipeline"]["episodes"], 3)

    @unittest.skipUnless(os.path.exists(runner.DEFAULT_REPLAY), "saved verdict file is not shipped in a shareable copy")
    def test_replay_mode_on_unknown_screenplay_falls_back_without_crashing(self):
        d = tempfile.mkdtemp(); write_synthetic(d)
        run = run_on(d, "replay")
        self.assertTrue(run.final["top_promos"])
        self.assertTrue(all("UNVERIFIED" in p["verification"].upper() for p in run.final["top_promos"]))

    def test_file_naming_does_not_matter(self):
        d = tempfile.mkdtemp()
        for i, (n, t) in enumerate({"zzz.txt": make_episode(7, [strong_scene()]), "a b c.txt": make_episode(3, [strong_scene()])}.items()):
            with open(os.path.join(d, n), "w", encoding="utf-8") as fh:
                fh.write(t)
        run = run_on(d)
        self.assertEqual(sorted(int(e["episode_id"]) for e in run.extracted["episodes"]), [3, 7])

    def test_non_contiguous_and_single_episode(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "only.txt"), "w", encoding="utf-8") as fh:
            fh.write(make_episode(42, [strong_scene()]))
        run = run_on(d)
        self.assertEqual(run.final["pipeline"]["episodes"], 1)

    def test_bad_inputs_give_clear_errors_not_stack_traces(self):
        for content in ("", "just some words", "\x00\x01\x02binary\xff", "EPISODE NUMBER: Episode 1\n\nno scenes at all"):
            d = tempfile.mkdtemp()
            with open(os.path.join(d, "ep.txt"), "w", encoding="utf-8", errors="replace") as fh:
                fh.write(content)
            with self.assertRaises(runner.PipelineError):
                run_on(d)

    def test_one_bad_file_among_good_ones_does_not_kill_the_run(self):
        d = tempfile.mkdtemp(); write_synthetic(d)
        with open(os.path.join(d, "broken.txt"), "w", encoding="utf-8") as fh:
            fh.write("garbage")
        run = run_on(d)
        self.assertGreaterEqual(run.final["pipeline"]["episodes"], 3)

    @need_pack
    def test_renumbered_subset_of_real_pack_gives_same_promos_as_full_run(self):
        """Same episodes, different filenames and a subset: results for those episodes must not depend on the rest."""
        d = tempfile.mkdtemp()
        keep = [36, 1, 9, 15, 12]
        for f in os.listdir(PACK):
            n = int(re.match(r"episode-(\d+)", f)[1])
            if n in keep:
                shutil.copy(os.path.join(PACK, f), os.path.join(d, f"renamed_{n}.txt"))
        sub = run_on(d)
        full = full_run("none")
        want = {p["candidate_id"] for p in full.final["top_promos"] if p["episodes"][0] in keep}
        got = {p["candidate_id"] for p in sub.final["top_promos"]}
        self.assertTrue(want <= got, f"promos lost when run on a subset: {want - got}")

    @need_pack
    def test_two_runs_do_not_contaminate_each_other(self):
        d1 = tempfile.mkdtemp(); write_synthetic(d1)
        a = run_on(d1); b = full_run("none"); c = run_on(d1)
        self.assertEqual([p["candidate_id"] for p in a.final["top_promos"]], [p["candidate_id"] for p in c.final["top_promos"]])
        self.assertFalse(os.path.exists(a.workdir), "temp working dir must be cleaned up after a run")


if __name__ == "__main__":
    unittest.main(verbosity=2)

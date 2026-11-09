"""Headless tests of the Streamlit app (streamlit.testing). They run the REAL pipeline on the sample pack."""
import io, os, sys, unittest, zipfile
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.abspath(__file__))]
from streamlit.testing.v1 import AppTest
from promo_app import runner, results

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); SAMPLE = os.path.join(ROOT, "data", "screenplay_input")
HAVE = os.path.isdir(SAMPLE) and any(f.endswith(".txt") for f in os.listdir(SAMPLE))

def processed(mode_index=0, top=None):
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    at.radio(key="src_kind").set_value("Folder on this computer").run()
    at.text_input(key="folder_path").set_value(SAMPLE).run()
    at.button[0].click().run()
    return at

@unittest.skipUnless(HAVE, "needs data/screenplay_input/ with the screenplay .txt files")
class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.at = processed()
    def test_no_exception_and_title(self):
        self.assertFalse(self.at.exception); self.assertIn("Promo Discovery Engine", " ".join(m.value for m in self.at.markdown[:3]))
    def test_results_are_real_pipeline_output(self):
        run = self.at.session_state["run"]
        self.assertEqual(run.final["pipeline"]["episodes"], 50); self.assertEqual(len(run.final["top_promos"]), 10)
        for p in run.final["top_promos"]: self.assertTrue(20 <= p["total_duration_sec"] <= 60)
    def test_tabs_and_downloads_render(self):
        self.assertEqual(len(self.at.tabs), 4); self.assertTrue(len(self.at.dataframe) >= 2)
    def test_csv_and_json_exports(self):
        run = self.at.session_state["run"]; import json, pandas as pd
        df = pd.read_csv(io.BytesIO(results.to_csv(run))); self.assertEqual(df["promo_id"].nunique(), 10)
        self.assertTrue({"start_timestamp", "end_timestamp", "hook_score", "overall_score"} <= set(df.columns))
        d = json.loads(results.to_detail_json(run)); self.assertEqual(len(d["final_promos"]), 10); self.assertEqual(len(d["all_evaluations"]), 119)
    def test_rejected_table_explains_failures(self):
        df = results.rejection_table(self.at.session_state["run"]); self.assertTrue(len(df) > 0)
        self.assertTrue(df["Why it failed"].str.len().gt(0).all()); self.assertIn("Rejected", set(df["Status"]))
    def test_validator_clean_on_final_list_and_still_catches_a_bad_clip(self):
        import copy
        run = self.at.session_state["run"]
        for p in run.final["top_promos"]:
            self.assertEqual([v for v in results.validate_promo(p) if v["level"] == "error"], [])
        bad = copy.deepcopy(run.final["top_promos"][0])                       # simulate the old F1 defect
        bad["ending_cliffhanger"]["last_spoken_lines"][-1]["end_ts"] = "09:59.00"
        self.assertTrue(any("not fully inside the clip" in v["msg"] for v in results.validate_promo(bad)))
    def test_clip_script_matches_window(self):
        run = self.at.session_state["run"]; p = run.final["top_promos"][0]; df = results.clip_script(run, p)
        self.assertTrue(len(df) > 0 and df["Start"].iloc[0] >= p["segments"][0]["start_ts"])

class UnitTests(unittest.TestCase):
    class F:                       # minimal UploadedFile stand-in
        def __init__(self, n, b): self.name, self._b = n, b
        def getvalue(self): return self._b
    def test_zip_upload_blocks_path_traversal(self):
        import tempfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z: z.writestr("../../evil.txt", "x"); z.writestr("ok/episode-001-A.txt", "y")
        d = tempfile.mkdtemp(); saved = runner.ingest_uploads([self.F("a.zip", buf.getvalue())], d)
        self.assertEqual(sorted(os.listdir(d)), sorted(saved)); self.assertFalse(os.path.exists(os.path.join(d, "..", "evil.txt")))
    def test_no_txt_is_a_clear_error(self):
        import tempfile
        with self.assertRaises(runner.PipelineError): runner.ingest_uploads([self.F("a.pdf", b"x")], tempfile.mkdtemp())
    def test_garbage_screenplay_gives_user_error(self):
        import tempfile
        d = tempfile.mkdtemp(); 
        with open(os.path.join(d, "episode-001-x.txt"), "w") as fh: fh.write("hello, not a screenplay")
        with self.assertRaises(runner.PipelineError): runner.run_pipeline(d, runner.LLMChoice())
    def test_missing_key_is_a_clear_error(self):
        with self.assertRaises(runner.PipelineError): runner.LLMChoice(mode="anthropic").apply_env()

if __name__ == "__main__": unittest.main(verbosity=1)

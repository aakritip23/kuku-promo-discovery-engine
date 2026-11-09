"""quality_checker.py must pass on real pipeline output AND fail when the output is tampered with."""
import copy, json, os, shutil, sys, tempfile, unittest
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.abspath(__file__))]
import quality_checker as qc
import testdata
from promo_app import runner


class QualityChecker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = testdata.PACK if testdata.HAVE_PACK else testdata.DEMO
        cls.out = tempfile.mkdtemp()
        runner.run_pipeline(cls.src, runner.LLMChoice(mode="none"), workdir=cls.out)

    def _tampered(self, edit):
        d = tempfile.mkdtemp()
        for f in os.listdir(self.out):
            shutil.copy(os.path.join(self.out, f), d)
        doc = json.load(open(os.path.join(d, "ranked_promos.json"), encoding="utf-8"))
        edit(doc)
        json.dump(doc, open(os.path.join(d, "ranked_promos.json"), "w", encoding="utf-8"))
        return qc.run_checks(self.src, d)

    def test_passes_on_real_output_and_writes_report(self):
        out = os.path.join(self.out, "quality_report.json")
        self.assertEqual(qc.main([self.src, self.out]), 0)
        rep = json.load(open(out, encoding="utf-8"))
        self.assertEqual(rep["summary"]["checks_run"], 5)
        self.assertTrue(rep["summary"]["all_passed"])
        self.assertTrue(rep["what_these_checks_do_not_cover"])

    def test_detects_a_changed_timestamp(self):
        def edit(d): d["top_promos"][0]["segments"][0]["end_ts"] = "09:59.00"
        rep = self._tampered(edit)
        self.assertFalse(rep["summary"]["all_passed"])

    def test_detects_an_invented_quote(self):
        def edit(d): d["top_promos"][0]["opening_hook"]["first_3_seconds"][0]["text"] = "This line was never in the screenplay."
        rep = self._tampered(edit)
        self.assertFalse(next(c for c in rep["checks"] if c["check"] == "ranked")["passed"])

    def test_detects_a_line_outside_the_clip(self):
        def edit(d):
            ls = d["top_promos"][0]["ending_cliffhanger"]["last_spoken_lines"]
            ls[-1]["end_ts"] = "09:59.00"
        rep = self._tampered(edit)
        self.assertFalse(next(c for c in rep["checks"] if c["check"] == "boundaries")["passed"])

    def test_detects_wrong_duration(self):
        def edit(d): d["top_promos"][0]["total_duration_sec"] += 5
        self.assertFalse(self._tampered(edit)["summary"]["all_passed"])


if __name__ == "__main__":
    unittest.main()

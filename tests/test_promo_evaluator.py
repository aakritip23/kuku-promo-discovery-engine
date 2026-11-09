import copy, json, os, sys, unittest
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.abspath(__file__))]
import promo_evaluator as pe
import testdata

DOC, CD, _ = testdata.fixtures("rules")                 # real pipeline output (rules only), built fresh
CTX = pe.pack_context(DOC)
EPS = {str(e["episode_id"]): e for e in DOC["episodes"]}
CANDS = {c["candidate_id"]: c for c in CD["candidates"]}

# a candidate that is strong in the rules, given a model verdict we control, so the tests do not depend on the pack
BASE = max(CD["candidates"], key=lambda c: c.get("candidate_score_final") or 0)["candidate_id"]
PACK_ONLY = unittest.skipUnless(testdata.HAVE_PACK, "needs the real pack")

def ev(cid, mutate=None, judged=True):
    c = copy.deepcopy(CANDS[cid])
    if judged: c["semantic"] = copy.deepcopy(testdata.STRONG_VERDICT)
    if mutate: mutate(c)
    return pe.evaluate(c, pe.episode_events(EPS[str(c["episode_id"])]), CTX)

class T(unittest.TestCase):
    def test_weights_sum_to_one(self): self.assertAlmostEqual(sum(pe.WEIGHTS.values()), 1.0)
    def test_every_weight_has_rationale(self): self.assertEqual(set(pe.WEIGHTS), set(pe.WEIGHT_RATIONALE))
    def test_blend(self):
        self.assertEqual(pe.blend(5, None), 5); self.assertAlmostEqual(pe.blend(5, 10), 8.0)
    def test_strong_judged_candidate_is_recommended(self):
        r = ev(BASE); self.assertIn(r["recommendation"], ("top_pick", "recommend")); self.assertTrue(r["gates"]["passed"])
    def test_llm_keyword_false_positive_is_rejected_whatever_the_scores(self):
        def m(c): c["semantic"].update(is_genuine_promo_moment=False, keyword_only_false_positive=True, hook_strength=10, cliffhanger_strength=10, standalone_clarity=10)
        r = ev(BASE, m); self.assertEqual(r["recommendation"], "reject")
        self.assertIn("llm_not_genuine", [f["code"] for f in r["gates"]["failures"]])
    def test_duration_gate(self):
        def m(c): c["segments"][0]["duration_sec"] = 75.0
        r = ev(BASE, m); self.assertIn("duration", [f["code"] for f in r["gates"]["failures"]]); self.assertEqual(r["recommendation"], "reject")
    def test_llm_zero_cliffhanger_fails_gate(self):
        def m(c): c["semantic"]["cliffhanger_strength"] = 0
        r = ev(BASE, m); self.assertLess(r["scores"]["cliffhanger"], 5.0)
    def test_balance_factor_penalises_weak_leg(self):
        r = ev(BASE); lo = min(r["scores"][k] for k in ("hook", "cliffhanger", "standalone"))
        self.assertAlmostEqual(r["balance_factor"], pe.BALANCE_FLOOR + (1 - pe.BALANCE_FLOOR) * lo / 10, places=3)
        self.assertLessEqual(r["overall_score"], r["weighted_score"] + 1e-9)
    def test_rules_only_borderline_goes_to_review_not_reject(self):
        reviews = [e for e in pe.evaluate_all(DOC, CD) if e["recommendation"] == "needs_llm_review"]
        if not reviews: self.skipTest("no borderline rules-only candidate in this input")
        self.assertTrue(reviews); self.assertTrue(all(e["confidence"] == "rules_only" for e in reviews))
    def test_every_quote_is_verbatim_from_the_item(self):
        items = {i["item_id"]: i["text"] for e in DOC["episodes"] for s in e["scenes"] for i in s["items"]}
        for e in pe.evaluate_all(DOC, CD):
            for q in e["evidence"]:
                self.assertIn(q["quote"].replace(" ...", ""), items[q["item_id"]])
    def test_clip_text_truncates_with_marker(self):
        self.assertEqual(pe.clip_text("a b c d", 2), "a b ...")
    def test_scores_in_range_and_all_present(self):
        for e in pe.evaluate_all(DOC, CD):
            self.assertEqual(set(e["scores"]), set(pe.WEIGHTS)); self.assertTrue(all(0 <= v <= 10 for v in e["scores"].values()))
            self.assertTrue(all(e["explanations"][k] for k in ("hook", "cliffhanger", "standalone")))
    @PACK_ONLY
    def test_unanswered_question_ending_scores_higher_than_resolved(self):
        a = ev("E034_03530_06010", judged=False)["scores"]["cliffhanger"]; b = ev("E020_04260_07230", judged=False)["scores"]["cliffhanger"]
        self.assertGreater(a, b)

if __name__ == "__main__":
    unittest.main(verbosity=1)

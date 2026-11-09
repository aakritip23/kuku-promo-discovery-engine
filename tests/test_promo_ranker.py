import json, os, sys, unittest
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.abspath(__file__))]
import promo_ranker as fr
import testdata
DOC, CD, ED = testdata.fixtures("judged")               # real pipeline output, built fresh (pack+verdicts, else demo_input)
EPS = {str(e["episode_id"]): e for e in DOC["episodes"]}; CANDS = {c["candidate_id"]: c for c in CD["candidates"]}
EV = ED["evaluations"]

class T(unittest.TestCase):
    def setUp(self): self.chosen, self.skipped, self.rejected = fr.select(EV, CANDS, 10, True)
    def test_only_verified_unless_allowed(self): self.assertTrue(all(e["confidence"] == "rules+llm" for e in fr.select(EV, CANDS, 10, False)[0]))
    def test_all_meet_final_thresholds(self):
        for e in self.chosen:
            self.assertEqual(fr.eligibility(e, CANDS[e["candidate_id"]]), [])
            self.assertTrue(20 <= e["duration_sec"] <= 60)
    def test_one_per_episode(self):
        eps = [e["episode_id"] for e in self.chosen]; self.assertEqual(len(eps), len(set(eps)))
    def test_sorted_by_score(self):
        s = [e["overall_score"] for e in self.chosen]; self.assertEqual(s, sorted(s, reverse=True))
    def test_weak_dimension_blocks_selection(self):
        e = dict(EV[0]); e["scores"] = dict(e["scores"], hook=5.0)
        self.assertTrue(any("hook" in r for r in fr.eligibility(e, CANDS[e["candidate_id"]])))
    def test_big_gap_blocks_selection(self):
        e = json.loads(json.dumps(EV[0])); s = e["segments"][0]
        e["segments"] = [s, dict(s, start_sec=s["end_sec"] + 20, end_sec=s["end_sec"] + 40, duration_sec=20)]
        self.assertTrue(any("gap" in r for r in fr.eligibility(e, CANDS[e["candidate_id"]])))
    def test_text_fields_are_verbatim_items(self):
        items = {i["item_id"]: i["text"].strip() for e in DOC["episodes"] for s in e["scenes"] for i in s["items"]}
        for e in self.chosen:
            p = fr.build_promo(1, e, CANDS[e["candidate_id"]], EPS)
            for l in p["opening_hook"]["first_3_seconds"] + p["ending_cliffhanger"]["last_spoken_lines"]:
                self.assertEqual(l["text"], items[l["item_id"]])
    def test_posting_plan_is_a_permutation(self):
        ps = [fr.build_promo(i, e, CANDS[e["candidate_id"]], EPS) for i, e in enumerate(self.chosen, 1)]
        plan = fr.posting_plan(ps); self.assertEqual(sorted(x["promo_id"] for x in plan), sorted(p["promo_id"] for p in ps))
        for a, b in zip(plan, plan[1:]):
            pa = next(p for p in ps if p["promo_id"] == a["promo_id"]); pb = next(p for p in ps if p["promo_id"] == b["promo_id"])
            self.assertGreater(abs(pa["episodes"][0] - pb["episodes"][0]), 1)

if __name__ == "__main__": unittest.main()

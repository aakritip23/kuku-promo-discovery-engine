"""Tests for screenplay_parser. Run: python -m pytest tests (uses data/screenplay_input/ or $PROMO_SAMPLE_DIR), or python tests/test_screenplay_parser.py [PACK_DIR]"""
import os
import sys
import tempfile
import unittest

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.abspath(__file__))]
import screenplay_parser as sp  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_default = os.environ.get("PROMO_SAMPLE_DIR", os.path.join(_ROOT, "data", "screenplay_input"))
PACK = sys.argv.pop(1) if (__name__ == "__main__" and len(sys.argv) > 1) else _default      # pytest must not have its args eaten

GOOD = """EPISODE NUMBER: Episode 9
EPISODE LOGLINE: A test logline.
CONTINUITY NOTES: Test notes.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. HALL - DAY
Scene start: 00:00.00
Scene end: 00:10.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:04.00)
Wide shot of the hall.

**LADY ANNA (V.O.) (Shaking):**
Start: (00:01.50) | End: (00:03.00)
"Why did you come?"

**ACTION:**
Start: (00:05.00) | End: (00:09.00)
Anna slaps him.

--- SCRIPT END ---
"""


def write(dirpath, name, content, mode="w"):
    p = os.path.join(dirpath, name)
    with open(p, mode, **({} if "b" in mode else {"encoding": "utf-8"})) as fh:
        fh.write(content)
    return p


def codes(ep):
    return {i["code"] for i in ep["issues"]}


class HelperTests(unittest.TestCase):
    def test_parse_timestamp(self):
        self.assertEqual(sp.parse_timestamp("00:11.30"), 11.3)
        self.assertEqual(sp.parse_timestamp("02:27.50"), 147.5)
        for bad in ("0:4.8", "00:61.00", "abc", "", None, "00:11"):
            self.assertIsNone(sp.parse_timestamp(bad), bad)

    def test_block_label(self):
        r = sp.parse_block_label("LADY ANNA (V.O.) (Shaking)")
        self.assertEqual((r["kind"], r["character"], r["voice_mode"], r["delivery"]),
                         ("dialogue", "LADY ANNA", "V.O.", "Shaking"))
        r = sp.parse_block_label("BLOND NOBLEMAN (Rupert, Cold)")
        self.assertEqual((r["character"], r["delivery"]), ("BLOND NOBLEMAN", "Rupert, Cold"))
        self.assertEqual(sp.parse_block_label("CAMERA")["kind"], "camera")
        self.assertEqual(sp.parse_block_label("SOUND FX")["kind"], "sound")
        self.assertEqual(sp.parse_block_label("CATHERINE (O.S.)")["voice_mode"], "O.S.")

    def test_filename(self):
        self.assertEqual(sp.parse_filename("x/episode-007-Episode 1671709.txt"), (7, "Episode 1671709"))
        self.assertEqual(sp.parse_filename("notes.txt"), (None, None))


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def parse(self, content, name="episode-009-Episode 1.txt", mode="w"):
        return sp.parse_episode_file(write(self.d, name, content, mode))

    def test_clean_file(self):
        ep = self.parse(GOOD)
        self.assertEqual(ep["parse_status"], "ok")
        self.assertEqual((ep["episode_id"], ep["n_scenes"], ep["n_items"]), (9, 1, 3))
        d = ep["scenes"][0]["items"][1]
        self.assertEqual((d["character"], d["voice_mode"], d["delivery"], d["text"]),
                         ("LADY ANNA", "V.O.", "Shaking", "Why did you come?"))
        self.assertEqual((d["start_ts"], d["end_ts"], d["start_sec"], d["end_sec"]),
                         ("00:01.50", "00:03.00", 1.5, 3.0))
        self.assertEqual(ep["duration_sec"], 10.0)

    def test_missing_item_timestamp_not_invented(self):
        ep = self.parse(GOOD.replace("Start: (00:01.50) | End: (00:03.00)\n", ""))
        it = ep["scenes"][0]["items"][1]
        self.assertIn("item_missing_timestamp", codes(ep))
        self.assertIsNone(it["start_ts"]); self.assertIsNone(it["start_sec"])
        self.assertEqual(it["text"], "Why did you come?")   # the words are still captured
        self.assertEqual(ep["parse_status"], "ok_with_errors")

    def test_malformed_timestamp_kept_raw(self):
        ep = self.parse(GOOD.replace("(00:01.50)", "(0:1.5)"))
        it = ep["scenes"][0]["items"][1]
        self.assertIn("item_timestamp_malformed", codes(ep))
        self.assertEqual(it["start_ts"], "0:1.5")      # raw text preserved
        self.assertIsNone(it["start_sec"])             # no guessed number
        self.assertEqual(it["end_sec"], 3.0)           # the good half is still used

    def test_unreadable_time_line(self):
        ep = self.parse(GOOD.replace("Start: (00:01.50) | End: (00:03.00)", "Start: 00:01.50 to 00:03.00"))
        it = ep["scenes"][0]["items"][1]
        self.assertIn("time_line_malformed", codes(ep))
        self.assertIsNone(it["start_ts"])
        self.assertEqual(it["text"], "Why did you come?")

    def test_missing_scene_end_not_inferred(self):
        ep = self.parse(GOOD.replace("Scene end: 00:10.00\n", ""))
        sc = ep["scenes"][0]
        self.assertIn("scene_timestamp_missing", codes(ep))
        self.assertIsNone(sc["end_ts"]); self.assertIsNone(sc["end_sec"])
        self.assertIsNone(ep["duration_sec"])

    def test_scene_timestamp_malformed(self):
        ep = self.parse(GOOD.replace("Scene end: 00:10.00", "Scene end: ten seconds"))
        self.assertIn("scene_timestamp_malformed", codes(ep))
        self.assertEqual(ep["scenes"][0]["end_ts"], "ten seconds")
        self.assertIsNone(ep["scenes"][0]["end_sec"])

    def test_end_before_start_warns_but_keeps(self):
        ep = self.parse(GOOD.replace("Start: (00:01.50) | End: (00:03.00)", "Start: (00:03.00) | End: (00:01.50)"))
        it = ep["scenes"][0]["items"][1]
        self.assertIn("item_end_not_after_start", codes(ep))
        self.assertEqual((it["start_ts"], it["end_ts"]), ("00:03.00", "00:01.50"))

    def test_item_outside_scene_flagged(self):
        ep = self.parse(GOOD.replace("End: (00:09.00)", "End: (00:12.00)"))
        self.assertIn("item_outside_scene", codes(ep))

    def test_missing_script_start(self):
        ep = self.parse(GOOD.replace("--- SCRIPT START ---", ""))
        self.assertEqual(ep["parse_status"], "failed")
        self.assertIn("script_start_missing", codes(ep))

    def test_missing_script_end_warns(self):
        ep = self.parse(GOOD.replace("--- SCRIPT END ---", ""))
        self.assertIn("script_end_missing", codes(ep))
        self.assertEqual(ep["n_items"], 3)

    def test_empty_file(self):
        ep = self.parse("\n\n")
        self.assertEqual(ep["parse_status"], "failed")
        self.assertIn("empty_file", codes(ep))

    def test_episode_number_falls_back_to_filename(self):
        ep = self.parse(GOOD.replace("EPISODE NUMBER: Episode 9\n", ""))
        self.assertEqual(ep["episode_id"], 9)
        self.assertIn("episode_number_missing", codes(ep))

    def test_episode_number_mismatch(self):
        ep = self.parse(GOOD, name="episode-004-Episode 1.txt")
        self.assertEqual(ep["episode_id"], 9)
        self.assertIn("episode_number_mismatch", codes(ep))

    def test_block_before_first_scene(self):
        ep = self.parse(GOOD.replace("SCENE 01", "**ACTION:**\nStart: (00:00.00) | End: (00:01.00)\nstray\n\nSCENE 01"))
        self.assertIn("content_before_first_scene", codes(ep))
        self.assertEqual(ep["n_items"], 3)

    def test_non_utf8_fallback(self):
        data = GOOD.replace("Why did you come?", "Café?").encode("latin-1")
        ep = self.parse(data, mode="wb")
        self.assertIn("encoding_fallback", codes(ep))
        self.assertEqual(ep["n_items"], 3)

    def test_batch_survives_a_broken_file(self):
        write(self.d, "episode-001-Episode a.txt", GOOD.replace("Episode 9", "Episode 1"))
        write(self.d, "episode-002-Episode b.txt", "garbage with no structure")
        write(self.d, "episode-003-Episode c.txt", GOOD.replace("Episode 9", "Episode 3"))
        doc = sp.parse_directory(self.d)
        self.assertEqual([e["episode_id"] for e in doc["episodes"]], [1, 2, 3])
        self.assertEqual([e["parse_status"] for e in doc["episodes"]], ["ok", "failed", "ok"])
        self.assertEqual(doc["summary"]["episode_status"], {"ok": 2, "failed": 1})

    def test_duplicate_episode_id(self):
        write(self.d, "episode-001-Episode a.txt", GOOD.replace("Episode 9", "Episode 1"))
        write(self.d, "episode-002-Episode b.txt", GOOD.replace("Episode 9", "Episode 1"))
        doc = sp.parse_directory(self.d)
        self.assertIn("duplicate_episode_id", codes(doc["episodes"][1]))

    def test_canonical_name_conservative(self):
        txt = GOOD.replace("LADY ANNA (V.O.) (Shaking)", "ANNA (Shaking)") + ""
        write(self.d, "episode-009-Episode a.txt", txt)
        write(self.d, "episode-010-Episode b.txt", GOOD.replace("Episode 9", "Episode 10"))
        doc = sp.parse_directory(self.d)
        canon = {(i["character"], i["character_canon"]) for e in doc["episodes"]
                 for s in e["scenes"] for i in s["items"] if i["kind"] == "dialogue"}
        self.assertEqual(canon, {("ANNA", "ANNA"), ("LADY ANNA", "ANNA")})


@unittest.skipUnless(os.path.isdir(PACK) and any(f.endswith(".txt") for f in os.listdir(PACK)), "needs the real pack in data/screenplay_input/")
class RealPackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = sp.parse_directory(PACK)
        cls.eps = {e["episode_id"]: e for e in cls.doc["episodes"]}

    def test_totals(self):
        s = self.doc["summary"]
        self.assertEqual((s["episodes"], s["scenes"], s["items"]), (50, 100, 3727))
        self.assertEqual(s["issues_by_severity"]["error"], 0)
        self.assertEqual(sorted(self.eps), list(range(1, 51)))

    def test_ep1_known_records(self):
        items = self.eps[1]["scenes"][0]["items"]
        m = next(i for i in items if i["text"] == "You should turn around, Juliana.")
        self.assertEqual((m["character"], m["delivery"], m["start_ts"], m["end_ts"]),
                         ("MATRIARCH", "Cold, Commanding", "00:11.30", "00:13.10"))
        sc = self.eps[1]["scenes"][0]
        self.assertEqual((sc["heading"], sc["start_ts"], sc["end_ts"]),
                         ("INT. CATHEDRAL – DAY", "00:00.00", "02:18.00"))

    def test_multi_scene_episode(self):
        ep = next(e for e in self.doc["episodes"] if e["n_scenes"] >= 4)
        starts = [s["start_sec"] for s in ep["scenes"]]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(ep["scenes"][0]["start_ts"], "00:00.00")

    def test_every_item_has_timestamps(self):
        for e in self.doc["episodes"]:
            for s in e["scenes"]:
                for i in s["items"]:
                    self.assertIsNotNone(i["start_ts"], i["item_id"])
                    self.assertIsNotNone(i["end_sec"], i["item_id"])

    def test_voice_modes(self):
        vm = {i["voice_mode"] for e in self.doc["episodes"] for s in e["scenes"]
              for i in s["items"] if i["kind"] == "dialogue"}
        self.assertEqual(vm, {None, "O.S.", "V.O."})


if __name__ == "__main__":
    unittest.main(verbosity=1)

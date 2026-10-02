"""Python ブリッジのテスト（unittest のみ。libmyln が必要: MYLN_LIB か build/ を参照）。"""
import math
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bridge" / "python"))
from myln import MylnCascade, MylnError, MylnFrame  # noqa: E402

RANSOM = [0.9, 0.95, 0.8, 0.99, 0.85]
IDLE = [0.0, 0.05, 0.01, 0.0, 0.1]
SCAN = [0.2, 0.2, 0.9, 0.1, 0.2]


class FrameTests(unittest.TestCase):
    def test_security_anchors(self):
        for size in ("SS", "T", "S"):
            with MylnFrame(size=size).tune_security() as f:
                self.assertEqual(f.predict(IDLE), "SAFE")
                self.assertEqual(f.predict(SCAN), "MEDIUM")
                self.assertEqual(f.predict(RANSOM), "CRITICAL")

    def test_probabilities_sum_to_one(self):
        with MylnFrame().tune_security() as f:
            probs = f.infer(RANSOM)
            self.assertEqual(len(probs), 5)
            self.assertAlmostEqual(sum(probs), 1.0, places=4)

    def test_sharpness_keeps_class_raises_confidence(self):
        with MylnFrame().tune_security() as a, MylnFrame().tune_security(sharpness=3.0) as b:
            la, ca = a.predict_with_score(RANSOM)
            lb, cb = b.predict_with_score(RANSOM)
            self.assertEqual(la, lb)
            self.assertGreater(cb, ca)

    def test_earthquake(self):
        with MylnFrame("SS").tune_earthquake() as f:
            self.assertEqual(f.predict([0, 0, 0, 0, 0]), "SAFE")
            self.assertEqual(f.predict([1, 1, 1, 1, 1]), "CRITICAL")
            self.assertNotEqual(f.predict([1 / 7, 3 / 9, 0.94, 0, 0.1]), "CRITICAL")

    def test_bad_input_raises(self):
        with MylnFrame().tune_security() as f:
            with self.assertRaises(ValueError):
                f.infer([0.0, math.nan, 0.0, 0.0, 0.0])
            with self.assertRaises(ValueError):
                f.infer([])
            with self.assertRaises(MylnError):
                f.infer([0.1, 0.2, 0.3])          # 次元違い
            with self.assertRaises(MylnError):
                f.tune_security(in_dim=3)
            # 失敗後も使える
            self.assertEqual(f.predict(RANSOM), "CRITICAL")

    def test_bad_construction(self):
        with self.assertRaises(MylnError):
            MylnFrame(size="XL")

    def test_close(self):
        f = MylnFrame()
        f.close()
        f.close()                                  # 二重 close は無害
        with self.assertRaises(MylnError):
            f.infer(RANSOM)


class CascadeTests(unittest.TestCase):
    def test_exact_is_default_and_matches_frame(self):
        with MylnCascade().tune_security() as c, MylnFrame("T").tune_security() as f:
            for x in (IDLE, SCAN, RANSOM):
                label, conf, relay = c.predict_with_path(x)
                self.assertFalse(relay)
                self.assertEqual(c.infer(x)[0], f.infer(x))
            self.assertEqual(c.relay_rate, 0.0)

    def test_heuristic_paths(self):
        with MylnCascade(threshold=0.8, exact=False).tune_security() as c:
            label, conf, relay = c.predict_with_path(IDLE)
            self.assertEqual((label, relay), ("SAFE", True))
            label, conf, relay = c.predict_with_path(RANSOM)
            self.assertEqual((label, relay), ("CRITICAL", True))
            label, conf, relay = c.predict_with_path(SCAN)
            self.assertEqual((label, relay), ("MEDIUM", False))
            self.assertAlmostEqual(c.relay_rate, 2 / 3, places=5)

    def test_bad_threshold(self):
        with self.assertRaises(MylnError):
            MylnCascade(threshold=2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

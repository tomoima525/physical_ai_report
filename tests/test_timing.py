import unittest
from pickup.timing import finish_inference_cycle, held_above_timed

class InferenceTimingTests(unittest.TestCase):
    def test_isolated_miss_rebases_and_on_time_clears_streak(self):
        base,streak=finish_inference_cycle(1.0,1.03,0)
        self.assertEqual((base,streak),(1.03,1))
        base,streak=finish_inference_cycle(base+.05,1.082,streak)
        self.assertAlmostEqual(base,1.08)
        self.assertEqual(streak,0)

    def test_large_or_three_consecutive_misses_stop(self):
        for deadline,now,streak in [(1,1.101,0),(1,1.02,2)]:
            with self.assertRaises(RuntimeError):
                finish_inference_cycle(deadline,now,streak)
        with self.assertRaises(RuntimeError):
            finish_inference_cycle(1,1.02,0,strict=True)

    def test_exact_tolerance_boundary_is_allowed(self):
        self.assertEqual(finish_inference_cycle(1000.,1000.015,1),(1000.,0))

    def test_hold_requires_elapsed_second_not_sample_count(self):
        self.assertFalse(held_above_timed([i*.04 for i in range(20)],[.061]*20,.02))
        self.assertTrue(held_above_timed([i*.06 for i in range(20)],[.061]*20,.02))

    def test_drop_or_missing_coverage_does_not_pass(self):
        ts=[i*.05 for i in range(25)]
        heights=[.061]*25
        heights[-1]=.02
        self.assertFalse(held_above_timed(ts,heights,.02))
        self.assertFalse(held_above_timed([0,.5,1],[.061]*3,.02))
        with self.assertRaises(ValueError):
            held_above_timed([0,0],[.061,.061],.02)

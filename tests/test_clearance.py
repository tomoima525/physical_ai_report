import math
import unittest
from pickup.clearance import minimum_trig

class AngularClearanceTests(unittest.TestCase):
    def test_worst_case_can_be_between_endpoints(self):
        low,angle=minimum_trig(1,0,-2,-.35,.175)
        self.assertAlmostEqual(low,-1)
        self.assertAlmostEqual(angle,0)
        self.assertLess(low,min(1-2*math.cos(-.35),1-2*math.cos(.175)))

    def test_continuous_min_bounds_dense_sample(self):
        for a,b,c,lo,hi in [(0,2,-3,-.35,.175),(2,-4,1,-3,2),(0,0,0,-1,1)]:
            actual,_=minimum_trig(a,b,c,lo,hi)
            sampled=min(a+b*math.sin(lo+(hi-lo)*i/10000)+c*math.cos(lo+(hi-lo)*i/10000) for i in range(10001))
            self.assertLessEqual(actual,sampled+1e-12)
            self.assertAlmostEqual(actual,sampled,places=6)

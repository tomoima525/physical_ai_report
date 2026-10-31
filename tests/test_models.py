import unittest
import xml.etree.ElementTree as ET
from pickup.models import cube_sdf

class CubeModelTests(unittest.TestCase):
    def cfg(self,size):
        return {'cube_size':size,'cube_mass':.02,'object_name':'pickup_cube','spawn_world_xyz':[.09,0,1.04]}

    def test_size_changes_collision_and_visual_and_inertia(self):
        for size in [.025,.04]:
            root=ET.fromstring(cube_sdf(self.cfg(size)))
            collision=[float(v) for v in root.find('.//collision/geometry/box/size').text.split()]
            visual=[float(v) for v in root.find('.//visual/geometry/box/size').text.split()]
            self.assertEqual(collision,[size]*3)
            self.assertEqual(visual,collision)
            self.assertAlmostEqual(float(root.find('.//ixx').text),.02*size**2/6)

    def test_invalid_size_rejected(self):
        for size in [0,-.01,float('nan')]:
            with self.assertRaises(ValueError):
                cube_sdf(self.cfg(size))

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from pickup.core import JOINTS, ordered, held_above, validate_frame

spec = importlib.util.spec_from_file_location('converter', 'scripts/pickup_dataset.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)

class ContractTests(unittest.TestCase):
    def frame(self, t=0):
        return {'timestamp': t,'joint_names': list(JOINTS),'object_frame':'crane_plus_base',
            'observation.state':[0.]*5,'observation.environment_state':[.1,0.,.02],'action':[0.]*5}

    def test_shuffled_joint_order(self):
        self.assertEqual(ordered(list(reversed(JOINTS)), [4,3,2,1,0]), [0,1,2,3,4])
        with self.assertRaises(KeyError):
            ordered(['wrong'], [0.])

    def test_physical_hold_requires_full_second(self):
        self.assertFalse(held_above([.1]*19, .02))
        self.assertTrue(held_above([.1]*20, .02))
        self.assertFalse(held_above([.1]*19+[.02], .02))

    def test_four_cm_threshold_and_explicit_legacy_five_cm(self):
        self.assertTrue(held_above([.061]*20, .02))
        self.assertFalse(held_above([.059]*20, .02))
        self.assertFalse(held_above([.061]*20, .02, lift=.05))

    def test_invalid_data(self):
        frame = self.frame()
        frame['action'][0] = float('nan')
        with self.assertRaises(ValueError):
            validate_frame(frame)

    def test_sampling_tolerance_boundary_at_large_sim_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'result.json').write_text(json.dumps({'success':True,'fps':20}))
            for delta, valid in [(.065, True), (.0651, False)]:
                (path/'frames.jsonl').write_text('\n'.join(json.dumps(self.frame(t)) for t in [1000.,1000.+delta]))
                if valid:
                    self.assertEqual(len(converter.read_episode(path)),2)
                else:
                    with self.assertRaises(ValueError):
                        converter.read_episode(path)

    def test_explicit_exclusion_preserves_invalid_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, times in [('good',[0,.05]),('bad',[0,.2])]:
                ep = root/name
                ep.mkdir()
                (ep/'result.json').write_text(json.dumps({'success':True,'fps':20}))
                (ep/'frames.jsonl').write_text('\n'.join(json.dumps(self.frame(t)) for t in times))
            before = (root/'bad'/'frames.jsonl').read_bytes()
            with self.assertRaises(ValueError):
                converter.select_episodes(root)
            chosen = converter.select_episodes(root,['bad'])
            self.assertEqual([p.name for p,_ in chosen],['good'])
            self.assertEqual((root/'bad'/'frames.jsonl').read_bytes(),before)
            with self.assertRaises(ValueError):
                converter.select_episodes(root,['typo'])
            with self.assertRaises(ValueError):
                converter.select_episodes(root,['good','bad'])

    def test_failed_and_irregular_episodes_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'result.json').write_text(json.dumps({'success':False,'fps':20}))
            self.assertEqual(converter.read_episode(path), [])
            (path/'result.json').write_text(json.dumps({'success':True,'fps':20}))
            (path/'frames.jsonl').write_text('\n'.join(json.dumps(self.frame(t)) for t in [0,.05,.1]))
            self.assertEqual(len(converter.read_episode(path)), 3)
            (path/'frames.jsonl').write_text('\n'.join(json.dumps(self.frame(t)) for t in [0,.2]))
            with self.assertRaises(ValueError):
                converter.read_episode(path)

if __name__ == '__main__':
    unittest.main()

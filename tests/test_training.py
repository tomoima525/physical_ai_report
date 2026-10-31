import unittest
from pickup.training import split_frame_indices

class SplitTests(unittest.TestCase):
    def setUp(self):
        self.episodes=[{'episode_index':i,'length':n} for i,n in enumerate([3,2,4])]

    def test_episode_boundaries_and_noncontiguous_selection(self):
        result=split_frame_indices(self.episodes,{'train_episodes':[0,2],'validation_episodes':[1]})
        self.assertEqual(result['train'],[0,1,2,5,6,7,8])
        self.assertEqual(result['validation'],[3,4])
        self.assertFalse(set(result['train']) & set(result['validation']))

    def test_invalid_splits_rejected(self):
        for train,val in [([0,1],[1,2]),([0,0],[1,2]),([],[0,1,2]),([0],[3]),([0],[1])]:
            with self.assertRaises(ValueError):
                split_frame_indices(self.episodes,{'train_episodes':train,'validation_episodes':val})

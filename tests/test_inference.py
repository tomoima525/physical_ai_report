import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from pickup.inference import configure_action_steps


class ActionStepsTests(unittest.TestCase):
    def policy(self):
        return SimpleNamespace(config=SimpleNamespace(n_action_steps=1,chunk_size=16,
            temporal_ensemble_coeff=None),reset=Mock())

    def test_default_preserves_saved_setting(self):
        p=self.policy()
        self.assertEqual(configure_action_steps(p)['n_action_steps'],1)

    def test_override_and_reset(self):
        p=self.policy()
        info=configure_action_steps(p,8)
        self.assertEqual(info,{'saved_n_action_steps':1,'n_action_steps':8,'chunk_size':16})
        self.assertEqual(p.config.n_action_steps,8)
        p.reset.assert_called_once()

    def test_invalid_overrides_rejected_without_mutation(self):
        for n in [0,-1,17,1.5,True]:
            p=self.policy()
            with self.assertRaises(ValueError):
                configure_action_steps(p,n)
            self.assertEqual(p.config.n_action_steps,1)
            p.reset.assert_not_called()

    def test_temporal_ensemble_rejected(self):
        p=self.policy();p.config.temporal_ensemble_coeff=.01
        with self.assertRaises(ValueError):
            configure_action_steps(p,8)


class ElapsedInputTests(unittest.TestCase):
    def test_append_relative_seconds_without_mutation(self):
        from pickup.inference import environment_with_elapsed
        xyz=[.1,.02,.03]
        self.assertEqual(environment_with_elapsed(xyz,0),[.1,.02,.03,0])
        self.assertEqual(environment_with_elapsed(xyz,100.4-100.)[-1],100.4-100.)
        self.assertEqual(len(xyz),3)

    def test_invalid_elapsed_rejected(self):
        from pickup.inference import environment_with_elapsed
        for value in [-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                environment_with_elapsed([0,0,0],value)


class ObservationRefreshTests(unittest.TestCase):
    def test_pending_callbacks_serviced_before_ready(self):
        from pickup.inference import refresh_observations
        node=SimpleNamespace(wait=Mock(),ready=Mock())
        self.assertFalse(refresh_observations(node))
        node.wait.assert_called_once_with(.005)

    def test_transient_stale_rechecked_after_callbacks(self):
        from pickup.inference import refresh_observations
        node=SimpleNamespace(wait=Mock(),ready=Mock(side_effect=[RuntimeError('Missing/stale joint states'),None]))
        self.assertTrue(refresh_observations(node))
        self.assertEqual([x.args[0] for x in node.wait.call_args_list],[.005,.05])
        self.assertEqual(node.ready.call_count,2)

    def test_persistent_stale_still_stops(self):
        from pickup.inference import refresh_observations
        node=SimpleNamespace(wait=Mock(),ready=Mock(side_effect=RuntimeError('Missing/stale joint states')))
        with self.assertRaises(RuntimeError):
            refresh_observations(node)
        self.assertEqual(node.ready.call_count,2)

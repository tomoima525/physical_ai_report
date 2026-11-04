import subprocess
import unittest
from unittest.mock import patch
from pickup.gazebo import ign_service


class GazeboServiceTests(unittest.TestCase):
    @patch('pickup.gazebo.time.sleep')
    @patch('pickup.gazebo.subprocess.run')
    def test_pose_timeout_then_success_reuses_request(self, run, sleep):
        run.side_effect = [subprocess.CompletedProcess([], 0, '', 'Service call timed out'),
                           subprocess.CompletedProcess([], 0, 'data: true', '')]
        ign_service('default', 'set_pose', 'Pose', 'same pose')
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0], run.call_args_list[1])

    @patch('pickup.gazebo.time.sleep')
    @patch('pickup.gazebo.subprocess.run')
    def test_pose_process_timeout_is_bounded(self, run, sleep):
        run.side_effect = subprocess.TimeoutExpired('ign', 8)
        with self.assertRaises(RuntimeError):
            ign_service('default', 'set_pose', 'Pose', 'pose')
        self.assertEqual(run.call_count, 3)

    @patch('pickup.gazebo.subprocess.run')
    def test_create_timeout_not_retried(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, '', 'Service call timed out')
        with self.assertRaises(RuntimeError):
            ign_service('default', 'create', 'Factory', 'model')
        self.assertEqual(run.call_count, 1)

    @patch('pickup.gazebo.subprocess.run')
    def test_explicit_failure_not_retried(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, 'data: false', '')
        with self.assertRaises(RuntimeError):
            ign_service('default', 'set_pose', 'Pose', 'pose')
        self.assertEqual(run.call_count, 1)

"""Regression checks for actual policy keys and target-fixture trace metrics."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from robocasa_eval.main import STATE_KEYS, _capture_trace_state


class TraceTests(unittest.TestCase):
    def make_env(self, raw_obs, drawer=None):
        raw_env = SimpleNamespace(_get_observations=Mock(return_value=raw_obs))
        if drawer is not None:
            raw_env.drawer = drawer
        return SimpleNamespace(unwrapped=SimpleNamespace(env=raw_env)), raw_env

    def observations(self):
        sizes = (3, 4, 3, 4, 2)
        return {key: np.arange(size, dtype=np.float32) for key, size in zip(STATE_KEYS, sizes)}

    def test_policy_keys_and_target_fixture_are_separate_from_object(self):
        drawer = SimpleNamespace(name="target_fixture", get_door_state=Mock(return_value={"door": np.float64(0.36)}))
        env, raw_env = self.make_env({"drawer_obj_pos": np.array([3., 2., 1.]), "drawer_obj_to_robot0_eef_pos": np.zeros(3)}, drawer)
        trace = _capture_trace_state(env, self.observations())
        self.assertEqual(set(trace["policy_observation"]), set(STATE_KEYS))
        self.assertEqual(sum(len(v) for v in trace["policy_observation"].values()), 16)
        self.assertEqual(trace["target_drawer"], {"fixture_name": "target_fixture", "door_state": {"door": 0.36}})
        self.assertEqual(trace["simulator_observation"]["drawer_obj_pos"], [3., 2., 1.])
        self.assertEqual(trace["missing_simulator_keys"], [])
        raw_env._get_observations.assert_called_once_with(force_update=False)
        drawer.get_door_state.assert_called_once_with(env=raw_env)
        json.dumps(trace, allow_nan=False)

    def test_missing_raw_fields_are_reported_without_losing_policy_state(self):
        env, _ = self.make_env({})
        trace = _capture_trace_state(env, self.observations())
        self.assertEqual(len(trace["policy_observation"]), 5)
        self.assertEqual(trace["simulator_observation"], {})
        self.assertEqual(trace["missing_simulator_keys"], ["drawer_obj_pos", "drawer_obj_to_robot0_eef_pos"])
        self.assertIsNone(trace["target_drawer"])

    def test_missing_required_policy_key_is_not_silently_ignored(self):
        env, _ = self.make_env({})
        obs = self.observations()
        del obs[STATE_KEYS[0]]
        with self.assertRaises(KeyError):
            _capture_trace_state(env, obs)

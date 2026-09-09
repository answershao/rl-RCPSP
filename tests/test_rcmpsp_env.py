import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from src.core.rcmpsp import Activity, Instance, generate_schedule, parse_rcmp, validate_schedule
from src.environments.rcmpsp_env import (
    INVALID_ACTION_PENALTY,
    RCMPSPEnv,
)
from test import TEST_INSTANCE


class RcmpspEnvTest(unittest.TestCase):
    def test_dynamic_features_preview_eligible_activity_placement(self) -> None:
        first = (1, 1)
        second = (1, 2)
        sink = (1, 3)
        instance = Instance(
            name="dynamic-features",
            capacities=(1,),
            activities={
                first: Activity(first, 2, (1,), (sink,)),
                second: Activity(second, 2, (1,), (sink,)),
                sink: Activity(sink, 1, (0,), ()),
            },
            predecessors={first: (), second: (), sink: (first, second)},
        )
        env = RCMPSPEnv(instance)
        observation, _ = env.reset()
        dynamic = observation["dynamic_activity_features"]
        np.testing.assert_array_equal(observation["resource_profile"], 0.0)
        np.testing.assert_allclose(
            dynamic[env.activity_index[first]],
            [0.0, 0.0, 0.4, 0.0, 0.4, 0.6],
        )
        np.testing.assert_array_equal(dynamic[env.activity_index[sink]], 0.0)

        observation, _, _, _, info = env.step(env.activity_index[first])
        dynamic = observation["dynamic_activity_features"]
        self.assertGreater(float(observation["resource_profile"].max()), 0.0)
        self.assertLessEqual(float(observation["resource_profile"].max()), 1.0)
        np.testing.assert_allclose(
            dynamic[env.activity_index[second]],
            [0.0, 0.4, 0.8, 0.4, 0.4, 1.0],
        )
        self.assertEqual(info["start"], 0)
        self.assertEqual(info["finish"], 2)
        np.testing.assert_array_equal(dynamic[env.activity_index[first]], 0.0)
        np.testing.assert_array_equal(dynamic[env.activity_index[sink]], 0.0)

        invalid_observation, invalid_reward, terminated, truncated, invalid_info = env.step(
            env.activity_index[sink]
        )
        self.assertEqual(invalid_reward, INVALID_ACTION_PENALTY)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(invalid_info["invalid_action"])
        np.testing.assert_allclose(
            invalid_observation["resource_profile"], observation["resource_profile"]
        )

    def test_gymnasium_interface(self) -> None:
        check_env(RCMPSPEnv(TEST_INSTANCE), skip_render_check=True)

    def test_random_episode_has_legal_schedule_and_makespan_reward(self) -> None:
        env = RCMPSPEnv(TEST_INSTANCE)
        observation, info = env.reset(seed=11)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertIn("eligible_mask", info)

        terminated = False
        total_reward = 0.0
        steps = 0
        rng = np.random.default_rng(11)
        while not terminated:
            eligible = np.flatnonzero(observation["eligible_mask"])
            action = int(rng.choice(eligible))
            observation, reward, terminated, truncated, terminal_info = env.step(action)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertFalse(truncated)
            total_reward += reward
            steps += 1

        schedule = env.schedule
        validate_schedule(env.instance, schedule)
        self.assertEqual(steps, len(env.instance.activities))
        expected_return = -schedule.makespan / env.horizon
        self.assertAlmostEqual(total_reward, expected_return)
        self.assertTrue(np.all(observation["activity_status"] == 2))
        self.assertEqual(terminal_info["makespan"], schedule.makespan)
        self.assertEqual(terminal_info["normalized_makespan"], schedule.makespan / env.horizon)
        self.assertGreaterEqual(terminal_info["resource_utilization"], 0.0)
        self.assertLessEqual(terminal_info["resource_utilization"], 1.0)
        self.assertEqual(terminal_info["activity_count"], env.activity_count)
        self.assertAlmostEqual(terminal_info["episode_makespan_penalty"], -schedule.makespan / env.horizon)
        self.assertAlmostEqual(terminal_info["episode_reward"], total_reward)

    def test_incremental_decoder_matches_batch_ssgs(self) -> None:
        instance = parse_rcmp(TEST_INSTANCE)
        env = RCMPSPEnv(instance)
        env.reset(seed=23)
        priorities = np.random.default_rng(23).uniform(-1.0, 1.0, env.activity_count)

        terminated = False
        while not terminated:
            eligible = np.flatnonzero(env._state.eligible_mask)
            action = int(eligible[np.argmax(priorities[eligible])])
            _, _, terminated, _, _ = env.step(action)

        priority_map = {
            activity_id: float(priorities[index])
            for index, activity_id in enumerate(env.activity_ids)
        }
        self.assertEqual(env.schedule, generate_schedule(instance, priority_map))


if __name__ == "__main__":
    unittest.main()

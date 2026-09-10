import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from gymnasium.utils.env_checker import check_env

from src.core.rcpsp import Activity, Instance, generate_schedule, validate_schedule
from src.core.rules import rule_makespan
from src.data.adapter import load_core_instance
from src.envs.observation import ObservationLayout, RESOURCE_PROFILE_BIN_COUNT
from src.envs.rcpsp_env import (
    INVALID_ACTION_PENALTY,
    TIME_SCALE_RULE,
    RCPSPEnv,
)
from src.envs.sb3_env import FlattenRCPSPObservation
from src.training.callbacks import TERMINAL_METRICS
from tests import TEST_INSTANCE


def parallel_instance() -> Instance:
    """Three unit-demand activities that all fit at once in a capacity-3 bin.

    Its serial SGS makespan (4) is three times smaller than the duration sum
    (12), which is exactly the gap ``time_scale`` is meant to close.
    """
    activities = {
        0: Activity(0, 4, (1,), ()),
        1: Activity(1, 4, (1,), ()),
        2: Activity(2, 4, (1,), ()),
    }
    return Instance(
        name="parallel",
        capacities=(3,),
        activities=activities,
        predecessors={0: (), 1: (), 2: ()},
    )


class RcpspEnvTest(unittest.TestCase):
    def test_dynamic_features_preview_eligible_activity_placement(self) -> None:
        first = 1
        second = 2
        sink = 3
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
        env = RCPSPEnv(instance)
        observation, _ = env.reset()
        dynamic = observation["dynamic_activity_features"]
        np.testing.assert_array_equal(observation["resource_profile"], 0.0)
        # The 5th column is makespan_increment / max(duration) = 2 / 2, not
        # / time_scale (which would be 2/5 here) -- see RCPSPEnv._max_duration.
        np.testing.assert_allclose(
            dynamic[env.activity_index[first]],
            [0.0, 0.0, 0.4, 0.0, 1.0, 0.6],
        )
        np.testing.assert_array_equal(dynamic[env.activity_index[sink]], 0.0)

        observation, _, _, _, info = env.step(env.activity_index[first])
        dynamic = observation["dynamic_activity_features"]
        self.assertGreater(float(observation["resource_profile"].max()), 0.0)
        self.assertLessEqual(float(observation["resource_profile"].max()), 1.0)
        np.testing.assert_allclose(
            dynamic[env.activity_index[second]],
            [0.0, 0.4, 0.8, 0.4, 1.0, 1.0],
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
        check_env(RCPSPEnv(TEST_INSTANCE), skip_render_check=True)

    def test_time_scale_is_the_lst_reference_makespan_not_the_duration_sum(self) -> None:
        instance = parallel_instance()
        env = RCPSPEnv(instance)
        self.assertEqual(env.horizon, 12)
        self.assertEqual(env.time_scale, 4)
        self.assertEqual(env.time_scale, rule_makespan(instance, TIME_SCALE_RULE))
        self.assertLessEqual(env.time_scale, env.horizon)

        observation, _ = env.reset()
        dynamic = observation["dynamic_activity_features"]
        # ready/start = 0, finish = 4 units = the whole reference, no resource
        # wait, the insertion adds 4 units and each downstream path is 4 units.
        # Under the previous /horizon denominator these would be [0, 0, 1/3,
        # 0, 1/3, 1/3] -- a third of the declared range.
        np.testing.assert_allclose(dynamic[0], [0.0, 0.0, 1.0, 0.0, 1.0, 1.0])

    def test_profile_bins_tile_the_usage_horizon_without_gaps(self) -> None:
        """The profile grid must stay gap-free after moving onto time_scale.

        Bin edges are still derived so that bin i ends where bin i+1 starts;
        the final bin is widened to ``horizon`` because a policy may overrun
        the reference and those insertions must not fall outside the grid.
        """
        for instance in (parallel_instance(), load_core_instance(TEST_INSTANCE)):
            env = RCPSPEnv(instance)
            ranges = env._resource_profile_ranges
            self.assertEqual(len(ranges), RESOURCE_PROFILE_BIN_COUNT)
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], env.horizon)
            covered = np.zeros(env.horizon, dtype=bool)
            for _, (start, stop) in enumerate(ranges):
                covered[start:stop] = True
            self.assertTrue(covered.all())

    def test_time_features_saturate_instead_of_leaving_the_declared_range(self) -> None:
        env = FlattenRCPSPObservation(RCPSPEnv(load_core_instance(TEST_INSTANCE)))
        # Force the reference far below any achievable makespan so the overflow
        # path (a policy that overruns the LST reference) is exercised directly.
        env.env.time_scale = 1
        observation, _ = env.reset(seed=5)
        self.assertTrue(env.observation_space.contains(observation))

        layout = ObservationLayout(env.env.activity_count, env.env.resource_count)
        rng = np.random.default_rng(5)
        terminated = False
        peak = 0.0
        while not terminated:
            eligible = np.flatnonzero(env.env._state.eligible_mask)
            observation, _, terminated, truncated, _ = env.step(int(rng.choice(eligible)))
            self.assertTrue(env.observation_space.contains(observation))
            self.assertFalse(truncated)
            peak = max(peak, float(observation[layout.current_time]))
        # current_time / time_scale would be ~40 without the clip.
        self.assertLessEqual(peak, 1.0)
        self.assertGreater(env.env.schedule.makespan, 1)

    def test_random_episode_has_legal_schedule_and_makespan_reward(self) -> None:
        env = RCPSPEnv(TEST_INSTANCE)
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
        # Reward and the episode metrics are scaled by time_scale (not horizon),
        # which is what keeps the gamma=1 return exactly -makespan/time_scale.
        expected_return = -schedule.makespan / env.time_scale
        self.assertAlmostEqual(total_reward, expected_return)
        self.assertTrue(np.all(observation["activity_status"] == 2))
        self.assertEqual(terminal_info["makespan"], schedule.makespan)
        self.assertEqual(terminal_info["normalized_makespan"], schedule.makespan / env.time_scale)
        self.assertGreaterEqual(terminal_info["resource_utilization"], 0.0)
        self.assertLessEqual(terminal_info["resource_utilization"], 1.0)
        self.assertEqual(terminal_info["activity_count"], env.activity_count)
        self.assertAlmostEqual(terminal_info["episode_makespan_penalty"],
                               -schedule.makespan / env.time_scale)
        self.assertAlmostEqual(terminal_info["episode_reward"], total_reward)
        self.assertLessEqual(abs(expected_return), 3.0)

    def test_rejected_actions_are_zero_reward_and_bounded_by_the_step_budget(self) -> None:
        """The invalid-action path must not be able to hurt the value function.

        It used to return -1.0, which dwarfs a whole episode return (~ -1), and
        nothing bounded an episode that kept proposing rejected actions.
        """
        self.assertEqual(INVALID_ACTION_PENALTY, 0.0)
        env = RCPSPEnv(load_core_instance(TEST_INSTANCE))
        observation, _ = env.reset(seed=37)
        blocked = int(np.flatnonzero(~observation["eligible_mask"])[0])
        total_reward = 0.0
        steps = 0
        while True:
            observation, reward, terminated, truncated, info = env.step(blocked)
            total_reward += reward
            steps += 1
            self.assertEqual(reward, 0.0)
            self.assertTrue(info["invalid_action"])
            self.assertFalse(terminated)
            if truncated:
                break
        self.assertEqual(steps, env._step_budget)
        self.assertEqual(total_reward, 0.0)
        self.assertTrue(info["aborted"])
        # Monitor persists TERMINAL_METRICS by direct dict lookup, so a
        # truncated episode has to carry them or SB3 raises mid-training.
        for key in TERMINAL_METRICS:
            self.assertIn(key, info)
        with self.assertRaises(RuntimeError):
            env.step(blocked)

    def test_critical_path_shaping_preserves_makespan_objective(self) -> None:
        shaping = 0.5
        env = RCPSPEnv(TEST_INSTANCE, reward_shaping_coef=shaping)
        observation, _ = env.reset(seed=13)
        total_reward = 0.0
        terminated = False
        rng = np.random.default_rng(13)
        while not terminated:
            eligible = np.flatnonzero(observation["eligible_mask"])
            observation, reward, terminated, _, terminal_info = env.step(
                int(rng.choice(eligible))
            )
            total_reward += reward

        expected = -(1.0 + shaping) * env.schedule.makespan / env.time_scale
        self.assertAlmostEqual(total_reward, expected)
        self.assertAlmostEqual(terminal_info["episode_makespan_penalty"],
                               -env.schedule.makespan / env.time_scale)
        self.assertAlmostEqual(terminal_info["episode_critical_path_penalty"],
                               -env.schedule.makespan / env.time_scale)

    def test_incremental_decoder_matches_batch_ssgs(self) -> None:
        instance = load_core_instance(TEST_INSTANCE)
        env = RCPSPEnv(instance)
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

    def test_incremental_resource_profile_matches_full_recomputation(self) -> None:
        env = RCPSPEnv(load_core_instance(TEST_INSTANCE))
        observation, _ = env.reset(seed=29)
        rng = np.random.default_rng(29)
        terminated = False
        while not terminated:
            eligible = np.flatnonzero(observation["eligible_mask"])
            action = int(rng.choice(eligible))
            observation, _, terminated, _, _ = env.step(action)

            normalized_usage = env._state.usage[: env.horizon].astype(np.float32)
            normalized_usage /= env._capacity_scale[None, :]
            expected = np.zeros_like(observation["resource_profile"])
            expected = expected.reshape(env.resource_count, 2, 16)
            for bin_index, (start, stop) in enumerate(env._resource_profile_ranges):
                values = normalized_usage[start:stop]
                expected[:, 0, bin_index] = values.mean(axis=0)
                expected[:, 1, bin_index] = values.max(axis=0)
            np.testing.assert_allclose(
                observation["resource_profile"].reshape(env.resource_count, 2, 16),
                expected,
                atol=1e-7,
            )


if __name__ == "__main__":
    unittest.main()

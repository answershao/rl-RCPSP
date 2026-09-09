import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.envs.multi_instance import MultiInstanceRCPSPEnv
from src.envs.observation import (
    MAX_SUCCESSORS,
    ObservationLayout,
    build_static_graph_cache,
)
from src.envs.sb3_env import make_sb3_env
from tests import TEST_INSTANCE, TEST_INSTANCE_2


class MultiInstanceTest(unittest.TestCase):
    def test_fixed_padding_and_episode(self):
        env = MultiInstanceRCPSPEnv([TEST_INSTANCE, TEST_INSTANCE_2])
        obs, _ = env.reset(seed=3)
        self.assertTrue(env.observation_space.contains(obs))

    def test_observation_uses_catalog_instance_index(self):
        env = MultiInstanceRCPSPEnv(
            [TEST_INSTANCE],
            instance_indices=[3],
            catalog_size=5,
        )
        observation, _ = env.reset(seed=3)
        layout = ObservationLayout(env.max_activities, env.max_resources)
        self.assertAlmostEqual(float(observation[layout.instance_index]), 0.8)

    def test_static_graph_cache_contains_successor_indices(self):
        env = MultiInstanceRCPSPEnv([TEST_INSTANCE])
        observation, _ = env.reset(seed=3)
        active = env.active_env
        layout = ObservationLayout(env.max_activities, env.max_resources)
        cache = build_static_graph_cache(
            [active.instance],
            max_activities=env.max_activities,
            max_resources=env.max_resources,
        )
        self.assertEqual(cache.instance_names, (active.instance.name,))
        self.assertEqual(int(cache.activity_mask.sum()), active.activity_count)
        activity_id = active.activity_ids[0]
        successors = active.instance.activities[activity_id].successors
        self.assertEqual(cache.successor_indices.shape[-1], MAX_SUCCESSORS)
        expected = [active.activity_index[item] for item in successors]
        np.testing.assert_array_equal(
            cache.successor_indices[0, 0, :len(successors)], expected
        )
        terminated = False
        while not terminated:
            eligible = np.flatnonzero(observation[layout.eligible_mask] > 0.5)
            observation, _, terminated, truncated, _ = env.step(int(eligible[0]))
            self.assertFalse(truncated)
        self.assertTrue(env.observation_space.contains(observation))

    def test_unpadded_encoding_matches_single_instance_wrapper(self):
        multi_env = MultiInstanceRCPSPEnv([TEST_INSTANCE])
        single_env = make_sb3_env(TEST_INSTANCE)
        multi_observation, _ = multi_env.reset(seed=7)
        single_observation, _ = single_env.reset(seed=7)
        np.testing.assert_allclose(multi_observation, single_observation)

    def test_observation_layout_covers_each_feature_once(self):
        layout = ObservationLayout(max_activities=5, max_resources=2)
        fields = (
            layout.activity_status,
            layout.precedence_satisfied,
            layout.eligible_mask,
            layout.dynamic_activity_features,
            layout.remaining_capacity,
            layout.resource_profile,
        )
        self.assertEqual(fields[0].start, 0)
        self.assertTrue(all(first.stop == second.start for first, second in zip(fields, fields[1:])))
        self.assertEqual(fields[-1].stop, layout.current_time)
        self.assertEqual(layout.current_time + 1, layout.instance_index)
        self.assertEqual(layout.instance_index + 1, layout.size)


if __name__ == "__main__":
    unittest.main()

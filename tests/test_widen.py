"""Tests for training on a small padded graph and widening it before evaluation."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.adapter import load_core_instance
from src.envs.observation import build_static_graph_cache
from src.training.environments import make_multi_env
from src.training.ppo import create_ppo, widen_policy
from tests import TEST_INSTANCE, TEST_INSTANCE_2


SMALL_CAP = 32
LARGE_CAP = 48


class WidenPolicyTest(unittest.TestCase):
    """The RCPSP policy must be resizable without retraining or recompiling."""

    def setUp(self):
        self.instances = [
            load_core_instance(TEST_INSTANCE),
            load_core_instance(TEST_INSTANCE_2),
        ]
        self.max_resources = max(
            instance.resource_count for instance in self.instances
        )

    def _build(self, cap: int):
        cache = build_static_graph_cache(
            self.instances, max_activities=cap, max_resources=self.max_resources
        )
        env = make_multi_env(
            [str(TEST_INSTANCE)],
            max_activities=cap,
            max_resources=self.max_resources,
            instance_indices=[0],
            catalog_size=len(self.instances),
        )
        model = create_ppo(
            env,
            instances=self.instances,
            static_cache=cache,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            gin_layers=2,
            seed=1,
            device="cpu",
        )
        model.verbose = 0
        return model, env

    @staticmethod
    def _logits_and_value(model, env):
        observation = env.reset()[0]
        tensor = torch.as_tensor(observation[None, :])
        with torch.no_grad():
            logits = model.policy.get_distribution(tensor).distribution.logits[0]
            value = float(model.policy.predict_values(tensor)[0])
        return logits, value

    def test_every_trainable_parameter_is_activity_count_agnostic(self):
        small, small_env = self._build(SMALL_CAP)
        large, large_env = self._build(LARGE_CAP)
        try:
            small_shapes = {
                name: tuple(value.shape)
                for name, value in small.policy.named_parameters()
            }
            large_shapes = {
                name: tuple(value.shape)
                for name, value in large.policy.named_parameters()
            }
            self.assertTrue(small_shapes)
            self.assertEqual(small_shapes, large_shapes)
        finally:
            small_env.close()
            large_env.close()

    def test_widening_resizes_the_graph_and_preserves_the_function(self):
        small, small_env = self._build(SMALL_CAP)
        try:
            widened = widen_policy(
                small,
                instances=self.instances,
                max_activities=LARGE_CAP,
                max_resources=self.max_resources,
                static_cache=build_static_graph_cache(
                    self.instances,
                    max_activities=LARGE_CAP,
                    max_resources=self.max_resources,
                ),
                device="cpu",
            )
            extractor = widened.policy.features_extractor
            self.assertEqual(extractor.max_activities, LARGE_CAP)
            self.assertEqual(extractor.max_resources, self.max_resources)
            # Widening must not disturb the source model.
            self.assertEqual(
                small.policy.features_extractor.max_activities, SMALL_CAP
            )

            large_env = make_multi_env(
                [str(TEST_INSTANCE)],
                max_activities=LARGE_CAP,
                max_resources=self.max_resources,
                instance_indices=[0],
                catalog_size=len(self.instances),
            )
            try:
                small_logits, small_value = self._logits_and_value(small, small_env)
                large_logits, large_value = self._logits_and_value(widened, large_env)
                # The padded graph must not change the policy for real activities.
                torch.testing.assert_close(
                    large_logits[:SMALL_CAP], small_logits[:SMALL_CAP]
                )
                self.assertAlmostEqual(large_value, small_value, places=6)
            finally:
                large_env.close()
        finally:
            small_env.close()

    def test_policy_is_bit_identical_across_caps_for_a_whole_episode(self):
        """Driving both caps with the same actions must give identical outputs.

        This is the property the training-cap shrink relies on.  Only the action
        sampling RNG depends on the action-space size, never the policy itself.
        """
        small, small_env = self._build(SMALL_CAP)
        try:
            widened = widen_policy(
                small,
                instances=self.instances,
                max_activities=LARGE_CAP,
                max_resources=self.max_resources,
                static_cache=build_static_graph_cache(
                    self.instances,
                    max_activities=LARGE_CAP,
                    max_resources=self.max_resources,
                ),
                device="cpu",
            )
            large_env = make_multi_env(
                [str(TEST_INSTANCE)],
                max_activities=LARGE_CAP,
                max_resources=self.max_resources,
                instance_indices=[0],
                catalog_size=len(self.instances),
            )
            try:
                obs_small = small_env.reset()[0]
                obs_large = large_env.reset()[0]
                steps = 0
                while True:
                    small_tensor = torch.as_tensor(obs_small[None, :])
                    large_tensor = torch.as_tensor(obs_large[None, :])
                    with torch.no_grad():
                        small_dist = small.policy.get_distribution(
                            small_tensor
                        ).distribution
                        large_dist = widened.policy.get_distribution(
                            large_tensor
                        ).distribution
                        small_value = small.policy.predict_values(small_tensor)
                        large_value = widened.policy.predict_values(large_tensor)
                    torch.testing.assert_close(
                        large_dist.logits[0, :SMALL_CAP], small_dist.logits[0]
                    )
                    torch.testing.assert_close(
                        large_dist.probs[0, :SMALL_CAP], small_dist.probs[0]
                    )
                    torch.testing.assert_close(large_value, small_value)
                    action = int(small_dist.probs.argmax())
                    obs_small, _, done_small, _, _ = small_env.step(action)
                    obs_large, _, done_large, _, _ = large_env.step(action)
                    steps += 1
                    self.assertEqual(bool(done_small), bool(done_large))
                    if bool(done_small) or steps > 64:
                        break
                self.assertGreater(steps, 1)
            finally:
                large_env.close()
        finally:
            small_env.close()

    def test_widening_rejects_smaller_caps_and_mismatched_catalogs(self):
        small, small_env = self._build(SMALL_CAP)
        try:
            with self.assertRaisesRegex(ValueError, "cannot widen from"):
                widen_policy(
                    small,
                    instances=self.instances,
                    max_activities=SMALL_CAP - 1,
                    max_resources=self.max_resources,
                    static_cache=build_static_graph_cache(
                        self.instances,
                        max_activities=SMALL_CAP,
                        max_resources=self.max_resources,
                    ),
                    device="cpu",
                )
            with self.assertRaisesRegex(ValueError, "static cache ordering"):
                widen_policy(
                    small,
                    instances=list(reversed(self.instances)),
                    max_activities=LARGE_CAP,
                    max_resources=self.max_resources,
                    static_cache=build_static_graph_cache(
                        self.instances,
                        max_activities=LARGE_CAP,
                        max_resources=self.max_resources,
                    ),
                    device="cpu",
                )
        finally:
            small_env.close()


if __name__ == "__main__":
    unittest.main()

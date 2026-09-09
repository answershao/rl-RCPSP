import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from src.core.rcmpsp import parse_rcmp
from scripts.train_ppo import load_baseline_results
from src.environments.observation import (
    ObservationLayout,
    build_static_graph_cache,
)
from src.environments.multi_instance import make_splits
from src.environments.rcmpsp_env import RCMPSPEnv
from src.environments.sb3_env import make_sb3_env
from src.training.environments import make_multi_env, make_single_env, make_vector_env
from src.training.callbacks import TERMINAL_METRICS, RCMPSPMetricsCallback
from src.training.features import _aggregate_edge_messages
from src.training.ppo import (
    create_ppo,
    evaluate_paths,
    evaluate_paths_sampled,
)
from test import TEST_INSTANCE


class Sb3Test(unittest.TestCase):
    def test_terminal_metrics_keep_only_target_and_diagnostic_values(self):
        self.assertEqual(
            TERMINAL_METRICS,
            ("makespan", "normalized_makespan", "resource_utilization"),
        )

    def test_validation_callback_uses_evaluation_patience_and_saves_best(self):
        gaps = iter((0.10, 0.095))
        callback = RCMPSPMetricsCallback(
            checkpoint_dir="checkpoints",
            early_stop_patience=1,
            validation_interval=2,
            validation_min_delta=0.01,
            validation_evaluator=lambda _model: next(gaps),
        )
        callback.model = Mock()
        callback.model.logger = Mock()

        with TemporaryDirectory() as directory:
            callback.checkpoint_dir = Path(directory)
            callback._on_training_start()
            callback._on_rollout_end()
            callback._on_rollout_start()
            self.assertFalse(callback._stop_requested)
            callback._on_rollout_end()
            callback._on_rollout_start()

        self.assertTrue(callback._stop_requested)
        self.assertEqual(callback.validation_evaluations, 2)
        self.assertAlmostEqual(callback.best_validation_gap, 0.095)
        self.assertEqual(callback.model.save.call_count, 2)
        self.assertFalse(callback._on_step())

    def test_compact_edge_messages_match_successor_slot_aggregation(self):
        embeddings = torch.tensor(
            [[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0]]]
        )
        successors = torch.tensor(
            [[[1, 2, -1], [3, -1, -1], [3, -1, -1], [-1, -1, -1]]]
        )
        valid = successors >= 0
        slot_targets = successors.clamp(min=0)
        flat_targets = slot_targets.reshape(1, -1)
        valid_slots = valid.unsqueeze(-1).to(embeddings.dtype)
        successor_sum = (
            embeddings.gather(1, flat_targets.unsqueeze(-1).expand(-1, -1, 2))
            .reshape(1, 4, 3, 2)
            .mul(valid_slots)
            .sum(dim=2)
        )
        predecessor_sum = torch.zeros_like(embeddings)
        source_messages = (
            embeddings.unsqueeze(2).expand(-1, -1, 3, -1) * valid_slots
        ).reshape(1, -1, 2)
        predecessor_sum.scatter_add_(
            1, flat_targets.unsqueeze(-1).expand(-1, -1, 2), source_messages
        )

        compact_predecessors, compact_successors = _aggregate_edge_messages(
            embeddings,
            torch.tensor([[0, 0, 1, 2]]),
            torch.tensor([[1, 2, 3, 3]]),
            torch.ones((1, 4), dtype=torch.bool),
        )
        torch.testing.assert_close(compact_predecessors, predecessor_sum)
        torch.testing.assert_close(compact_successors, successor_sum)

    def test_shared_baseline_results_are_validated(self):
        with TemporaryDirectory() as directory:
            result_path = Path(directory) / "baselines.csv"
            result_path.write_text(
                "instance,FIFO,Shortest,Random,CP-SAT\n"
                "sample.rcmp,10,11,12,9\n",
                encoding="ascii",
            )
            self.assertEqual(
                load_baseline_results(result_path, ["sample.rcmp"]),
                {
                    "sample.rcmp": {
                        "FIFO": 10,
                        "Shortest": 11,
                        "Random": 12,
                        "CP-SAT": 9,
                    }
                },
            )
            with self.assertRaisesRegex(ValueError, "does not cover"):
                load_baseline_results(result_path, ["missing.rcmp"])

    def test_ppo_rejects_misaligned_static_cache(self):
        base = RCMPSPEnv(TEST_INSTANCE)
        env = make_sb3_env(TEST_INSTANCE)
        cache = build_static_graph_cache(
            [base.instance],
            max_activities=base.activity_count,
            max_resources=base.resource_count,
        )
        misaligned_cache = replace(cache, instance_names=("different-instance",))
        with self.assertRaisesRegex(ValueError, "static cache ordering"):
            create_ppo(
                env,
                instances=[base.instance],
                static_cache=misaligned_cache,
                n_steps=8,
                batch_size=8,
                n_epochs=1,
                seed=1,
                device="cpu",
            )
        env.close()

    def test_training_environment_factories(self):
        single = make_single_env(TEST_INSTANCE)
        observation, _ = single.reset(seed=1)
        self.assertTrue(single.observation_space.contains(observation))

        vector_env = make_vector_env(
            [lambda: make_multi_env([TEST_INSTANCE])],
            backend="dummy",
        )
        try:
            observation = vector_env.reset()
            self.assertTrue(vector_env.observation_space.contains(observation[0]))
        finally:
            vector_env.close()

    def test_sampled_evaluation_uses_prefix_minima_and_restores_cache(self):
        instance_text = "\n".join(
            (
                "1",
                "2",
                "1 1",
                "3",
                "1 1",
                "2 1 1 1 1:3",
                "2 1 1 1 1:3",
                "1 0 0 0",
            )
        ) + "\n"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            training_path = root / "training.rcmp"
            evaluation_path = root / "evaluation.rcmp"
            training_path.write_text(instance_text, encoding="ascii")
            evaluation_path.write_text(instance_text, encoding="ascii")
            instance = parse_rcmp(training_path)
            env = make_sb3_env(training_path)
            model = create_ppo(
                env,
                instances=[instance],
                n_steps=3,
                batch_size=3,
                n_epochs=1,
                gin_layers=1,
                seed=1,
                device="cpu",
            )
            try:
                extractor = model.policy.features_extractor
                reference_env = SimpleNamespace(
                    max_activities=len(instance.activities),
                    max_resources=instance.resource_count,
                )
                training_cache = build_static_graph_cache(
                    [instance],
                    max_activities=reference_env.max_activities,
                    max_resources=reference_env.max_resources,
                    max_successors=extractor.max_successors,
                )
                sampled = evaluate_paths_sampled(
                    model,
                    [str(evaluation_path)],
                    seed=11,
                    reference_env=reference_env,
                    sample_budgets=(1, 2),
                    batch_size=1,
                    restore_cache=training_cache,
                )
                self.assertEqual(set(sampled), {1, 2})
                self.assertLessEqual(sampled[2][0][1], sampled[1][0][1])
                self.assertEqual(extractor.instance_names, ("training",))
                with self.assertRaisesRegex(ValueError, "between 1 and 32"):
                    evaluate_paths_sampled(
                        model,
                        [],
                        seed=11,
                        reference_env=reference_env,
                        sample_budgets=(33,),
                    )
            finally:
                env.close()

    def test_adapter_and_short_learning_run(self):
        base = RCMPSPEnv(TEST_INSTANCE)
        check_env(base, warn=True, skip_render_check=True)
        env = make_sb3_env(TEST_INSTANCE)
        model = create_ppo(
            env, instances=[base.instance], n_steps=8, batch_size=8,
            n_epochs=1, gin_layers=1,
            seed=1, device="cpu",
        )
        self.assertEqual(model.gamma, 0.999)
        self.assertEqual(model.gae_lambda, 0.98)
        self.assertEqual(model.learning_rate, 2e-4)
        self.assertEqual(model.ent_coef, 0.01)
        self.assertEqual(model.vf_coef, 0.5)
        self.assertIsNone(model.target_kl)
        model.learn(total_timesteps=16, progress_bar=False)
        observation, _ = env.reset(seed=2)
        action, _ = model.predict(observation)
        self.assertTrue(env.action_space.contains(action))

        observation_tensor, _ = model.policy.obs_to_tensor(observation)
        distribution = model.policy.get_distribution(observation_tensor)
        probabilities = distribution.distribution.probs.detach().cpu().numpy()[0]
        base_env = env.unwrapped
        layout = ObservationLayout(base_env.activity_count, base_env.resource_count)
        eligible = observation[layout.eligible_mask] > 0.5
        np.testing.assert_array_equal(probabilities[~eligible], 0.0)
        self.assertAlmostEqual(float(probabilities[eligible].sum()), 1.0, places=6)
        self.assertTrue(model.policy.share_features_extractor)
        actor_parameters = {id(item) for item in model.policy.mlp_extractor.actor.parameters()}
        critic_parameters = {id(item) for item in model.policy.mlp_extractor.critic.parameters()}
        self.assertTrue(actor_parameters.isdisjoint(critic_parameters))

        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model.save(model_path)
            restored = PPO.load(model_path, device="cpu")
            restored_action, _ = restored.predict(observation, deterministic=True)
            expected_action, _ = model.predict(observation, deterministic=True)
            np.testing.assert_array_equal(restored_action, expected_action)

        paths = make_splits()["train"][:2]
        reference_env = SimpleNamespace(
            max_activities=base.activity_count,
            max_resources=base.resource_count,
        )
        training_cache = build_static_graph_cache(
            [base.instance],
            max_activities=base.activity_count,
            max_resources=base.resource_count,
        )
        sequential = evaluate_paths(
            model,
            paths,
            seed=11,
            reference_env=reference_env,
            batch_size=1,
            restore_cache=training_cache,
        )
        self.assertEqual(
            model.policy.features_extractor.instance_names,
            training_cache.instance_names,
        )
        batched = evaluate_paths(
            model,
            paths,
            seed=11,
            reference_env=reference_env,
            batch_size=2,
            restore_cache=training_cache,
        )
        self.assertEqual(batched, sequential)
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            evaluate_paths(
                model, paths, seed=11, reference_env=reference_env, batch_size=0
            )


if __name__ == "__main__":
    unittest.main()

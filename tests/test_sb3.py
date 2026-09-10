import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from scripts.train_ppo import load_reference_rules
from src.data.adapter import load_core_instance
from src.envs.observation import (
    ObservationLayout,
    build_static_graph_cache,
)
from src.envs.rcpsp_env import RCPSPEnv
from src.envs.sb3_env import make_sb3_env
from src.training.environments import make_multi_env, make_single_env, make_vector_env
from src.training.callbacks import TERMINAL_METRICS, RCPSPMetricsCallback
from src.training.features import (
    _aggregate_compact_edge_messages,
    _aggregate_edge_messages,
    _compact_edge_indices,
    _edge_degrees,
)
from src.data.instances import instance_id, loader_for, read_protocol
from src.training.ppo import (
    create_ppo,
    evaluate_paths,
    evaluate_paths_sampled,
)
from tests import TEST_INSTANCE, TEST_INSTANCE_2


class Sb3Test(unittest.TestCase):
    def test_terminal_metrics_keep_only_target_and_diagnostic_values(self):
        self.assertEqual(
            TERMINAL_METRICS,
            ("makespan", "normalized_makespan", "resource_utilization"),
        )

    def test_validation_callback_uses_evaluation_patience_and_saves_best(self):
        gaps = iter((0.10, 0.095))
        callback = RCPSPMetricsCallback(
            checkpoint_dir="checkpoints",
            early_stop_patience=1,
            validation_interval=2,
            validation_min_delta=0.01,
            validation_evaluator=lambda _model: next(gaps),
            reference_rule_name="serial_LST",
        )
        callback.model = Mock()
        callback.model.logger = Mock()

        with TemporaryDirectory() as directory:
            callback.checkpoint_dir = Path(directory)
            callback._on_training_start()
            callback._on_rollout_end()
            with redirect_stdout(StringIO()) as captured:
                callback._on_rollout_start()
                self.assertFalse(callback._stop_requested)
                callback._on_rollout_end()
                callback._on_rollout_start()

        self.assertTrue(callback._stop_requested)
        self.assertEqual(callback.validation_evaluations, 2)
        self.assertAlmostEqual(callback.best_validation_gap, 0.095)
        self.assertEqual(callback.model.save.call_count, 2)
        self.assertFalse(callback._on_step())
        # The logged keys must not claim a stale reference rule (they used to
        # read ``fifo_relative_gap`` while the gap was measured against LST).
        logged = [entry.args[0] for entry in callback.model.logger.record.call_args_list]
        self.assertIn("validation/rule_relative_gap", logged)
        self.assertIn("validation/best_rule_relative_gap", logged)
        self.assertFalse(any("fifo" in key for key in logged))
        self.assertIn("serial_LST", captured.getvalue())

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
        # Aggregation is degree-normalized, so the dense reference must divide
        # each node's message by its in/out degree as well.
        out_degree = valid.sum(dim=2).to(embeddings.dtype)
        in_degree = torch.zeros(1, 4).scatter_add_(
            1, flat_targets, valid_slots.reshape(1, -1)
        )
        successor_mean = successor_sum / out_degree.clamp(min=1).unsqueeze(-1)
        predecessor_mean = predecessor_sum / in_degree.clamp(min=1).unsqueeze(-1)

        compact_predecessors, compact_successors = _aggregate_edge_messages(
            embeddings,
            torch.tensor([[0, 0, 1, 2]]),
            torch.tensor([[1, 2, 3, 3]]),
            torch.ones((1, 4), dtype=torch.bool),
            in_degree,
            out_degree,
        )
        torch.testing.assert_close(compact_predecessors, predecessor_mean)
        torch.testing.assert_close(compact_successors, successor_mean)

    def test_edge_messages_are_averaged_over_neighbours_not_summed(self):
        # Node 3 has two predecessors, so its message must be their mean.  A sum
        # would double the magnitude and grow with in-degree, which is what
        # makes high-degree graphs unstable.
        embeddings = torch.tensor(
            [[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [0.0, 0.0]]]
        )
        edge_sources = torch.tensor([[0, 0, 1, 2]])
        edge_targets = torch.tensor([[1, 2, 3, 3]])
        edge_mask = torch.ones((1, 4), dtype=torch.bool)
        in_degree, out_degree = _edge_degrees(
            edge_sources, edge_targets, edge_mask, 4, embeddings.dtype
        )
        torch.testing.assert_close(in_degree, torch.tensor([[0.0, 1.0, 1.0, 2.0]]))
        torch.testing.assert_close(out_degree, torch.tensor([[2.0, 1.0, 1.0, 0.0]]))
        predecessors, _ = _aggregate_edge_messages(
            embeddings, edge_sources, edge_targets, edge_mask, in_degree, out_degree
        )
        torch.testing.assert_close(
            predecessors[0, 3], (embeddings[0, 1] + embeddings[0, 2]) / 2
        )
        # An isolated node has no neighbours; the clamp must keep it at zero
        # instead of dividing by zero.
        torch.testing.assert_close(predecessors[0, 0], torch.zeros(2))

    def test_unpadded_edge_aggregation_matches_padded_forward_and_gradient(self):
        embeddings = torch.randn(3, 5, 4, requires_grad=True)
        edge_sources = torch.tensor(
            [[0, 1, 0, 0], [0, 2, 3, 0], [1, 0, 0, 0]]
        )
        edge_targets = torch.tensor(
            [[1, 2, 0, 0], [2, 3, 4, 0], [4, 0, 0, 0]]
        )
        edge_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False], [True, False, False, False]]
        )
        in_degree, out_degree = _edge_degrees(
            edge_sources, edge_targets, edge_mask, 5, embeddings.dtype
        )
        padded = _aggregate_edge_messages(
            embeddings,
            edge_sources,
            edge_targets,
            edge_mask,
            in_degree,
            out_degree,
        )
        flat_sources, flat_targets = _compact_edge_indices(
            edge_sources, edge_targets, edge_mask, 5
        )
        compact = _aggregate_compact_edge_messages(
            embeddings, flat_sources, flat_targets, in_degree, out_degree
        )
        torch.testing.assert_close(compact[0], padded[0])
        torch.testing.assert_close(compact[1], padded[1])

        padded_gradient = torch.autograd.grad(
            padded[0].sum() + padded[1].sum(), embeddings, retain_graph=True
        )[0]
        compact_gradient = torch.autograd.grad(
            compact[0].sum() + compact[1].sum(), embeddings
        )[0]
        torch.testing.assert_close(compact_gradient, padded_gradient)

    def test_reference_rules_are_loaded_by_unique_instance_id(self):
        with TemporaryDirectory() as directory:
            result_path = Path(directory) / "baselines.csv"
            result_path.write_text(
                "file,serial_LST,serial_FIFO\n"
                "psplib/j30/j3010_1.sm,42,59\n"
                "psplib/j30/j3010_2.sm,43,58\n",
                encoding="utf-8",
            )
            loaded = load_reference_rules(result_path, "serial_LST")
            self.assertEqual(
                loaded,
                {
                    "psplib/j30/j3010_1": 42,
                    "psplib/j30/j3010_2": 43,
                },
            )
            with self.assertRaisesRegex(ValueError, "missing reference column"):
                load_reference_rules(result_path, "does_not_exist")

    def test_ppo_rejects_misaligned_static_cache(self):
        base = RCPSPEnv(TEST_INSTANCE)
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
        training_instance = load_core_instance(TEST_INSTANCE)
        env = make_sb3_env(TEST_INSTANCE)
        model = create_ppo(
            env,
            instances=[training_instance],
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
                max_activities=len(training_instance.activities),
                max_resources=training_instance.resource_count,
            )
            training_cache = build_static_graph_cache(
                [training_instance],
                max_activities=reference_env.max_activities,
                max_resources=reference_env.max_resources,
                max_successors=extractor.max_successors,
            )
            # Evaluation on a *different* instance exercises the temporary
            # static-cache switch and its restoration afterwards.
            sampled = evaluate_paths_sampled(
                model,
                [str(TEST_INSTANCE_2)],
                seed=11,
                reference_env=reference_env,
                sample_budgets=(1, 2),
                batch_size=1,
                restore_cache=training_cache,
            )
            self.assertEqual(set(sampled), {1, 2})
            self.assertLessEqual(sampled[2][0][1], sampled[1][0][1])
            self.assertEqual(extractor.instance_names, ("j3010_1",))
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
        base = RCPSPEnv(TEST_INSTANCE)
        # SB3's env_checker warns that the per-activity observation matrices
        # (dynamic_activity_features / resource_demands / resource_profile) are
        # neither images nor flat 1D vectors.  That 2D activity x feature layout
        # is intentional: the custom GIN policy consumes it directly.  We silence
        # only this specific structural hint so the remaining env_checker
        # contract (spaces, step semantics, determinism) still runs as errors.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Your observation .* has an unconventional shape .*",
                category=UserWarning,
            )
            check_env(base, warn=True, skip_render_check=True)
        # The static graph catalog holds two instances so that the later
        # evaluate_paths call (two paths, catalog_size=2) stays in range.
        training_instances = [
            load_core_instance(TEST_INSTANCE),
            load_core_instance(TEST_INSTANCE_2),
        ]
        env = make_sb3_env(TEST_INSTANCE)
        model = create_ppo(
            env, instances=training_instances, n_steps=8, batch_size=8,
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

        # Evaluate on two training instances through the protocol loader.  The
        # generated pool is size-stratified (generated/.../n30|n60|n90|n120/),
        # and this test's model is built on TEST_INSTANCE's small envelope, so
        # pick n30 entries that fit it.
        protocol = read_protocol(Path("splits.json"))
        paths = [p for p in protocol["train"] if "/n30/" in p][:2]
        self.assertEqual(len(paths), 2)
        loader = loader_for(Path("data"))
        name_fn = instance_id
        reference_env = SimpleNamespace(
            max_activities=base.activity_count,
            max_resources=base.resource_count,
        )
        training_cache = build_static_graph_cache(
            training_instances,
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
            loader=loader,
            name_fn=name_fn,
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
            loader=loader,
            name_fn=name_fn,
        )
        self.assertEqual(batched, sequential)
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            evaluate_paths(
                model, paths, seed=11, reference_env=reference_env, batch_size=0
            )


if __name__ == "__main__":
    unittest.main()

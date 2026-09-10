"""Tests for the size-stable critic graph pooling.

The critic summarizes a variable number of activity embeddings.  Its pooled
vector must stay in the same numeric range whether an instance has 32
activities (PSPLIB j30) or 302 (RG300), otherwise the value head extrapolates
far outside the range it was trained on.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.adapter import load_core_instance
from src.envs.observation import (
    DYNAMIC_ACTIVITY_FEATURE_COUNT,
    ObservationLayout,
    build_static_graph_cache,
)
from src.training.environments import make_multi_env
from src.training.features import pool_graph_embedding
from src.training.ppo import create_ppo
from tests import TEST_INSTANCE, TEST_INSTANCE_2


EMBEDDING_DIM = 4
GLOBAL_DIM = 2
PADDED_CAP = 48


def _row(
    node_count: int,
    valid_count: int,
    *,
    real_value: float = 0.5,
    fill_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One batch row: ``valid_count`` real nodes followed by padding."""
    embeddings = torch.full((1, node_count, EMBEDDING_DIM), fill_value)
    embeddings[0, :valid_count] = real_value
    mask = torch.zeros((1, node_count))
    mask[0, :valid_count] = 1.0
    return embeddings, mask, torch.ones((1, GLOBAL_DIM))


class GraphPoolingTest(unittest.TestCase):
    """Pooled statistics must not depend on how many real nodes exist."""

    def test_pooled_summary_does_not_scale_with_real_node_count(self):
        # 32 real nodes (j30) and 302 real nodes (RG300) holding the same
        # per-node value must pool to the same vector.  A node-count sum would
        # return 32*v versus 302*v here, which is the magnitude drift this
        # pooling exists to avoid.
        j30_like = pool_graph_embedding(*_row(302, 32))
        rg300_like = pool_graph_embedding(*_row(302, 302))
        torch.testing.assert_close(j30_like, rg300_like)

    def test_padding_slots_cannot_win_the_max(self):
        embeddings, mask, global_embedding = _row(8, 3, real_value=-1.0)
        # Padded slots hold values that would dominate the max if unmasked.
        embeddings[0, 3:] = 1000.0
        pooled = pool_graph_embedding(embeddings, mask, global_embedding)
        max_block = pooled[0, EMBEDDING_DIM:2 * EMBEDDING_DIM]
        torch.testing.assert_close(max_block, torch.full((EMBEDDING_DIM,), -1.0))

    def test_all_padding_row_stays_finite(self):
        pooled = pool_graph_embedding(
            torch.zeros(1, 5, EMBEDDING_DIM),
            torch.zeros(1, 5),
            torch.zeros(1, GLOBAL_DIM),
        )
        self.assertTrue(bool(torch.isfinite(pooled).all()))


class CriticPaddingInvarianceTest(unittest.TestCase):
    """The full policy must ignore whatever sits in the padded slots."""

    def setUp(self):
        self.instances = [
            load_core_instance(TEST_INSTANCE),
            load_core_instance(TEST_INSTANCE_2),
        ]
        self.activity_count = len(self.instances[0].activities)
        self.max_resources = max(
            instance.resource_count for instance in self.instances
        )
        self.layout = ObservationLayout(PADDED_CAP, self.max_resources)
        self.env = make_multi_env(
            [str(TEST_INSTANCE)],
            max_activities=PADDED_CAP,
            max_resources=self.max_resources,
            instance_indices=[0],
            catalog_size=len(self.instances),
        )
        self.model = create_ppo(
            self.env,
            instances=self.instances,
            static_cache=build_static_graph_cache(
                self.instances,
                max_activities=PADDED_CAP,
                max_resources=self.max_resources,
            ),
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            gin_layers=2,
            seed=1,
            device="cpu",
        )
        self.model.verbose = 0

    def tearDown(self):
        self.env.close()

    def _corrupted_observation(self, observation: np.ndarray) -> np.ndarray:
        """Fill every padded slot with a value the policy must ignore."""
        corrupted = np.array([observation], dtype=np.float32)
        tail = slice(self.activity_count, PADDED_CAP)
        corrupted[0, self.layout.activity_status][tail] = 0.9
        corrupted[0, self.layout.precedence_satisfied][tail] = 0.9
        corrupted[0, self.layout.eligible_mask][tail] = 0.9
        dynamic = corrupted[0, self.layout.dynamic_activity_features].reshape(
            PADDED_CAP, DYNAMIC_ACTIVITY_FEATURE_COUNT
        )
        dynamic[tail] = 0.9
        return corrupted

    def test_value_and_logits_ignore_padded_slots(self):
        observation = self.env.reset()[0]
        clean = torch.as_tensor(np.array([observation], dtype=np.float32))
        dirty = torch.as_tensor(self._corrupted_observation(observation))
        with torch.no_grad():
            clean_value = float(self.model.policy.predict_values(clean)[0])
            dirty_value = float(self.model.policy.predict_values(dirty)[0])
            clean_logits = self.model.policy.get_distribution(clean).distribution.logits
            dirty_logits = self.model.policy.get_distribution(dirty).distribution.logits
        self.assertAlmostEqual(clean_value, dirty_value, places=6)
        torch.testing.assert_close(
            dirty_logits[0, : self.activity_count],
            clean_logits[0, : self.activity_count],
        )

    def test_short_learn_keeps_the_value_head_finite(self):
        self.model.learn(total_timesteps=8, progress_bar=False)
        for name, parameter in self.model.policy.named_parameters():
            self.assertTrue(
                bool(torch.isfinite(parameter).all()), f"{name} became non-finite"
            )


if __name__ == "__main__":
    unittest.main()

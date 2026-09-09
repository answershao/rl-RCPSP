import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from src.environments.multi_instance import (
    MPLIB2_GROUP_FIELDS,
    MultiInstanceRCMPSPEnv,
    make_overfit_splits,
    make_splits,
    write_split_manifest,
    write_splits,
)
from src.environments.observation import (
    MAX_SUCCESSORS,
    ObservationLayout,
    build_static_graph_cache,
)
from src.environments.sb3_env import make_sb3_env
from test import TEST_INSTANCE


class MultiInstanceTest(unittest.TestCase):
    def test_splits_each_mplib2_parameter_group_three_one_one(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fields = ["Instance", *MPLIB2_GROUP_FIELDS, "destination"]
            rows = []
            for group_index, resource_strength in enumerate(("0.1", "0.9")):
                for replicate in range(5):
                    instance_number = group_index * 5 + replicate
                    filename = f"MPLIB2_Set1_{instance_number}.rcmp"
                    (root / filename).touch()
                    row = {field: "0" for field in fields}
                    row.update(
                        {
                            "Instance": str(instance_number),
                            "set": "MPLIB 2 - Set 1",
                            "J": "10",
                            "I": "500",
                            "K": "5",
                            "RS": resource_strength,
                            "destination": str(root / filename),
                        }
                    )
                    rows.append(row)
            with (root / "instances.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";")
                writer.writeheader()
                writer.writerows(rows)

            splits = make_splits(root)

            self.assertEqual(
                {name: len(paths) for name, paths in splits.items()},
                {"train": 6, "validation": 2, "test": 2},
            )
            self.assertEqual(
                {Path(path).name for paths in splits.values() for path in paths},
                {f"MPLIB2_Set1_{index}.rcmp" for index in range(10)},
            )
            self.assertEqual(
                [Path(path).stem.rsplit("_", 1)[1] for path in splits["validation"]],
                ["3", "8"],
            )
            self.assertEqual(
                [Path(path).stem.rsplit("_", 1)[1] for path in splits["test"]],
                ["4", "9"],
            )

    def test_write_splits_creates_parent_directory(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "splits.json"
            self.assertEqual(write_splits(path), path)
            self.assertTrue(path.is_file())

    def test_overfit_splits_sample_distinct_groups_reproducibly(self):
        splits = {
            "train": [f"group-{group}-rep-{rep}" for group in range(5) for rep in range(3)],
            "validation": [f"group-{group}-rep-3" for group in range(5)],
            "test": [f"group-{group}-rep-4" for group in range(5)],
        }

        first = make_overfit_splits(splits, 3, seed=17)
        second = make_overfit_splits(splits, 3, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(first["train"], first["validation"])
        self.assertEqual(first["test"], [])
        self.assertEqual(len({path.split("-")[1] for path in first["train"]}), 3)

    def test_overfit_split_count_must_not_exceed_group_count(self):
        splits = {"train": [str(index) for index in range(6)]}
        with self.assertRaisesRegex(ValueError, "between 1 and 2"):
            make_overfit_splits(splits, 3, seed=17)

    def test_write_split_manifest_preserves_resolved_splits(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "overfit" / "splits.json"
            splits = {"train": ["a"], "validation": ["a"], "test": []}
            self.assertEqual(write_split_manifest(splits, path), path)
            self.assertEqual(json.loads(path.read_text()), splits)

    def test_fixed_padding_and_episode(self):
        splits = make_splits()
        env = MultiInstanceRCMPSPEnv(splits["train"][:2])
        obs, _ = env.reset(seed=3)
        self.assertTrue(env.observation_space.contains(obs))

    def test_observation_uses_catalog_instance_index(self):
        env = MultiInstanceRCMPSPEnv(
            [TEST_INSTANCE],
            instance_indices=[3],
            catalog_size=5,
        )
        observation, _ = env.reset(seed=3)
        layout = ObservationLayout(env.max_activities, env.max_resources)
        self.assertAlmostEqual(float(observation[layout.instance_index]), 0.8)

    def test_static_graph_cache_contains_successor_indices(self):
        env = MultiInstanceRCMPSPEnv([TEST_INSTANCE])
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
        multi_env = MultiInstanceRCMPSPEnv([TEST_INSTANCE])
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

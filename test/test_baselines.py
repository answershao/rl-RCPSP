import csv
from pathlib import Path
import tempfile
import unittest

from scripts.baselines import find_instances, instance_sort_key, write_summary


class BaselinesTest(unittest.TestCase):
    def test_instance_sort_key_uses_numeric_set_and_instance_numbers(self) -> None:
        paths = [
            Path("MPLIB2_Set2_0.rcmp"),
            Path("MPLIB2_Set1_1002.rcmp"),
            Path("MPLIB2_Set1_99.rcmp"),
            Path("MPLIB2_Set1_9.rcmp"),
            Path("other.rcmp"),
        ]

        ordered = sorted(paths, key=instance_sort_key)

        self.assertEqual(
            [path.name for path in ordered],
            [
                "MPLIB2_Set1_9.rcmp",
                "MPLIB2_Set1_99.rcmp",
                "MPLIB2_Set1_1002.rcmp",
                "MPLIB2_Set2_0.rcmp",
                "other.rcmp",
            ],
        )

    def test_find_instances_uses_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "MPLIB2_Set2_0.rcmp",
                "MPLIB2_Set1_1002.rcmp",
                "MPLIB2_Set1_99.rcmp",
            ):
                (root / name).touch()

            self.assertEqual(
                [path.name for path in find_instances(root)],
                [
                    "MPLIB2_Set1_99.rcmp",
                    "MPLIB2_Set1_1002.rcmp",
                    "MPLIB2_Set2_0.rcmp",
                ],
            )

    def test_write_summary_includes_exact_solver_metadata(self) -> None:
        rows = [{
            "instance": "sample.rcmp",
            "FIFO": 12,
            "Shortest": 11,
            "Random": 13,
            "CP-SAT": 10,
            "Remark": "feasible",
            "Bound": 9.0,
            "Wall Time": 1.25,
        }]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "summary.csv"

            write_summary(rows, output_path, include_exact=True)

            with output_path.open(newline="", encoding="ascii") as output_file:
                result = list(csv.DictReader(output_file))
            self.assertEqual(
                list(result[0]),
                [
                    "instance", "FIFO", "Shortest", "Random", "CP-SAT",
                    "Remark", "Bound", "Wall Time",
                ],
            )
            self.assertEqual(result[0]["Remark"], "feasible")
            self.assertEqual(result[0]["Bound"], "9.0")
            self.assertEqual(result[0]["Wall Time"], "1.25")


if __name__ == "__main__":
    unittest.main()

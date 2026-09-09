import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.core.rcpsp import (
    generate_schedule,
    preview_serial_sgs_insert,
    priority_fifo,
    priority_shortest_duration,
    random_priorities,
    serial_sgs_insert,
    validate_schedule,
)
from src.data.adapter import load_core_instance
from src.visualization.aon import plot_aon
from src.visualization.gantt import plot_gantt
from tests import TEST_INSTANCE


class RcpspTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.instance = load_core_instance(TEST_INSTANCE)

    def test_adapter_reads_expected_structure(self) -> None:
        instance = self.instance
        # j3010_1 is a 32-activity / 4-resource PSPLIB j30 instance (two of the
        # 32 rows are the zero-duration dummy source and sink).
        self.assertEqual(len(instance.activities), 32)
        self.assertEqual(instance.resource_count, 4)
        self.assertEqual(len(instance.capacities), 4)
        zero_duration = [
            activity.id
            for activity in instance.activities.values()
            if activity.duration == 0
        ]
        self.assertEqual(len(zero_duration), 2)
        for activity in instance.activities.values():
            for successor in activity.successors:
                self.assertIn(activity.id, instance.predecessors[successor])

    def test_baselines_generate_valid_schedules(self) -> None:
        priorities = [
            priority_fifo,
            priority_shortest_duration,
            random_priorities(self.instance, seed=7),
        ]
        for priority in priorities:
            schedule = generate_schedule(self.instance, priority)
            validate_schedule(self.instance, schedule)
            self.assertEqual(len(schedule.starts), 32)
            self.assertGreater(schedule.makespan, 0)

    def test_random_priority_is_reproducible(self) -> None:
        first = generate_schedule(self.instance, random_priorities(self.instance, seed=19))
        second = generate_schedule(self.instance, random_priorities(self.instance, seed=19))
        self.assertEqual(first, second)

    def test_serial_sgs_preview_is_non_mutating_and_matches_insert(self) -> None:
        # The dummy source has no predecessors, so it is eligible on an empty
        # partial schedule (duration 0 -> placement at time 0).
        activity_id = next(
            activity.id
            for activity in self.instance.activities.values()
            if not self.instance.predecessors[activity.id]
        )
        duration = self.instance.activities[activity_id].duration
        finishes = {}
        usage = np.zeros((duration + 1, self.instance.resource_count), dtype=np.int32)
        usage_before = usage.copy()

        preview = preview_serial_sgs_insert(
            self.instance, activity_id, finishes, usage
        )

        self.assertEqual(finishes, {})
        np.testing.assert_array_equal(usage, usage_before)
        starts = {}
        inserted = serial_sgs_insert(
            self.instance, activity_id, starts, finishes, usage
        )
        self.assertEqual(inserted, preview)
        self.assertEqual(inserted, (0, duration))

    def test_gantt_chart_is_written(self) -> None:
        output = Path("test-output-gantt.png")
        try:
            schedule = generate_schedule(self.instance, priority_fifo)
            result = plot_gantt(self.instance, schedule, output)
            self.assertEqual(result, output)
            self.assertGreater(output.stat().st_size, 0)
        finally:
            if output.exists():
                output.unlink()

    def test_aon_chart_is_written(self) -> None:
        output = Path("test-output-aon.png")
        try:
            schedule = generate_schedule(self.instance, priority_fifo)
            result = plot_aon(self.instance, output, schedule=schedule)
            self.assertEqual(result, output)
            self.assertGreater(output.stat().st_size, 0)
        finally:
            if output.exists():
                output.unlink()


if __name__ == "__main__":
    unittest.main()

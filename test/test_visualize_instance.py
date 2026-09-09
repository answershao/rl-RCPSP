from pathlib import Path
import tempfile
import unittest

from scripts.visualize_instance import render_instance
from test import TEST_INSTANCE


class VisualizeInstanceTest(unittest.TestCase):
    def test_render_instance_writes_both_charts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            gantt, aon = render_instance(TEST_INSTANCE, Path(directory))
            self.assertEqual(gantt.name, "gantt.png")
            self.assertEqual(aon.name, "aon.png")
            self.assertGreater(gantt.stat().st_size, 0)
            self.assertGreater(aon.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visualize_instance import render_instance
from tests import TEST_INSTANCE


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

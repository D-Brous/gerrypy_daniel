import sys

sys.path.append("../gerrypy_daniel")

from data.partition import Partition
import numpy as np
import unittest


class TestPartition(unittest.TestCase):
    def setUp(self):
        self.partition = Partition(2, 10)
        self.partition_full = Partition(2, 10)
        self.partition_full.set_part(0, [0, 2, 4, 6, 8])
        self.partition_full.set_part(1, [1, 3, 5, 7, 9])
        self.partition_none = Partition(2)
        self.partition_none_full = Partition(2)
        self.partition_none_full.set_part(0, [0, 2, 4, 6, 8])
        self.partition_none_full.set_part(1, [1, 3, 5, 7, 9])

        def array_equal(a, b, msg=None):
            if not np.array_equal(a, b):
                return msg

        self.addTypeEqualityFunc(np.ndarray, array_equal)

    def test_get_parts(self):
        self.assertEqual(self.partition.get_parts(), {0: [], 1: []})
        self.assertEqual(self.partition_none.get_parts(), {0: [], 1: []})
        self.assertEqual(
            self.partition_full.get_parts(),
            {0: [0, 2, 4, 6, 8], 1: [1, 3, 5, 7, 9]},
        )
        self.assertEqual(
            self.partition_none_full.get_parts(),
            {0: [0, 2, 4, 6, 8], 1: [1, 3, 5, 7, 9]},
        )

    def test_get_assignment(self):
        self.assertEqual(
            self.partition.get_assignment(),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int),
        )
        self.assertEqual(
            self.partition_full.get_assignment(),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int),
        )
        self.assertEqual(
            self.partition_none.get_assignment(), np.zeros(0, dtype=int)
        )
        self.assertEqual(
            self.partition_none_full.get_assignment(),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int),
        )

    def test_get_part(self):
        self.assertEqual(self.partition.get_part(0), [])
        self.assertEqual(self.partition.get_part(1), [])
        self.assertEqual(self.partition_full.get_part(0), [0, 2, 4, 6, 8])
        self.assertEqual(self.partition_full.get_part(1), [1, 3, 5, 7, 9])
        with self.assertRaises(ValueError) as cm:
            self.partition.get_part(2)
        self.assertEqual(
            cm.exception.args[0],
            "Expected int in the interval [0, 1] but got 2",
        )
        with self.assertRaises(ValueError) as cm_2:
            self.partition_none.get_part(0.5)
        self.assertEqual(
            cm_2.exception.args[0],
            "Expected int in the interval [0, 1] but got 0.5",
        )

    def test_set_part(self):
        with self.assertRaises(ValueError) as cm:
            self.partition.set_part(2, [0, 1, 2])
        self.assertEqual(
            cm.exception.args[0],
            "Expected int in the interval [0, 1] but got 2",
        )
        self.assertEqual(self.partition.get_parts(), {0: [], 1: []})


if __name__ == "__main__":
    unittest.main()

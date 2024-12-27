import numpy as np
import networkx as nx
import unittest
import sys

sys.path.append("../gerrypy_daniel")
from data.config import StateConfig
from data.partition import Partition
from data.graph import Graph
from data.df import ShapeDataFrame, DemoDataFrame  # , ACSDataFrame

# class TestStateConfig(unittest.TestCase):


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


class TestShapeDataFrame(unittest.TestCase):
    """
    Requires: the shapefiles for LA, 2010, blockgroup have been downloaded.
    """

    def setUp(self):
        self.config = StateConfig("LA", 2010, "block_group")
        self.shape_df = ShapeDataFrame.from_config(self.config)

    def test_from_config(self):
        self.assertEqual(isinstance(self.shape_df, ShapeDataFrame), True)
        self.assertEqual("GEOID" in self.shape_df.columns, True)
        self.assertEqual("geometry" in self.shape_df.columns, True)
        self.assertEqual("GEOID10" in self.shape_df.columns, False)


class TestDemoDataFrame(unittest.TestCase):
    """
    Requires: the shapefiles for LA, 2010, blockgroup have been downloaded.
    """

    def setUp(self):
        self.config = StateConfig("LA", 2010, "block_group")
        self.acs_df = DemoDataFrame.from_config(self.config)

    def test_from_config(self):
        self.assertEqual(isinstance(self.acs_df, DemoDataFrame), True)
        self.assertEqual("BVAP" in self.acs_df.columns, True)
        self.assertEqual("GEOID10" in self.acs_df.columns, False)


'''
class TestACSDataFrame(unittest.TestCase):
    """
    Requires: the shapefiles for LA, 2010, blockgroup have been downloaded.
    """

    def setUp(self):
        self.config = StateConfig("LA", "2010", "block_group")
        self.acs_df = ACSDataFrame.from_config(self.config)

    def test_from_config(self):
        self.assertEqual(isinstance(self.acs_df, ACSDataFrame), True)
        self.assertEqual("BVAP" in self.acs_df.columns, True)
        self.assertEqual("GEOID10" in self.acs_df.columns, False)
'''

if __name__ == "__main__":
    # G = Graph.from_file("./data/optimization_data/block_group/LA/2010/G.p")
    # print(G.nodes())

    state_config = StateConfig("LA", 2010, "block_group")
    shape_df = ShapeDataFrame.from_config(state_config)
    # acs_df = ACSDataFrame.from_config(state_config)
    # cgu_adj_graph = Graph.cgu_adjacency_graph(state_config)
    # print(type(cgu_adj_graph))
    # print(cgu_adj_graph.nodes)
    # cgu_adj_graph.save_to("data/G.p")
    # cgu_adj_graph_2 = Graph.from_file("data/G.l")
    # print(nx.difference(cgu_adj_graph, cgu_adj_graph_2))
    # cgu_adj_graph_2.save_to("data/G.l")
    print(shape_df.columns)
    # print(acs_df.columns)
    print(type(shape_df))
    # print(type(acs_df))

    pops = DemoDataFrame.from_config(state_config)
    import pandas as pd
    import os
    import constants

    pops_old = pd.read_csv(
        os.path.join(
            constants.DEMO_DATA_PATH,
            state_config.get_dirname(),
            "pops_old.csv",
        ),
        low_memory=False,
    )
    pops_old = pops_old.sort_values("GEOID")
    pops_old = pops_old.reset_index(drop=True)
    print(pops)
    print(pops_old)
    for col in constants.COL_DICT_DEC_2010.values():
        if col != "POP":
            if pops[col].equals(pops_old[col]):
                print(f"column {col} is equal between the two")
            else:
                print(f"error at column {col}")
        else:
            if pops[col].equals(pops_old["TOTPOP"]):
                print(f"column {col} is equal between the two")
            else:
                print(f"error at column {col}")
    unittest.main()

import networkx as nx
import os
import libpysal
import sys

sys.path.append("../gerrypy_daniel")
import constants
from data.partition import Partition
from data.config import StateConfig
from data.df import ShapeDataFrame


class Graph(nx.Graph):
    @classmethod
    def from_nx(cls, graph: nx.Graph) -> "Graph":
        return cls(graph)

    @classmethod
    def from_file(cls, file_path: str) -> "Graph":
        return cls.from_nx(nx.read_gpickle(file_path))

    @classmethod
    def cgu_adjacency_graph(cls, state_config: StateConfig) -> "Graph":
        adjacency_graph_path = os.path.join(
            constants.OPT_DATA_PATH,
            state_config.granularity,
            state_config.state,
            str(state_config.year),
            "G.p",
        )
        return cls.from_file(adjacency_graph_path)

    @classmethod
    def district_adjacency_graph(
        cls, state_config: StateConfig, partition: Partition
    ) -> "Graph":
        shape_df = ShapeDataFrame.from_config(state_config)
        shape_df["Plan"] = partition.get_assignment()
        district_shapes = shape_df.dissolve(by="Plan")
        shape_list = district_shapes.geometry.to_list()
        return cls(
            libpysal.weights.Rook.from_iterable(shape_list).to_networkx()
        )

    def save_to(self, file_path: str):
        """KEEP IN MIND that the file extension must be .p"""
        nx.write_gpickle(self, os.path.join(file_path))


class CguGraph(nx.Graph):
    @classmethod
    def from_shape_df(cls, shape_df):
        shape_list = shape_df.geometry.to_list()
        return libpysal.weights.Rook.from_iterable(shape_list).to_networkx()

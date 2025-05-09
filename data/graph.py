import networkx as nx
import os
import libpysal
import pickle
from typing import Optional
import sys

sys.path.append(".")
import constants
from data.partition import Partition
from data.config import StateConfig
from data.shape_df import ShapeDataFrame


class Graph(nx.Graph):
    @classmethod
    def from_nx(cls, graph: nx.Graph) -> "Graph":
        return cls(graph)

    def to_nx(self) -> nx.Graph:
        return self

    @classmethod
    def from_file(cls, file_path: str) -> "Graph":
        return cls.from_nx(nx.read_gpickle(file_path))

    @classmethod
    def get_cgu_adjacency_graph(cls, state_config: StateConfig) -> "Graph":
        optimization_cache_path = os.path.join(
            constants.OPT_DATA_PATH, state_config.get_dirname()
        )
        adjacency_graph_path = os.path.join(
            optimization_cache_path,
            "G.gpickle",
        )
        subregion = state_config.subregion
        if os.path.exists(adjacency_graph_path):
            # Load full state cgu adjacency graph
            G = cls.from_file(adjacency_graph_path)
        else:
            # Cache full state cgu adjacency graph
            state_config.subregion = None
            shape_df = ShapeDataFrame.from_config(state_config)
            G = cls.from_shape_df(shape_df)
            if not os.path.exists(optimization_cache_path):
                os.mkdir(optimization_cache_path)
            G.save_to(adjacency_graph_path)

        # Return graph or subgraph if specified
        if subregion is not None:
            G = G.get_subgraph(state_config.subregion)
            state_config.subregion = subregion
        G.state_config = state_config
        return G

    @classmethod
    def from_shape_df(cls, shape_df: ShapeDataFrame) -> "Graph":
        shape_list = shape_df.geometry.to_list()
        return cls(
            libpysal.weights.Rook.from_iterable(shape_list).to_networkx()
        )

    @classmethod
    def get_district_adjacency_graph(
        cls, state_config: StateConfig, partition: Partition
    ) -> "Graph":
        shape_df = ShapeDataFrame.from_config(state_config)
        district_shape_df = shape_df.get_district_shape_df(partition)
        shape_list = district_shape_df.geometry.to_list()
        return cls(
            libpysal.weights.Rook.from_iterable(shape_list).to_networkx()
        )

    def save_to(self, file_path: str):
        """By convention, the file extension should be .gpickle"""
        nx.write_gpickle(self.to_nx(), os.path.join(file_path))

    def get_subgraph(self, subregion: list[int]) -> "Graph":
        return nx.subgraph(self, subregion)

    def get_edge_dists(
        self,
        centers: list[int],
        state_config: StateConfig,
        subregion: Optional[list[int]],
    ) -> dict[int, dict[int, int]]:
        """Creates a dict of edge distances for pairs of nodes. If the
        graph is for the full state, then centers is ignored and a dict
        of edge distances for all pairs of nodes is returned (this is to
        save time, since such a dict only needs to be created once). If
        the graph is for a subregion of the full state, then the dict only
        stores edge distances for each node from each center.

        Args:
            centers (list[int]): List of centers

        Returns:
            (dict[int, dict[int, int]]): dict of node1 : node2 : edge_dist
            where edge_dist is the edge distance between node1 and node2
            in the graph. The node1 keys are only the centers in centers
            if the graph is for a particular subregion
        """
        if subregion is None:
            edge_dists_path = os.path.join(
                constants.OPT_DATA_PATH,
                state_config.get_dirname(),
                "edge_dists.pickle",
            )

            if os.path.exists(edge_dists_path):
                return pickle.load(open(edge_dists_path, "rb"))
            else:
                edge_dists = dict(nx.all_pairs_shortest_path_length(self))
                pickle.dump(edge_dists, open(edge_dists_path, "wb"))
                return edge_dists
        else:
            return {
                center: nx.shortest_path_length(self, source=center)
                for center in centers
            }

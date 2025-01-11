import math
import random
import numpy as np
from typing import Optional, Tuple

import sys

sys.path.append(".")
from constants import IPStr, VapCol, flatten
from analyze.maj_min import is_maj_cvap
from data.config import SHPConfig
from data.partition import Partition
from data.df import DemoDataFrame


class SHPNode:
    def __init__(
        self,
        n_districts: int,
        subregion: list[int],
        id: int,
        parent_id: Optional[int] = None,
        center: Optional[int] = None,
    ):
        """
        SHPNodes store information needed to reconstruct the tree and
        gathers metadata from the generation process.

        Args:
            n_districts: (int) node capacity
            subregion: (list[int]) list of cgu ixs associated with the
              region that self represents
            id: (int) node id of self
            parent_id: (Optional[int]) parent node id, or None if self
                is the root
            center: (Optional[int]) cgu ix that represents the center
                around which self was built, or None if self is the root
        """
        self.n_districts = n_districts
        self.subregion = subregion
        self.id = id
        self.parent_id = parent_id
        self.center = center

        # self.child_centers_dict = {}
        # self.warm_starts = []

        # Lists of dictionaries where each dict is for a different sample,
        # and the dicts have information keyed by ip names
        self.child_ids = (
            []
        )  # List of dictionaries of branches, i.e. lists of child ids
        self.ip_info = (
            []
        )  # List of dictionaries of ip information, i.e. 3-tuple of ip_time, ip_status, and ip_obj_val.

        # self.ip_times = []
        # self.ip_obj_values = []
        # self.ips_used = []
        # self.ip_statuses = []

        self.n_infeasible_partitions = 0
        self.infeasible_children = 0

    def get_n_samples(self):
        """
        Returns the number of samples that have been taken so far
        """
        return len(self.child_ids)

    def update_child_ids(self, sample_ix, ip_str, ids):
        if sample_ix >= self.get_n_samples():
            self.child_ids.append({ip_str: ids})
        else:
            self.child_ids[sample_ix][ip_str] = ids

    def update_ip_info(self, sample_ix, ip_str, ip_info):
        if sample_ix >= self.get_n_samples():
            self.ip_info.append({ip_str: ip_info})
        else:
            self.ip_info[sample_ix][ip_str] = ip_info

    def get_branch(self, sample_ix: int, ip_str: IPStr) -> list[int]:
        return self.child_ids[sample_ix][ip_str]

    def get_all_branches(self) -> list[list[int]]:
        return [
            branch
            for branch_dict in self.child_ids
            for branch in branch_dict.values()
        ]

    def get_ip_branchs(self, ip_str: IPStr) -> list[list[int]]:
        return [
            branch_dict[ip_str]
            for branch_dict in self.child_ids
            if ip_str in branch_dict
        ]

    def find_branch(self, child_id: int) -> Tuple[int, IPStr, list[int]]:
        """Returns the sample id, ip and branch of self that contains
        the node given by child_id. Raises ValueError if none of the
        child nodes have that id number.

        Args:
            child_id (int): Child node id
        """
        for sample_ix, sample in enumerate(self.child_ids):
            for ip_str, branch in sample.items():
                if child_id in branch:
                    return sample_ix, ip_str, branch
        raise ValueError(
            f"Node {child_id} is not an existing child of node {self.id}"
        )

    def delete_sample(self, sample_ix: int):
        try:
            del self.child_ids[sample_ix]
            del self.ip_info[sample_ix]
        except IndexError:
            raise IndexError(
                f"{sample_ix} is not an existing sample ix for node {self.id}"
            )

    def delete_branch(self, sample_ix: int, ip_str: IPStr):
        """Deletes the branch associated the given sample ix and ip, along
        with any associated values stored. Raises ValueError if the branch
        does not exist.

        Args:
            sample_ix (int): Sample index
            ip_str (IPStr): IP name
        """
        n_ips = len(self.child_ids[sample_ix].keys())
        if n_ips > 1:
            del self.child_ids[sample_ix][ip_str]
            del self.ip_info[sample_ix][ip_str]
        elif n_ips == 1:
            self.delete_sample(sample_ix)
        else:
            raise ValueError(
                f"The branch of node {self.id} at (sample {sample_ix}, ip {ip_str}) doesn't exist."
            )

    def sample_n_splits_and_child_sizes(self, config: SHPConfig):
        """
        Samples both the split size and the capacity of all children.

        Args:
            config: (SHPConfig) SHP configuration

        Returns: (list[int]) List of child node capacities.

        """
        n_distrs = self.n_districts
        n_splits = random.randint(
            min(config.min_n_splits, n_distrs),
            min(config.max_n_splits, n_distrs),
        )
        if n_distrs in config.final_partition_range:
            n_splits = n_distrs

        ub = max(
            math.ceil(
                config.max_split_population_difference * n_distrs / n_splits
            ),
            2,
        )
        lb = max(
            math.floor(
                (1 / config.max_split_population_difference)
                * n_distrs
                / n_splits
            ),
            1,
        )

        child_n_distrs = np.zeros(n_splits, dtype="int") + lb
        while int(sum(child_n_distrs)) != n_distrs:
            ix = random.randint(0, n_splits - 1)
            if child_n_distrs[ix] < ub:
                child_n_distrs[ix] += 1

        return child_n_distrs

    def is_root(self):
        return self.id == 0

    def is_leaf(self):
        return self.n_districts == 1

    def is_maj_cvap(self, col: VapCol, demo_df: DemoDataFrame):
        return is_maj_cvap(col, self.subregion, demo_df)

    def __repr__(self):
        """Utility function for printing a SHPNode."""
        print_str = "Node %d \n" % self.id
        internals = self.__dict__
        for k, v in internals.items():
            if k == "area":
                continue
            print_str += k + ": " + v.__repr__() + "\n"
        return print_str


from data.df import DemoDataFrame
import pickle


class SHPTree:
    def __init__(self, config: SHPConfig):
        self.config = config
        demo_df = DemoDataFrame.from_config(config)
        self.root = SHPNode(self.config.n_districts, list(demo_df.index), 0)
        self.all_internal_nodes = {}
        self.all_leaf_nodes = {}
        self.max_root_partition_id = 0
        self.max_id = 0

        self.failed_root_samples = 0
        self.generation_times = np.zeros(
            (self.config.n_root_samples), dtype=float
        )
        self.n_infeasible_partitions = np.zeros(
            (self.config.n_root_samples), dtype=int
        )
        self.n_successful_partitions = np.zeros(
            (self.config.n_root_samples), dtype=int
        )
        self.n_optimal_partitions = np.zeros(
            (self.config.n_root_samples), dtype=int
        )
        self.n_time_limit_partitions = np.zeros(
            (self.config.n_root_samples), dtype=int
        )
        self.n_deleted_nodes = np.zeros((self.config.n_root_samples), dtype=int)

    def assign_id(self) -> int:
        """Returns next unused node id"""
        self.max_id += 1
        return self.max_id

    def get_root(self) -> SHPNode:
        return self.root

    def save_nodes(
        self, internal_nodes: dict[int, SHPNode], leaf_nodes: dict[int, SHPNode]
    ):
        self.all_internal_nodes[self.max_root_partition_id] = internal_nodes
        self.all_leaf_nodes[self.max_root_partition_id] = leaf_nodes
        self.max_root_partition_id += 1

    def get_nodes(
        self, root_partition_id: int
    ) -> tuple[dict[int, SHPNode], dict[int, SHPNode]]:
        if (
            root_partition_id not in self.all_internal_nodes
            or root_partition_id not in self.all_leaf_nodes
        ):
            raise ValueError(
                f"{root_partition_id} is not a valid root partition id in this tree. Expected a value in the range [0, {self.max_root_partition_id}]"
            )
        return (
            self.all_internal_nodes[root_partition_id],
            self.all_leaf_nodes[root_partition_id],
        )

    def save_to(self, file_path: str):
        pickle.dump(self, open(file_path, "wb"))

    @classmethod
    def from_file(cls, file_path: str) -> "SHPTree":
        return pickle.load(open(file_path, "rb"))

    def get_leaf_nodes_from_ip(
        self,
        ip_str: IPStr,
        root_partition_id: int,
    ):
        internal_nodes, leaf_nodes = self.get_nodes(root_partition_id)
        ip_partitioned_nodes = {}
        ip_leaf_nodes = {}

        for node in leaf_nodes.values():
            parent_id = node.parent_id
            parent_node = self.root
            if parent_id != 0:
                parent_node = internal_nodes[parent_id]
            if parent_node.n_districts in self.config.final_partition_range:
                ip_partitioned_nodes[parent_id] = parent_node
            else:
                ip_leaf_nodes[node.id] = node

        for node in ip_partitioned_nodes.values():
            ip_leaf_ids = flatten(node.get_ip_branches(ip_str))
            ip_leaf_nodes.update({id: leaf_nodes[id] for id in ip_leaf_ids})

        return ip_leaf_ids

    def get_solution_nodes_dp(
        self,
        internal_nodes: dict[int, SHPNode],
        leaf_nodes: dict[int, SHPNode],
        root_partition_id: int,
        col: VapCol,
    ) -> dict[int, SHPNode]:
        demo_df = DemoDataFrame.from_config(self.config)
        root_branch = self.root.get_branch(root_partition_id, "base")
        nodes = {**internal_nodes, **leaf_nodes}
        dp_queue = []
        parent_layer = [self.root]
        children_layer = [
            internal_nodes[id]
            for id in root_branch
            if id in nodes and not nodes[id].is_leaf()
        ]

        while len(children_layer) > 0:
            dp_queue += children_layer
            parent_layer = children_layer
            children_layer = [
                nodes[id]
                for node in parent_layer
                for id in flatten(node.get_all_branches())
                if id in nodes and not nodes[id].is_leaf()
            ]

        for node in leaf_nodes.values():
            node.best_subtree = (
                [node.id],
                int(node.is_maj_cvap(col, demo_df)),
            )

        for i in range(len(dp_queue) - 1, -1, -1):
            current_node = dp_queue[i]
            best_subtree_ids = []
            best_subtree_score = -1
            for branch in current_node.get_all_branches():
                try:
                    sample_score = sum(
                        nodes[id].best_subtree[1] for id in branch
                    )
                    if sample_score > best_subtree_score:
                        best_subtree_ids = [
                            subtree_id
                            for id in branch
                            for subtree_id in nodes[id].best_subtree[0]
                        ]
                        best_subtree_score = sample_score
                except:
                    continue
            current_node.best_subtree = (best_subtree_ids, best_subtree_score)
        return {
            subtree_id: leaf_nodes[subtree_id]
            for node_id in root_branch
            for subtree_id in nodes[node_id].best_subtree[0]
        }

    def get_partition(
        self,
        solution_nodes: dict[int, SHPNode],
    ) -> Partition:
        partition = Partition()
        for district_id, leaf_node in enumerate(solution_nodes.values()):
            partition.set_part(district_id, leaf_node.subregion)
        return partition

    def n_internal_nodes(self) -> int:
        return sum(
            len(internal_nodes)
            for internal_nodes in self.all_internal_nodes.values()
        )

    def n_leaf_nodes(self) -> int:
        return sum(
            len(leaf_nodes) for leaf_nodes in self.all_leaf_nodes.values()
        )

    def n_districtings(self) -> int:
        """
        Dynamic programming method to compute the total number of
        districtings
        """

        def recursive_compute(node: SHPNode, nodes: dict[int, SHPNode]):
            if node.is_leaf():
                return 1

            n_districtings = 0
            for branch in node.get_all_branches():
                branch_districtings = 1
                for child_id in branch:
                    branch_districtings *= recursive_compute(
                        nodes[child_id], nodes
                    )
                n_districtings += branch_districtings
            return branch_districtings

        all_nodes = dict()
        for root_partition_id in range(self.max_root_partition_id):
            internal_nodes, leaf_nodes = self.get_nodes(root_partition_id)
            all_nodes.update({**internal_nodes, **leaf_nodes})
        return recursive_compute(self.root, all_nodes)


'''
class ExampleTree:
    """
    For illustration purposes.
    """

    def __init__(self, config, n_districts, level=0):
        self.n_districts = n_districts
        self.level = 0

        if n_districts > 1:
            children_n_distrs = SHPNode.sample_n_splits_and_child_sizes()
            self.children = [
                ExampleTree(config, n, level + 1) for n in children_n_distrs
            ]
        else:
            self.children = None

        self.max_levels_to_leaf = 0
        self.max_layer()

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.n_districts) + "\n"
        if self.children is not None:
            for child in self.children:
                ret += child.__repr__(level + 1)
        return ret

    def max_layer(self):
        try:
            to_leaf = 1 + max([child.max_layer() for child in self.children])
            self.max_levels_to_leaf = to_leaf
            return to_leaf
        except TypeError:
            return 0
'''

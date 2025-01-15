import sys

sys.path.append(".")

import time
import numpy as np
import networkx as nx
from typing import Optional
from collections import OrderedDict

from optimize.center_selection import *

from constants import VapCol, flatten
from optimize.ip import *
from optimize.tree import SHPNode, SHPTree
from data.config import SHPConfig, StateConfig
from data.df import DemoDataFrame, ShapeDataFrame
from data.graph import Graph
from optimize.logging import Logger


def load_opt_data(config: StateConfig):  # TODO: get rid of?
    demo_df = DemoDataFrame.from_config(config)
    G = Graph.get_cgu_adjacency_graph(config)
    shape_df = ShapeDataFrame.from_config(config)
    lengths = shape_df.get_lengths(config)
    return demo_df, G, lengths


class ColumnGenerator:
    """
    Generates columns with the Stochastic Hierarchical Paritioning algorithm.
    Maintains samples tree and logging data.
    """

    def __init__(self, config: SHPConfig, tree: SHPTree, logger: Logger):
        self.config = config
        self.tree = tree
        self.logger = logger

        demo_df, G, lengths = load_opt_data(config)
        lengths /= 1000  # TODO: get rid of?
        self.G = G
        self.demo_df = demo_df
        self.lengths = lengths
        self.shape_df = ShapeDataFrame.from_config(config)

        self.ideal_pop = demo_df.get_ideal_pop(config.n_districts)
        self.max_pop_variation = self.ideal_pop * config.population_tolerance

        self.root = tree.get_root()

        # random.seed(0)
        # np.random.seed(0)

        self.logger.log_root()

    def retry_sample(
        self,
        problem_node: SHPNode,
        internal_nodes: dict[int, SHPNode],
        leaf_nodes: dict[int, SHPNode],
        queue: list[SHPNode],
    ):
        def get_descendents(node_id: int):
            direct_descendents = flatten(
                internal_nodes[node_id].get_all_branches()
            )
            indirect_descendants = flatten(
                [
                    get_descendents(child_id)
                    for child_id in direct_descendents
                    if (child_id in internal_nodes)
                ]
            )
            return direct_descendents + indirect_descendants

        if problem_node.id == 0:
            raise RuntimeError("Root partition failed")
        if problem_node.parent_id == 0:
            raise RuntimeError("Root partition region not subdivisible")
        parent = internal_nodes[problem_node.parent_id]

        parent.infeasible_children += 1
        if parent.infeasible_children > self.config.parent_resample_tries:
            # Failure couldn't be corrected -- retry from the next node up
            return self.retry_sample(parent, internal_nodes, leaf_nodes, queue)

        sample_ix, ip_str, branch = parent.find_branch(problem_node.id)
        nodes_to_delete = set()
        for node_id in branch:
            nodes_to_delete.add(node_id)
            if node_id in internal_nodes:
                for child_id in get_descendents(node_id):
                    nodes_to_delete.add(child_id)

        n_deleted_nodes = 0
        for node_id in nodes_to_delete:
            if node_id in leaf_nodes:
                del leaf_nodes[node_id]
                self.logger.log_node_deletion(node_id)
                n_deleted_nodes += 1
            elif node_id in internal_nodes:
                del internal_nodes[node_id]
                self.logger.log_node_deletion(node_id)
                n_deleted_nodes += 1
        self.tree.n_deleted_nodes[
            self.tree.max_root_partition_id
        ] += n_deleted_nodes

        parent.delete_branch(sample_ix, ip_str)
        return [parent] + [n for n in queue if n.id not in nodes_to_delete]

    def generate_root_partition_tree(
        self,
    ) -> tuple[dict[int, SHPNode], dict[int, SHPNode]]:
        t_init = time.thread_time()
        self.logger.print_new_root_partition(self.tree.max_root_partition_id)
        while True:
            try:
                internal_nodes, leaf_nodes = self.attempt_generation()
                self.tree.save_nodes(internal_nodes, leaf_nodes)
                self.tree.generation_times[self.tree.max_root_partition_id] = (
                    time.thread_time() - t_init
                )
                self.logger.print_completed_root_partition(
                    self.tree.max_root_partition_id, self.tree
                )
                self.tree.max_root_partition_id += 1
                return internal_nodes, leaf_nodes
            except (
                RuntimeError
            ) as error:  # Errors propagated up to the root enough times for us to give up and try again from the beginning.
                self.logger.print_failed_root_partition(str(error))
                self.logger.log_failed_root_partition(str(error))
                self.root.delete_sample(self.tree.max_root_partition_id)
                self.tree.failed_root_samples += 1

    def attempt_generation(
        self,
    ) -> tuple[dict[int, SHPNode], dict[int, SHPNode]]:
        leaf_nodes = {}
        internal_nodes = {}
        queue = [self.root]
        while len(queue) > 0:
            # node = queue.pop() #DFS
            node = queue.pop(0)  # BFS
            children = self.sample_node(node)
            if len(children) == 0:  # Failure detected
                # Try to correct failure
                queue = self.retry_sample(
                    node, internal_nodes, leaf_nodes, queue
                )
                self.logger.log_failed_partition(node.id)
                self.logger.log_queue(queue)
                self.logger.log_n_nodes(
                    len(internal_nodes) + len(leaf_nodes) + len(queue)
                )
                continue
            for child in children:
                if child.is_leaf():
                    leaf_nodes[child.id] = child
                    self.logger.log_leaf_node(child.id)
                else:
                    queue.append(child)
            internal_nodes[node.id] = node
            self.logger.log_internal_node(node.id)
        del internal_nodes[0]  # Removes root from dictionary
        return internal_nodes, leaf_nodes

    def sample_node(self, node: SHPNode) -> list[SHPNode]:
        """
        Generate children partitions of a region contained by [node].

        Args:
            node: (SHPnode) Node to be samples

        Returns: A flattened list of child regions from multiple partitions.

        """
        t_init = time.thread_time()
        subregion_df = self.demo_df.get_subregion_df(node.subregion)
        shape_subregion_df = self.shape_df.get_subregion_df(node.subregion)

        sample_ix = (
            node.get_n_samples()
        )  # Number of successful samples so far / index of next sample to be taken

        max_sample_ix = (
            1 + sample_ix if node.is_root() else self.config.n_samples
        )
        if not isinstance(max_sample_ix, int):
            max_sample_ix = int(
                (max_sample_ix // 1) + (random.random() < max_sample_ix % 1)
            )
        # max_sample_ix - sample_ix is the number of samples this
        # execution of sample_node() will take

        trial_id = 0  # Number of sample attempts so far
        children = []  # List of all child nodes created
        while (
            sample_ix < max_sample_ix
            and trial_id < self.config.max_sample_tries
        ):
            child_sizes = node.sample_n_splits_and_child_sizes(self.config)
            child_centers = OrderedDict(
                self.select_centers(
                    subregion_df, shape_subregion_df, child_sizes
                )
            )
            sample = []  # List of child nodes from the current sample

            partition_ips = ["base"]
            if (
                self.config.final_partition_range is not None
                and node.n_districts in self.config.final_partition_range
            ):
                partition_ips = self.config.final_partition_ips

            warm_start = None
            for ip_str in partition_ips:
                child_nodes, warm_start = self.make_partition(
                    subregion_df,
                    node,
                    child_centers,
                    self.tree.max_root_partition_id,
                    sample_ix,
                    ip_str,
                    warm_start=warm_start,
                )
                if not self.config.use_warm_starts:
                    warm_start = None
                sample += child_nodes

            if len(sample) > 0:
                children += sample
                sample_ix += 1
            trial_id += 1
        self.logger.log_partition(node.id, children, trial_id, t_init)
        return children

    def make_partition(
        self,
        subregion_df: DemoDataFrame,
        node: SHPNode,
        child_centers: dict[int, int],
        root_partition_id: int,
        sample_ix: int,
        ip_str: IPStr,
        warm_start: Optional[dict[int, dict[int, float]]] = None,
    ) -> tuple[list[SHPNode], Optional[dict[int, dict[int, float]]]]:
        """
        Using a random seed, attempt one partition from a sample tree node.
        If successful, returns a list of nodes for each subregion in the
        partition and a dict storing warm start information for following
        partitions using the same centers. If not successful, returns an
        empty list and None.
        Args:
            subregion_df: (DemoDataFrame) subregion demographic dataframe
            node: (SHPnode) the node to sample from
            child_centers: (dict[int, int]) dict of center : n_districts
            root_partition_id: (int) root partition id
            sample_ix: (int) sample index
            ip_str: (IPStr) ip name
            warm_start: (Optional[dict[int, dict[int, float]]]) None or
                a dict storing center : cgu : corresponding ip districts
                variable value

        Returns: (list) of nodes for each subregion in the partition


        """
        t_init = time.thread_time()
        ip_setup = IPSetup(
            self.config,
            subregion_df,
            self.G,
            self.lengths,
            node,
            child_centers,
            self.ideal_pop,
        )
        partition_ip, xs = ip_setup.make(ip_str)

        if warm_start is not None:
            districts = xs["districts"]
            maj_cvap = xs["maj_cvap"]
            prods = xs["prods"]
            for center in districts:
                center_df = subregion_df.loc[districts[center].keys()]
                is_maj_cvap = (
                    center_df[self.config.col].sum() / center_df["VAP"].sum()
                    > 0.5
                )
                maj_cvap[center].Start = is_maj_cvap
                for cgu in districts[center]:
                    districts[center][cgu].Start = warm_start[center][cgu]
                    prods[center][cgu].Start = (
                        warm_start[center][cgu] * is_maj_cvap
                    )
            partition_ip.update()

        partition_ip.optimize()
        partition_time = time.thread_time() - t_init
        districts = xs["districts"]
        try:
            districting = {
                i: [j for j in districts[i] if districts[i][j].X > 0.5]
                for i in child_centers
            }
            feasible = all(
                [
                    nx.is_connected(self.G.get_subgraph(subregion))
                    for subregion in districting.values()
                ]
            )
            if not feasible:
                print("WARNING: PARTITION NOT CONNECTED")
        except AttributeError:
            feasible = False
        if feasible:
            child_nodes = [
                SHPNode(
                    child_centers[center],
                    subregion,
                    self.tree.assign_id(),
                    node.id,
                    center,
                )
                for center, subregion in districting.items()
            ]
            self.tree.n_successful_partitions[root_partition_id] += 1
            status = partition_ip.Status
            ip_info = (
                partition_time,
                status,
                partition_ip.getObjective().getValue(),
            )
            node.save_branch(
                sample_ix, ip_str, [child.id for child in child_nodes], ip_info
            )
            if status == 2:
                self.tree.n_optimal_partitions[root_partition_id] += 1
            elif status == 9:
                self.tree.n_time_limit_partitions[root_partition_id] += 1
            return (
                child_nodes,
                ip_setup.get_warm_start(xs),
            )
        else:
            node.n_infeasible_partitions += 1
            self.tree.n_infeasible_partitions[root_partition_id] += 1
            return ([], None)

    def select_centers(
        self,
        subregion_df: DemoDataFrame,
        shape_subregion_df: ShapeDataFrame,
        child_sizes: list[int],
    ) -> dict[int, int]:
        """
        Routes arguments to the right seed selection function.
        Args:
            subregion_df: (DataFrame) Subset of rows of demo_df of the node region
            child_sizes: (int list) Capacity of the child regions
        Returns: (dict) {center index: # districts assigned to that center}

        """
        method = self.config.selection_method
        if method == "random_method":
            key = random.random()
            if key < 0.5:
                method = "random_iterative"
            elif key < 0.95:
                method = "uncapacitated_kmeans"
            else:
                method = "uniform_random"

        if method == "random_iterative":
            centers = iterative_random(
                subregion_df, len(child_sizes), self.lengths
            )
        elif method == "capacitated_random_iterative":
            pop_capacity = self.ideal_pop * np.array(child_sizes)
            centers = iterative_random(subregion_df, pop_capacity, self.lengths)
        elif method == "uncapacitated_kmeans":
            weight_perturbation_scale = self.config.perturbation_scale
            n_random_seeds = self.config.n_random_seeds
            centers = kmeans_seeds(
                subregion_df,
                len(child_sizes),
                n_random_seeds,
                weight_perturbation_scale,
            )
        elif method == "uniform_random":
            centers = uniform_random(subregion_df, len(child_sizes))
        else:
            raise ValueError("center selection_method not valid")

        center_capacities = get_capacities(
            centers, child_sizes, subregion_df, shape_subregion_df, self.config
        )

        return center_capacities

    def beta_reoptimize(
        self,
        solution_nodes: dict[int, SHPNode],
        internal_nodes: dict[int, SHPNode],
        root_partition_id: int,
        ip_str: IPStr,
        col: VapCol,
        n_steps: int,  # TODO: maybe get rid of this and pull from config?
    ) -> dict[int, SHPNode]:
        t_init = time.thread_time()
        # Save initial config
        ip_timeout = self.config.ip_timeout
        self.config.ip_timeout = None
        initial_beta = self.config.beta
        # Create dict parent_information of parent_id : tuple of information,
        # where the tuple stores the following information at each index:
        # 0 -> parent_node
        # 1 -> center dict[int, int] of centers of the current best partition
        # 2 -> lower bound of beta for the parent node
        # 3 -> upper bound of beta for the parent node
        # 4 -> number of majority cvap districts in the current best partition
        # 5 -> list of leaf nodes of the current best partition
        parent_information = dict()
        new_solution_nodes = dict()

        for leaf_id, leaf_node in solution_nodes.items():
            parent_id = leaf_node.parent_id
            if parent_id in parent_information:
                parent_information[parent_id][1].update({leaf_node.center: 1})
                parent_information[parent_id][5].append(leaf_node)
                parent_information[parent_id][4] += int(
                    leaf_node.is_maj_cvap(col, self.demo_df)
                )
            else:
                parent_node = (
                    self.tree.root
                    if parent_id == 0
                    else internal_nodes[parent_id]
                )
                if parent_node.n_districts in self.config.final_partition_range:
                    parent_information[parent_id] = [
                        parent_node,
                        {leaf_node.center: 1},
                        initial_beta,
                        1,
                        int(leaf_node.is_maj_cvap(col, self.demo_df)),
                        [leaf_node],
                    ]
                else:
                    new_solution_nodes[leaf_id:leaf_node]

        # Beta reoptimize each of the parent nodes for n_steps rounds each,
        # and then save the leaf_nodes from the best partition over those rounds
        for parent_id, information in parent_information.items():
            for _ in range(n_steps):
                parent_node = information[0]
                self.config.beta = (information[2] + information[3]) / 2
                child_nodes, _ = self.make_partition(
                    self.demo_df.get_subregion_df(parent_node.subregion),
                    parent_node,
                    information[1],
                    root_partition_id,
                    parent_node.get_n_samples(),
                    ip_str,
                )
                n_maj_cvap = sum(
                    int(node.is_maj_cvap(col, self.demo_df))
                    for node in child_nodes
                )
                if n_maj_cvap < information[4]:
                    parent_information[parent_id][3] = self.config.beta
                else:
                    parent_information[parent_id][2] = self.config.beta
                    parent_information[parent_id][4] = n_maj_cvap
                    parent_information[parent_id][5] = child_nodes
            new_solution_nodes.update(
                {
                    leaf_node.id: leaf_node
                    for leaf_node in parent_information[parent_id][5]
                }
            )

        self.config.ip_timeout = ip_timeout
        self.config.beta = initial_beta
        self.logger.print_beta_reopt_time(ip_str, time.thread_time() - t_init)
        return new_solution_nodes

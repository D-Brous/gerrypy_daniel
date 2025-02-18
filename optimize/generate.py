import time
import numpy as np
import networkx as nx
from typing import Optional
from collections import OrderedDict
import os
import sys

sys.path.append(".")
from data.config import SHPConfig
from data.demo_df import DemoDataFrame
from data.shape_df import ShapeDataFrame
from data.partition import Partitions
from data.graph import Graph
from optimize.center_selection import *
from constants import VapCol, IPStr, flatten
from optimize.ip import IPSetup
from optimize.tree import SHPNode, SHPTree
from optimize.logging import Logger
from analyze.feasibility import check_feasibility


class ColumnGenerator:
    ROOT_PARTITION_FAILURE = "Root partition failed"
    """
    Generates columns with the Stochastic Hierarchical Paritioning algorithm.
    Maintains samples tree and logging data.
    """

    def __init__(
        self,
        config: SHPConfig,
        tree: Optional[SHPTree] = None,
    ):
        self.config = config
        if tree is None:
            tree = SHPTree(self.config)
        self.tree = tree
        self.logger = Logger(False, False, "")

        self.demo_df = DemoDataFrame.from_config(config)
        self.G = Graph.get_cgu_adjacency_graph(config)
        self.shape_df = ShapeDataFrame.from_config(config)
        self.lengths = self.shape_df.get_lengths(config)
        self.lengths /= 1000  # TODO: get rid of?

        self.partitions = Partitions()
        if self.config.n_beta_reoptimize_steps > 0:
            self.beta_reopt_partitions = Partitions()

        self.save_path = config.get_save_path()
        self.ideal_pop = self.demo_df.get_ideal_pop(config.n_districts)
        self.max_pop_variation = self.ideal_pop * config.population_tolerance

        self.root = tree.get_root()

        self.logger.log_root()

    def run_shp(
        self,
        save_config: bool = False,
        save_tree: bool = False,
        save_partitions: bool = False,
        logging: bool = False,
        printing: bool = False,
        run_beta_reoptimization: bool = False,
    ):
        """Performs generation and finds solutions for each root partition.
        Also performs beta reoptimization if specified.
        """
        save_path = self.save_path
        config_file_path = os.path.join(save_path, "config.json")
        tree_file_path = os.path.join(save_path, "tree.pickle")
        partitions_path = os.path.join(save_path, "partitions")
        debug_file_path = os.path.join(save_path, "debug.txt")
        if save_config or save_tree or save_partitions:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            elif (
                os.path.exists(config_file_path)
                or os.path.exists(tree_file_path)
                or os.path.exists(partitions_path)
            ):
                raise NameError(
                    (
                        "The current save path has already been used. "
                        "Please give the save path an unused name."
                    )
                )
        if save_partitions:
            os.mkdir(partitions_path)
        if save_config:
            self.config.save_to(config_file_path)

        self.logger = Logger(logging, printing, debug_file_path)
        self.logger.print_generation_initiation()

        for root_partition_id in range(self.config.n_root_samples):
            internal_nodes, _ = self.generate_root_partition_tree()
            if save_tree:
                self.tree.save_to(tree_file_path)

            n_final_ips = len(self.config.final_partition_ips)
            for ip_ix, ip_str in enumerate(self.config.final_partition_ips):
                ip_leaf_nodes = self.tree.get_leaf_nodes_from_ip(
                    root_partition_id, ip_str
                )
                solution_nodes = self.tree.get_solution_nodes_dp(
                    internal_nodes,
                    ip_leaf_nodes,
                    root_partition_id,
                    ip_str,
                    self.config.col,
                )
                partition_id = n_final_ips * root_partition_id + ip_ix
                partition = self.tree.get_partition(solution_nodes)
                self.partitions.set_plan(partition_id, partition)

                if (
                    run_beta_reoptimization
                    and self.config.n_beta_reoptimize_steps > 0
                ):
                    beta_reopt_solution_nodes = self.beta_reoptimize(
                        solution_nodes,
                        internal_nodes,
                        root_partition_id,
                        ip_str,
                        self.config.col,
                        self.config.n_beta_reoptimize_steps,
                    )
                    beta_reopt_partition = self.tree.get_partition(
                        beta_reopt_solution_nodes
                    )
                    self.beta_reopt_partitions.set_plan(
                        partition_id, beta_reopt_partition
                    )
                if save_partitions:
                    self.partitions.to_csv(
                        os.path.join(partitions_path, "shp.csv"),
                        self.config,
                    )
                    if (
                        run_beta_reoptimization
                        and self.config.n_beta_reoptimize_steps > 0
                    ):
                        self.beta_reopt_partitions.to_csv(
                            os.path.join(
                                partitions_path,
                                "shp_br.csv",
                            ),
                            self.config,
                        )

        self.logger.log_generation_completion(self.tree)
        self.logger.print_generation_completion(
            self.config, self.tree, self.partitions
        )

        check_feasibility(self.config, self.partitions, self.demo_df, self.G)
        self.logger.print_feasible("\nOriginal partitions")
        if run_beta_reoptimization and self.config.n_beta_reoptimize_steps > 0:
            check_feasibility(
                self.config, self.beta_reopt_partitions, self.demo_df, self.G
            )
            self.logger.print_feasible("Beta reoptimized partitions")
        self.logger.close()

    def run_beta_reopt(
        self,
        save_partitions: bool = False,
        logging: bool = False,
        printing: bool = False,
    ):
        if self.config.n_beta_reoptimize_steps == 0:
            return
        save_path = self.save_path
        csv_path = os.path.join(save_path, "partitions", "shp_br.csv")
        debug_file_path = os.path.join(save_path, "debug.txt")
        if save_partitions:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            elif os.path.exists(csv_path):
                raise NameError(
                    (
                        "Beta reopt partitions were already saved in the"
                        " given save path."
                    )
                )
        self.logger = Logger(logging, printing, debug_file_path)
        for root_partition_id in range(self.config.n_root_samples):
            internal_nodes = self.tree.get_internal_nodes(root_partition_id)
            n_final_ips = len(self.config.final_partition_ips)
            for ip_ix, ip_str in enumerate(self.config.final_partition_ips):
                ip_leaf_nodes = self.tree.get_leaf_nodes_from_ip(
                    root_partition_id, ip_str
                )
                solution_nodes = self.tree.get_solution_nodes_dp(
                    internal_nodes,
                    ip_leaf_nodes,
                    root_partition_id,
                    ip_str,
                    self.config.col,
                )
                partition_id = n_final_ips * root_partition_id + ip_ix
                beta_reopt_solution_nodes = self.beta_reoptimize(
                    solution_nodes,
                    internal_nodes,
                    root_partition_id,
                    ip_str,
                    self.config.col,
                    self.config.n_beta_reoptimize_steps,
                )
                beta_reopt_partition = self.tree.get_partition(
                    beta_reopt_solution_nodes
                )
                self.beta_reopt_partitions.set_plan(
                    partition_id, beta_reopt_partition
                )
                if save_partitions:
                    self.beta_reopt_partitions.to_csv(
                        csv_path,
                        self.config,
                    )

    def generate_root_partition_tree(
        self,
    ) -> tuple[dict[int, SHPNode], dict[int, SHPNode]]:
        t_init = time.thread_time()
        self.logger.print_new_root_partition(self.tree.max_root_partition_id)
        trial_id = 0
        while trial_id < self.config.max_sample_tries:
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
                error_str = str(error)
                self.logger.print_failed_root_partition(str(error_str))
                self.logger.log_failed_root_partition(str(error_str))
                if error_str != ColumnGenerator.ROOT_PARTITION_FAILURE:
                    self.root.delete_sample(self.tree.max_root_partition_id)
                self.tree.failed_root_samples += 1
                trial_id += 1
        raise RuntimeError("Failed to generate tree")

    def attempt_generation(
        self,
    ) -> tuple[dict[int, SHPNode], dict[int, SHPNode]]:
        leaf_nodes = {}
        internal_nodes = {}
        queue = [self.root]
        while len(queue) > 0:
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
            raise RuntimeError(ColumnGenerator.ROOT_PARTITION_FAILURE)
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
        partition_ips = ["base"]
        if node.n_districts <= self.config.final_partition_threshold:
            partition_ips = self.config.final_partition_ips

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
        n_steps: int,
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
                if (
                    parent_node.n_districts
                    <= self.config.final_partition_threshold
                ):
                    parent_information[parent_id] = [
                        parent_node,
                        {leaf_node.center: 1},
                        initial_beta,
                        1,
                        int(leaf_node.is_maj_cvap(col, self.demo_df)),
                        [leaf_node],
                    ]
                else:
                    new_solution_nodes[leaf_id] = leaf_node

        # Beta reoptimize each of the parent nodes for n_steps rounds each,
        # and then save the leaf_nodes from the best partition over those rounds
        for parent_id, information in parent_information.items():
            parent_node = information[0]
            parent_subregion_df = self.demo_df.get_subregion_df(
                parent_node.subregion
            )
            self.config.beta = 1
            child_nodes, _ = self.make_partition(
                parent_subregion_df,
                parent_node,
                information[1],
                root_partition_id,
                parent_node.get_n_samples(),
                ip_str,
            )
            n_maj_cvap = sum(
                int(node.is_maj_cvap(col, self.demo_df)) for node in child_nodes
            )
            if n_maj_cvap < information[4]:
                for _ in range(n_steps):
                    self.config.beta = (information[2] + information[3]) / 2
                    child_nodes, _ = self.make_partition(
                        parent_subregion_df,
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
            else:
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

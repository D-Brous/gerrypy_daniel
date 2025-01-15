import os
import sys

sys.path.append(".")
from data.df import DemoDataFrame
from data.config import SHPConfig
from data.partition import Partitions
from optimize.generate import ColumnGenerator
from optimize.logging import Logger
from optimize.tree import SHPTree


class SHP:
    """
    SHP class to test different generation configurations. Supports the
    original SHP algorithm as well as the ability to run one or more
    majority cvap maximizing partition ips at the desired level of the
    tree, and the ability to take a solution set of leaf nodes and
    reoptimize the partitions that led to them in order to get the same
    numbers of majority cvap districts but with higher compactness scores
    """

    def __init__(self, config: SHPConfig):
        """Initialize using the shp config

        Args:
            config (SHPConfig): config class with all the parameters
                needed to perform the algorithms mentioned above
        """
        self.config = config
        self.save_path = self.config.save_path
        self.demo_df = DemoDataFrame.from_config(config)

        # self.n_cgus = len(self.demo_df["GEOID"])
        self.partitions = Partitions()
        if self.config.n_beta_reoptimize_steps > 0:
            self.beta_reopt_partitions = Partitions()

    def shp(
        self,
        save_config=True,
        save_tree=True,
        save_partitions=True,
        logging=True,
        printing=True,
    ):
        """Performs generation and finds solutions for each root partition.
        Also performs beta reoptimization if specified.
        """
        config_file_path = os.path.join(self.save_path, "config.json")
        tree_file_path = os.path.join(self.save_path, "tree.pickle")
        partitions_path = os.path.join(self.save_path, "partitions")
        debug_file_path = os.path.join(self.save_path, "debug.txt")
        if save_config or save_tree or save_partitions:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
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
            # TODO: fix this to allow config to be saved
            # self.config.save_to(config_file_path)

        tree = SHPTree(self.config)
        logger = Logger(logging, printing, debug_file_path)
        cg = ColumnGenerator(self.config, tree, logger)

        logger.print_generation_initiation()

        for root_partition_id in range(self.config.n_root_samples):
            internal_nodes, leaf_nodes = cg.generate_root_partition_tree()
            if save_tree:
                tree.save_to(tree_file_path)

            n_final_ips = len(self.config.final_partition_ips)
            for ip_ix, ip_str in enumerate(self.config.final_partition_ips):
                ip_leaf_nodes = tree.get_leaf_nodes_from_ip(
                    ip_str, root_partition_id
                )
                solution_nodes = tree.get_solution_nodes_dp(
                    internal_nodes,
                    ip_leaf_nodes,
                    root_partition_id,
                    self.config.col,
                )
                partition_id = n_final_ips * root_partition_id + ip_ix
                partition = tree.get_partition(solution_nodes)
                self.partitions.set(partition_id, partition)

                if self.config.n_beta_reoptimize_steps > 0:
                    beta_reopt_solution_nodes = cg.beta_reoptimize(
                        solution_nodes,
                        internal_nodes,
                        root_partition_id,
                        ip_str,
                        self.config.col,
                        self.config.n_beta_reoptimize_steps,
                    )
                    beta_reopt_partition = tree.get_partition(
                        beta_reopt_solution_nodes
                    )
                    self.beta_reopt_partitions.set(
                        partition_id, beta_reopt_partition
                    )
                if save_partitions:
                    self.partitions.to_csv(
                        os.path.join(partitions_path, "shp.csv"),
                        self.config,
                    )
                    if self.config.n_beta_reoptimize_steps > 0:
                        self.beta_reopt_partitions.to_csv(
                            os.path.join(
                                partitions_path,
                                "shp_beta_reopt.csv",
                            ),
                            self.config,
                        )

        logger.log_generation_completion(tree)
        logger.print_generation_completion(self.config, tree, self.partitions)
        logger.close()
        # TODO: put in a solution feasibility check

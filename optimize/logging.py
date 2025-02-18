import time
import numpy as np
import sys

sys.path.append(".")
from constants import IPStr
from data.config import SHPConfig
from data.demo_df import DemoDataFrame
from data.partition import Partitions
from optimize.tree import SHPNode, SHPTree
from analyze.maj_min import n_maj_cvap

# TODO clean up this file a bit


class Logger:
    def __init__(self, logging: bool, printing: bool, debug_file_path: str):
        self.logging = logging
        self.printing = printing
        self.buffer = []
        if self.logging:
            self.file = open(debug_file_path, "w")

    def close(self):
        if self.logging:
            self.file.close()

    def if_logging(func):
        def modified_func(self, *args, **kwargs):
            if self.logging:
                return self.file.write(func(self, *args, **kwargs))

        return modified_func

    def if_printing(func):
        def modified_func(self, *args, **kwargs):
            if self.printing:
                print(func(self, *args, **kwargs))

        return modified_func

    @if_printing
    def print_new_root_partition(self, root_partition_id):
        return f"\n---------------Generating root partition number {root_partition_id}---------------"

    @if_logging
    def log_root(self):
        return f"Logged root node 0 at time {time.thread_time()}\n"

    @if_logging
    def log_internal_node(self, id: int):
        return f"Logged internal node {id} at time {time.thread_time()}\n"

    @if_logging
    def log_leaf_node(self, id: int):
        return f"Logged leaf node {id} at time {time.thread_time()}\n"

    def log_node_deletion(self, id: int):
        if self.logging:
            self.buffer.append(id)

    @if_logging
    def log_failed_partition(self, id: int):
        buffer = self.buffer
        self.bugger = []
        return f"Failed split of node {id}.\n    Deleted nodes:\n    {buffer}"

    @if_logging
    def log_failed_root_partition(self, error_str: str):
        return f"\n-------------------------------Root partition failed ({error_str})-------------------------------"

    @if_printing
    def print_failed_root_partition(self, error_str: str):
        return f"\n-------------------------------Root partition failed ({error_str})-------------------------------"

    @if_logging
    def log_queue(self, queue: list[SHPNode]):
        queue_ids = [node.id for node in queue]
        return f"\n    Remaining sample queue:\n    {queue_ids}"

    @if_logging
    def log_n_nodes(self, n_nodes: int):
        return (
            f"\n    Total number of nodes in tree after deletion: {n_nodes}\n"
        )

    @if_logging
    def log_partition(
        self, id: int, children: list[SHPNode], n_trials: int, t_init: float
    ):
        str = f"Split node {id} with {n_trials} trials in {time.thread_time()-t_init} sec. Children are:\n"
        for child in children:
            str += self.log_child(child)
        return str

    def log_child(self, child: SHPNode):
        if child.n_districts == 1:
            return f"    {child.id}, leaf, {child.n_districts}, {len(child.subregion)}\n"
        else:
            return f"    {child.id}, internal, {child.n_districts}, {len(child.subregion)}\n"

    @if_printing
    def print_completed_root_partition(
        self, root_partition_id: int, tree: SHPTree
    ):
        return (
            f"\nGeneration time: {tree.generation_times[root_partition_id]:0.2f}\n"
            + f"Number of deleted nodes: {tree.n_deleted_nodes[root_partition_id]}\n"
            + f"Number of infeasible partitions: {tree.n_infeasible_partitions[root_partition_id]}\n"
            + f"Number of successful partitions: {tree.n_successful_partitions[root_partition_id]}\n"
            + f"Number of partitions that found an optimal solution: {tree.n_optimal_partitions[root_partition_id]}\n"
            + f"Number of partitions that reached their time limit: {tree.n_time_limit_partitions[root_partition_id]}\n"
        )

    @if_printing
    def print_beta_reopt_time(self, ip_str: IPStr, time: float):
        return f"Total beta reoptimization time for {ip_str}: {time:0.2f}\n"

    @if_printing
    def print_generation_initiation(self):
        return "\n<><><><><><><><><><><> SHP Algorithm Start <><><><><><><><><><><>\n"

    @if_logging
    def log_generation_completion(self, tree: SHPTree):
        return (
            "\n-------------------------------------------\n"
            + f"Number of internal nodes: {tree.n_internal_nodes()}\n"
            + f"Number of leaf nodes: {tree.n_leaf_nodes()}\n"
        )

    @if_printing
    def print_generation_completion(
        self, config: SHPConfig, tree: SHPTree, partitions: Partitions
    ):
        string = (
            "\n<><><><><><><><><><><>  SHP Algorithm End  <><><><><><><><><><><>\n\n"
            + f"Number of internal nodes: {tree.n_internal_nodes()}\n"
            + f"Number of leaf nodes: {tree.n_leaf_nodes()}\n"
            + f"Tree generation times: {tree.generation_times.astype('int32')}\n"
            + f"--> Total generation time: {np.sum(tree.generation_times):0.2f}\n"
            + f"Number of districtings = {tree.n_districtings()}\n"
            + f"Numbers of deleted nodes: {tree.n_deleted_nodes}\n"
            + f"Numbers of infeasible partitions: {tree.n_infeasible_partitions}\n"
            + f"Numbers of successful partitions: {tree.n_successful_partitions}\n"
            + f"Numbers of partitions that found an optimal solution: {tree.n_optimal_partitions}\n"
            + f"Numbers of partitions that reached their time limit: {tree.n_time_limit_partitions}\n"
            + f"Number of failed root samples: {tree.failed_root_samples}\n"
        )
        n_final_ips = len(config.final_partition_ips)
        demo_df = DemoDataFrame.from_config(config)
        n_partitions = config.n_root_samples * n_final_ips
        n_maj_cvap_arr = np.zeros((n_partitions), dtype=int)
        for partition_id in range(n_partitions):
            partition = partitions.get_plan(partition_id)
            n_maj_cvap_arr[partition_id] = n_maj_cvap(
                config.col, partition, demo_df
            )
        for ip_ix, ip_str in enumerate(config.final_partition_ips):
            string += f"Numbers of maj {config.col} districts using {ip_str}: {n_maj_cvap_arr[ip_ix::n_final_ips]}\n"
        return string

    @if_printing
    def print_feasible(self, partitions_str: str):
        return f"{partitions_str} are feasible\n"

import numpy as np
import os
import heapq
import time
from typing import Optional
import copy
import sys

sys.path.append(".")
from constants import VapCol
from data.config import SHPConfig, PostProcessorConfig
from data.graph import Graph
from data.demo_df import DemoDataFrame
from data.partition import Partition, Partitions
from optimize.shp import SHP
from optimize.generate import ColumnGenerator
from analyze.maj_min import n_maj_cvap, cvap_props


class PriorityFuncSetup:
    """
    Class for running priority functions, which take in a partition of a
    subregion and output a float representing how much the post processor
    should prioritize repartitioning that subregion. If the priority value
    returned is negative, then that region should not be repartitioned.
    """

    def __init__(self, col: VapCol, demo_df: DemoDataFrame, delta: float):
        self.col = col
        self.demo_df = demo_df
        self.delta = delta

    def average_geq_threshold_excess(self, subpartition: Partition) -> float:
        subpartition_cvap_props = cvap_props(
            self.col, subpartition, self.demo_df
        )
        n_maj_cvap = sum(
            int(cvap_prop > 0.5) for cvap_prop in subpartition_cvap_props
        )
        if n_maj_cvap == subpartition.n_districts:
            return -1
        else:
            return sum(subpartition_cvap_props) - (0.5 + self.delta) * (
                n_maj_cvap + 1
            )


class PartitionFuncSetup:
    """
    Class for running partition functions, which take in a partition of a
    subregion and output one or more new partitions. The number and
    interpretation of such partitions may vary, but by convention the
    partition with id = 0 is the partition which will represent the final
    repartitioning and the other partitions are just intermediary.
    """

    def __init__(
        self,
        shp_config: SHPConfig,
    ):
        self.shp_config = copy.deepcopy(shp_config)
        self.shp_config.n_root_samples = 1
        self.shp_config.ip_timeout = None
        self.cg = None

    def maj_cvap_ip(self, subpartition: Partition) -> Partition:
        # TODO: potentially make this do multiple root samples and then grab the best partition, b/c right now it just does 1, and the centers chosen might just be suboptimal
        self.shp_config.subregion = subpartition.get_region()
        self.cg = ColumnGenerator(self.shp_config)
        self.cg.run_shp()
        return self.cg.partitions.get_plan(0)

        # shp = SHP(self.shp_config)
        # shp.shp(
        #     save_config=False,
        #     save_tree=False,
        #     save_partitions=False,
        #     logging=False,
        #     printing=False,
        # )
        # partitions = Partitions()
        # if self.shp_config.n_beta_reoptimize_steps == 0:
        #     partitions.set_plan(0, shp.partitions.get_plan(0))
        # else:
        #     partitions.set_plan(0, shp.beta_reopt_partitions.get_plan(0))
        #     partitions.set_plan(1, shp.partitions.get_plan(0))
        # return partitions

    def beta_reopt(self) -> Partition:
        self.cg.run_beta_reopt()
        return self.cg.beta_reopt_partitions.get_plan(0)


class PostProcessor:
    def __init__(
        self,
        shp_config: SHPConfig,
        pp_config: PostProcessorConfig,
    ):
        self.shp_config = shp_config
        self.pp_config = pp_config
        self.save_path = shp_config.get_save_path()
        self.partitions = Partitions.from_csv(
            os.path.join(
                self.save_path,
                "partitions",
                pp_config.loaded_partitions_file_name,
            )
        )
        self.demo_df = DemoDataFrame.from_config(shp_config)
        self.priority_func_setup = PriorityFuncSetup(
            shp_config.col, self.demo_df, pp_config.delta
        )
        if pp_config.priority_func_str == "average_geq_threshold_excess":
            self.get_priority = (
                self.priority_func_setup.average_geq_threshold_excess
            )
        self.partition_func_setup = PartitionFuncSetup(shp_config)
        if pp_config.partition_func_str == "maj_cvap_ip":
            self.repartition = self.partition_func_setup.maj_cvap_ip
        self.n_successful_repartitions = dict()
        self.n_failed_repartitions = dict()

    def get_csv_path(self, plan_id):
        csv_path = os.path.join(
            self.save_path,
            "partitions",
            "_".join(
                (
                    self.pp_config.loaded_partitions_file_name[:-4],
                    f"p{plan_id}",
                    self.pp_config.saved_partitions_file_name,
                )
            ),
        )
        # if os.path.exists(csv_path):
        #     raise ValueError(
        #         "Partitions file with the given name already exists. \
        #             Please provide a different file name."
        #     )
        return csv_path

    """
    def get_priority(self, district_subregions: list[list[int]]) -> float:
        if self.pp_config.priority_func_str == :
            return self.priority_func_setup.average_geq_threshold_excess(
                district_subregions
            )

    def repartition(self, subpartition: Partition) -> Partitions:
        return self.partition_func(subpartition)
    """

    def update_pq(
        self,
        pq: list[tuple[int, tuple[int]]],
        partition: Partition,
        district_ids: tuple[int],
    ):
        priority = self.get_priority(partition.get_subpartition(district_ids))
        if priority >= 0:
            heapq.heappush(pq, (-1 * priority, district_ids))

    def get_priority_dict(self, partition: Partition, tup_set: set[tuple[int]]):
        tups_with_priorities = {
            (tup, self.get_priority(partition.get_subpartition(tup)))
            for tup in tup_set
        }
        return {
            tup: priority
            for (tup, priority) in tups_with_priorities
            if priority >= 0
        }

    def get_priority_dict_2(
        self, partition: Partition, tup_set: set[tuple[int]]
    ) -> dict[tuple[int], float]:
        return {
            tup: self.get_priority(partition.get_subpartition(tup))
            for tup in tup_set
            if self.get_priority(partition.get_subpartition(tup)) >= 0
        }

    def get_priority_queue(self, priority_dict: dict[tuple[int], float]):
        # Tups are returned in increasing order of priority so that popping the highest priority element is O(1)
        tups = list(priority_dict.keys())
        tups.sort(key=lambda tup: priority_dict[tup])
        return tups

    def print_begin_post_processing(self, plan_id: int):
        print(
            f"\n\n---------------Post processing plan {plan_id}---------------\n"
        )

    def print_summary(
        self, plan_id: int, partitions: Partitions, n_plans: int, time: float
    ):
        n_maj_cvap_arr = np.zeros((n_plans), dtype=int)
        for plan in range(n_plans):
            partition = partitions.get_plan(plan)
            n_maj_cvap_arr[plan] = n_maj_cvap(
                self.shp_config.col, partition, self.demo_df
            )
        print(f"Post processing time: {time}")
        print(
            f"Numbers of maj {self.shp_config.col} districts: {n_maj_cvap_arr}"
        )
        print(
            f"Total number of repartitioned subregions: {self.n_successful_repartitions[plan_id]}"
        )
        print(
            f"Total number of failed repartitions: {self.n_failed_repartitions[plan_id]}"
        )

    """
    def shp_repartition(self, subpartition: Partition):
        self.shp_config.subregion = subpartition.get_region()
        n_beta_reoptimize_steps = self.shp_config.n_beta_reoptimize_steps
        self.shp_config.n_beta_reoptimize_steps = 0
        shp = SHP(self.shp_config)
        shp.shp(
            save_config=False,
            save_tree=False,
            save_partitions=False,
            logging=False,
            printing=False,
        )
        partitions = Partitions()
        if self.shp_config.n_beta_reoptimize_steps == 0:
            partitions.set_plan(0, shp.partitions.get_plan(0))
        else:
            partitions.set_plan(0, shp.beta_reopt_partitions.get_plan(0))
            partitions.set_plan(1, shp.partitions.get_plan(0))
        return partitions
    """

    def priority_n_opt(self, n: int, plan_id: int):
        """
        For the plan given by plan_id, add all connected n-tuples of
        districts in the district adjacency graph to a priority queue
        according to the priority_func. Reoptimize in decreasing order
        of priorty. If we have an improvement in the number of maj cvap
        districts, update the priorty queue with all neighboring
        n-tuples and change it. Returns partitions for each improvement.
        """
        csv_path = self.get_csv_path(plan_id)
        t_init = time.thread_time()
        self.print_begin_post_processing(plan_id)
        self.n_successful_repartitions[plan_id] = 0
        self.n_failed_repartitions[plan_id] = 0
        self.partition_func_setup.shp_config.n_districts = n
        self.partition_func_setup.shp_config.final_partition_threshold = n
        saved_partitions = Partitions()
        partition = self.partitions.get_plan(plan_id)
        G_d = Graph.get_district_adjacency_graph(self.shp_config, partition)
        n_tups = get_connected_n_tuples(G_d, n)
        pq = []
        deleted_tups = set()
        n_saved = 0
        for chosen_districts in n_tups:
            self.update_pq(pq, partition, chosen_districts)
        step = 0
        while len(pq) > 0:
            chosen_districts = heapq.heappop(pq)[1]
            if chosen_districts in deleted_tups:
                continue
            curr_subpartition = partition.get_subpartition(chosen_districts)
            try:
                new_subpartition = self.repartition(curr_subpartition)
            except:  # This should only happen if the region is not able to be partitioned after how ever many root sample tries
                self.n_failed_repartitions[plan_id] += 1
                continue
            self.n_successful_repartitions[plan_id] += 1
            if n_maj_cvap(
                self.shp_config.col, new_subpartition, self.demo_df
            ) > n_maj_cvap(
                self.shp_config.col, curr_subpartition, self.demo_df
            ):
                if self.shp_config.n_beta_reoptimize_steps > 0:
                    partition = copy.deepcopy(partition)
                    partition.update_via_subpartition(
                        new_subpartition, chosen_districts
                    )
                    saved_partitions.set_plan(n_saved, partition)
                    n_saved += 1
                    new_subpartition = self.partition_func_setup.beta_reopt()
                partition = copy.deepcopy(partition)
                partition.update_via_subpartition(
                    new_subpartition, chosen_districts
                )
                saved_partitions.set_plan(n_saved, partition)
                n_saved += 1
                deleted_tups.update(
                    get_connected_n_tuples(
                        G_d, n, intersecting_districts=chosen_districts
                    )
                )
                G_d = Graph.get_district_adjacency_graph(
                    self.shp_config, partition
                )
                for tup in get_connected_n_tuples(
                    G_d, n, intersecting_districts=chosen_districts
                ):
                    self.update_pq(pq, partition, tup)
            step += 1
            print(f"Partition attempts so far: {step}", end="\r")
        saved_partitions.to_csv(csv_path, self.shp_config)
        self.print_summary(
            plan_id, saved_partitions, n_saved, time.thread_time() - t_init
        )

    def priority_n_opt_2(self, n: int, plan_id: int):
        """
        For the plan given by plan_id, add all connected n-tuples of
        districts in the district adjacency graph to a priority queue
        according to the priority_func. Reoptimize in decreasing order
        of priorty. If we have an improvement in the number of maj cvap
        districts, update the priorty queue with all neighboring
        n-tuples and change it. Returns partitions for each improvement.
        """
        csv_path = self.get_csv_path(plan_id)[:-4] + "_2.csv"
        t_init = time.thread_time()
        self.print_begin_post_processing(plan_id)
        self.n_successful_repartitions[plan_id] = 0
        self.n_failed_repartitions[plan_id] = 0
        self.partition_func_setup.shp_config.n_districts = n
        self.partition_func_setup.shp_config.final_partition_threshold = n
        saved_partitions = Partitions()
        partition = self.partitions.get_plan(plan_id)
        G_d = Graph.get_district_adjacency_graph(self.shp_config, partition)
        tup_set = get_connected_n_tuples(G_d, n)
        priority_dict = self.get_priority_dict(partition, tup_set)
        pq = self.get_priority_queue(priority_dict)
        n_saved = 0
        while len(pq) > 0:
            chosen_districts = pq.pop()
            del priority_dict[chosen_districts]
            # if chosen_districts in deleted_tups:
            #    continue
            curr_subpartition = partition.get_subpartition(chosen_districts)
            try:
                new_subpartition = self.repartition(curr_subpartition)
            except:  # This should only happen if the region is not able to be partitioned after how ever many root sample tries
                self.n_failed_repartitions[plan_id] += 1
                continue
            self.n_successful_repartitions[plan_id] += 1
            if n_maj_cvap(
                self.shp_config.col, new_subpartition, self.demo_df
            ) > n_maj_cvap(
                self.shp_config.col, curr_subpartition, self.demo_df
            ):
                if self.shp_config.n_beta_reoptimize_steps > 0:
                    partition.update_via_subpartition(
                        new_subpartition, chosen_districts
                    )
                    saved_partitions.set_plan(n_saved, partition)
                    n_saved += 1
                    new_subpartition = self.partition_func_setup.beta_reopt()
                partition.update_via_subpartition(
                    new_subpartition, chosen_districts
                )
                saved_partitions.set_plan(n_saved, partition)
                n_saved += 1
                for tup in get_connected_n_tuples(
                    G_d, n, intersecting_districts=chosen_districts
                ):
                    if tup in priority_dict:
                        del priority_dict[tup]
                G_d = Graph.get_district_adjacency_graph(
                    self.shp_config, partition
                )
                tups_to_add = get_connected_n_tuples(
                    G_d, n, intersecting_districts=chosen_districts
                )
                priority_dict.update(
                    self.get_priority_dict(partition, tups_to_add)
                )
                pq = self.get_priority_queue(priority_dict)
        saved_partitions.to_csv(csv_path, self.shp_config)
        self.print_summary(
            plan_id, saved_partitions, n_saved, time.thread_time() - t_init
        )

    def priority_n_opt_3(self, n: int, plan_id: int):
        """
        For the plan given by plan_id, add all connected n-tuples of
        districts in the district adjacency graph to a priority queue
        according to the priority_func. Reoptimize in decreasing order
        of priorty. If we have an improvement in the number of maj cvap
        districts, update the priorty queue with all neighboring
        n-tuples and change it. Returns partitions for each improvement.
        """
        csv_path = self.get_csv_path(plan_id)[:-4] + "_3.csv"
        t_init = time.thread_time()
        self.print_begin_post_processing(plan_id)
        self.n_successful_repartitions[plan_id] = 0
        self.n_failed_repartitions[plan_id] = 0
        self.partition_func_setup.shp_config.n_districts = n
        self.partition_func_setup.shp_config.final_partition_threshold = n
        saved_partitions = Partitions()
        partition = self.partitions.get_plan(plan_id)
        G_d = Graph.get_district_adjacency_graph(self.shp_config, partition)
        tup_set = get_connected_n_tuples(G_d, n)
        priority_dict = self.get_priority_dict_2(partition, tup_set)
        pq = self.get_priority_queue(priority_dict)
        n_saved = 0
        while len(pq) > 0:
            chosen_districts = pq.pop()
            del priority_dict[chosen_districts]
            # if chosen_districts in deleted_tups:
            #    continue
            curr_subpartition = partition.get_subpartition(chosen_districts)
            try:
                new_subpartition = self.repartition(curr_subpartition)
            except:  # This should only happen if the region is not able to be partitioned after how ever many root sample tries
                self.n_failed_repartitions[plan_id] += 1
                continue
            self.n_successful_repartitions[plan_id] += 1
            if n_maj_cvap(
                self.shp_config.col, new_subpartition, self.demo_df
            ) > n_maj_cvap(
                self.shp_config.col, curr_subpartition, self.demo_df
            ):
                if self.shp_config.n_beta_reoptimize_steps > 0:
                    partition.update_via_subpartition(
                        new_subpartition, chosen_districts
                    )
                    saved_partitions.set_plan(n_saved, partition)
                    n_saved += 1
                    new_subpartition = self.partition_func_setup.beta_reopt()
                partition.update_via_subpartition(
                    new_subpartition, chosen_districts
                )
                saved_partitions.set_plan(n_saved, partition)
                n_saved += 1
                for tup in get_connected_n_tuples(
                    G_d, n, intersecting_districts=chosen_districts
                ):
                    if tup in priority_dict:
                        del priority_dict[tup]
                G_d = Graph.get_district_adjacency_graph(
                    self.shp_config, partition
                )
                tups_to_add = get_connected_n_tuples(
                    G_d, n, intersecting_districts=chosen_districts
                )
                priority_dict.update(
                    self.get_priority_dict_2(partition, tups_to_add)
                )
                pq = self.get_priority_queue(priority_dict)
        saved_partitions.to_csv(csv_path, self.shp_config)
        self.print_summary(
            plan_id, saved_partitions, n_saved, time.thread_time() - t_init
        )


def insert_unique(element: int, tup: tuple[int], length: int):
    lower = 0
    upper = length - 1
    if element < tup[lower]:
        return (element,) + tup
    if element > tup[upper]:
        return tup + (element,)
    while upper - lower > 1:
        midpoint = (lower + upper) // 2
        if element > tup[midpoint]:
            lower = midpoint
        else:
            upper = midpoint
    return tup[:upper] + (element,) + tup[upper:]


def get_connected_three_tuples(G: Graph):
    triples = set()
    for edge in G.edges:
        edge = tuple(sorted(edge))
        neighbors = set(list(G[edge[0]]) + list(G[edge[1]]))
        neighbors.remove(edge[0])
        neighbors.remove(edge[1])
        for neighbor in neighbors:
            triple = insert_unique(neighbor, edge, 3)
            triples.add(triple)
    return list(triples)


def get_connected_n_tuples(
    G: Graph, n: int, intersecting_districts: Optional[tuple[int]] = None
) -> set[tuple[int]]:
    curr_tups = [tuple(sorted(edge)) for edge in G.edges]
    if intersecting_districts is not None:
        curr_tups = [
            edge
            for edge in curr_tups
            if edge[0] in intersecting_districts
            or edge[1] in intersecting_districts
        ]
    for curr_tup_len in range(2, n):
        next_tups = set()
        for tup in curr_tups:
            neighbors = set(
                [
                    element
                    for i in range(curr_tup_len)
                    for element in list(G[tup[i]])
                ]
            )
            for element in tup:
                neighbors.remove(element)
            for neighbor in neighbors:
                next_tups.add(insert_unique(neighbor, tup, curr_tup_len))
        curr_tups = next_tups
    return curr_tups


'''

def random_two_choice(G):
    return random.sample(G.edges, 1)[0]

def get_unique_plans(assignments_df, first_plan=None):
    unique_df = pd.DataFrame()
    unique_df["GEOID"] = assignments_df["GEOID"]
    curr_plan_id = 0
    if first_plan is not None:
        unique_df[f"District{curr_plan_id}"] = first_plan
        curr_plan_id += 1
    for column in list(assignments_df.columns)[1:]:
        if not unique_df[f"District{curr_plan_id - 1}"].equals(
            assignments_df[column]
        ):
            unique_df[f"District{curr_plan_id}"] = assignments_df[column]
            curr_plan_id += 1
    return unique_df

def average_geq_threshold_random(
    state, year, granularity, epsilon, assignment_ser
):
    state_df = load_state_df(state, year, granularity)
    G_d = load_district_adjacency_graph(
        state, year, granularity, assignment_ser
    )
    assignment_dict = assignment_ser_to_dict(assignment_ser)
    while True:
        district_ids = random_two_choice(G_d)
        district_subregions = [
            assignment_dict[district_id] for district_id in district_ids
        ]
        if average_geq_threshold(district_subregions, epsilon):
            return [
                cgu_ix
                for district_subregion in district_subregions
                for cgu_ix in district_subregion
            ]


def average_geq_threshold(state_df, epsilon, district_subregions):
    bvap_props = [
        bvap_prop(district_subregion, state_df)
        for district_subregion in district_subregions
    ]
    n_black_maj = sum(int(bvap_prop > 0.5) for bvap_prop in bvap_props)
    return n_black_maj != len(district_subregions) and sum(bvap_props) > (
        0.5 + epsilon
    ) * (n_black_maj + 1)

def priority_n_opt_with_alpha_reoptimize_one_region(
        self, n, alpha_reoptimize_steps, tup, n_trials
    ):
        self.config.alpha_reoptimize_steps = alpha_reoptimize_steps
        for plan in self.plans:
            saved_assignments_df = pd.DataFrame()
            saved_assignments_df["GEOID"] = self.state_df["GEOID"]
            curr_assignment = self.assignments_df[f"District{plan}"]
            curr_assignment_alpha_reopt = curr_assignment.copy()
            assignment_dict = assignment_ser_to_dict(
                curr_assignment_alpha_reopt
            )
            G_d = load_district_adjacency_graph(self.config, curr_assignment)
            n_tups = [tup] * n_trials
            pq = []
            deleted_tups = set()
            step = 0
            n_saved = 0
            for chosen_districts in n_tups:
                district_subregions = [
                    assignment_dict[district_id]
                    for district_id in chosen_districts
                ]
                priority = self.priority_func(district_subregions)
                if priority >= 0:
                    heapq.heappush(pq, (-1 * priority, chosen_districts))
            improvement_found = False
            while len(pq) > 0 and (not improvement_found):
                step += 1
                chosen_districts = heapq.heappop(pq)[1]
                if chosen_districts not in deleted_tups:
                    district_subregions = [
                        assignment_dict[district_id]
                        for district_id in chosen_districts
                    ]
                    subregion = [
                        cgu_ix
                        for district_subregion in district_subregions
                        for cgu_ix in district_subregion
                    ]
                    self.config.subregion = subregion
                    self.config.n_districts = n
                    reassignment_df, reassignment_df_alpha_reopt = (
                        self.partition_func(self.config)
                    )
                    reassignment_dict = assignment_ser_to_dict(
                        reassignment_df["District0"]
                    )
                    reassignment_dict_alpha_reopt = assignment_ser_to_dict(
                        reassignment_df_alpha_reopt["District0"]
                    )

                    if n_maj_black(
                        reassignment_dict, self.state_df
                    ) > n_maj_black(
                        {
                            ix: district_subregions[ix]
                            for ix in range(len(district_subregions))
                        },
                        self.state_df,
                    ):
                        improvement_found = True
                        deleted_tups.update(
                            get_connected_n_tuples(
                                G_d, n, intersecting_districts=chosen_districts
                            )
                        )
                        for (
                            chosen_district_ix,
                            district_subregion,
                        ) in reassignment_dict.items():
                            curr_assignment.loc[district_subregion] = (
                                chosen_districts[chosen_district_ix]
                            )
                        for (
                            chosen_district_ix,
                            district_subregion,
                        ) in reassignment_dict_alpha_reopt.items():
                            curr_assignment_alpha_reopt.loc[
                                district_subregion
                            ] = chosen_districts[chosen_district_ix]
                        assignment_dict = assignment_ser_to_dict(
                            curr_assignment_alpha_reopt
                        )
                        G_d = load_district_adjacency_graph(
                            self.config, curr_assignment_alpha_reopt
                        )
                        new_tups = get_connected_n_tuples(
                            G_d, n, intersecting_districts=chosen_districts
                        )
                        for chosen_districts in new_tups:
                            district_subregions = [
                                assignment_dict[district_id]
                                for district_id in chosen_districts
                            ]
                            priority = self.priority_func(district_subregions)
                            if priority >= 0:
                                heapq.heappush(
                                    pq, (-1 * priority, chosen_districts)
                                )
                        saved_assignments_df = pd.concat(
                            [
                                saved_assignments_df,
                                pd.DataFrame(
                                    {f"District{n_saved}": curr_assignment}
                                ),
                            ],
                            axis=1,
                        )
                        saved_assignments_df = pd.concat(
                            [
                                saved_assignments_df,
                                pd.DataFrame(
                                    {
                                        f"District{n_saved + 1}": curr_assignment_alpha_reopt
                                    }
                                ),
                            ],
                            axis=1,
                        )
                        self.saved_steps[plan].append(step)
                        n_saved += 2
            saved_assignments_df.to_csv(
                os.path.join(
                    self.save_path,
                    "assignments",
                    f"post_process_exp_{self.exp_id}_plan_{plan}_alpha_reopt.csv",
                ),
                index=False,
            )
            self.assignments_df[f"District{plan}"] = curr_assignment_alpha_reopt
        return self.assignments_df

        
 def priority_n_opt_with_beta_reoptimize(self, n, beta_reoptimize_steps):
        """
        For each of the chosen plans, add all connected n-tuples of
        districts in the district adjacency graph to a priority queue according to the
        comparison_func and heuristic_func. Reoptimize in decreasing order of priorty. If we have an improvement in the number of maj
        black districts, update the priorty queue with all neighboring n-tuples and change it.
        ------------------------------------------------------------------------
        Returns:
            self.assignments_df: (pd.DataFrame) assignments dataframe
        """
        self.config.alpha_reoptimize_steps = beta_reoptimize_steps
        for plan in self.plans:
            saved_assignments_df = pd.DataFrame()
            saved_assignments_df["GEOID"] = self.state_df["GEOID"]
            curr_assignment = self.assignments_df[f"District{plan}"]
            curr_assignment_alpha_reopt = curr_assignment.copy()
            assignment_dict = assignment_ser_to_dict(
                curr_assignment_alpha_reopt
            )
            G_d = load_district_adjacency_graph(self.config, curr_assignment)
            n_tups = get_connected_n_tuples(G_d, n)
            pq = []
            deleted_tups = set()
            step = 0
            n_saved = 0
            for chosen_districts in n_tups:
                district_subregions = [
                    assignment_dict[district_id]
                    for district_id in chosen_districts
                ]
                priority = self.priority_func(district_subregions)
                if priority >= 0:
                    heapq.heappush(pq, (-1 * priority, chosen_districts))
            while len(pq) > 0:
                step += 1
                chosen_districts = heapq.heappop(pq)[1]
                if chosen_districts not in deleted_tups:
                    district_subregions = [
                        assignment_dict[district_id]
                        for district_id in chosen_districts
                    ]
                    subregion = [
                        cgu_ix
                        for district_subregion in district_subregions
                        for cgu_ix in district_subregion
                    ]
                    self.config.subregion = subregion
                    self.config.n_districts = n
                    reassignment_df, reassignment_df_alpha_reopt = (
                        self.partition_func(self.config)
                    )
                    reassignment_dict = assignment_ser_to_dict(
                        reassignment_df["District0"]
                    )
                    reassignment_dict_alpha_reopt = assignment_ser_to_dict(
                        reassignment_df_alpha_reopt["District0"]
                    )

                    if n_maj_black(
                        reassignment_dict, self.state_df
                    ) > n_maj_black(
                        {
                            ix: district_subregions[ix]
                            for ix in range(len(district_subregions))
                        },
                        self.state_df,
                    ):
                        deleted_tups.update(
                            get_connected_n_tuples(
                                G_d, n, intersecting_districts=chosen_districts
                            )
                        )
                        for (
                            chosen_district_ix,
                            district_subregion,
                        ) in reassignment_dict.items():
                            curr_assignment.loc[district_subregion] = (
                                chosen_districts[chosen_district_ix]
                            )
                        for (
                            chosen_district_ix,
                            district_subregion,
                        ) in reassignment_dict_alpha_reopt.items():
                            curr_assignment_alpha_reopt.loc[
                                district_subregion
                            ] = chosen_districts[chosen_district_ix]
                        assignment_dict = assignment_ser_to_dict(
                            curr_assignment_alpha_reopt
                        )
                        G_d = load_district_adjacency_graph(
                            self.config, curr_assignment_alpha_reopt
                        )
                        new_tups = get_connected_n_tuples(
                            G_d, n, intersecting_districts=chosen_districts
                        )
                        for chosen_districts in new_tups:
                            district_subregions = [
                                assignment_dict[district_id]
                                for district_id in chosen_districts
                            ]
                            priority = self.priority_func(district_subregions)
                            if priority >= 0:
                                heapq.heappush(
                                    pq, (-1 * priority, chosen_districts)
                                )
                        saved_assignments_df = pd.concat(
                            [
                                saved_assignments_df,
                                pd.DataFrame(
                                    {f"District{n_saved}": curr_assignment}
                                ),
                            ],
                            axis=1,
                        )
                        saved_assignments_df = pd.concat(
                            [
                                saved_assignments_df,
                                pd.DataFrame(
                                    {
                                        f"District{n_saved + 1}": curr_assignment_alpha_reopt
                                    }
                                ),
                            ],
                            axis=1,
                        )
                        self.saved_steps[plan].append(step)
                        n_saved += 2
            saved_assignments_df.to_csv(
                os.path.join(
                    self.save_path,
                    "assignments",
                    f"post_process_exp_{self.exp_id}_plan_{plan}_alpha_reopt.csv",
                ),
                index=False,
            )
            self.assignments_df[f"District{plan}"] = curr_assignment_alpha_reopt
        return self.assignments_df
    

def enumerative_n_opt(self, n):
        """
        For each of the chosen plans, iterate through all connected n-tuples of
        districts in the district adjacency graph and reoptimize if the
        heuristic_func says to. If we have an improvement in the number of maj
        black districts, start over from the beginning.
        ------------------------------------------------------------------------
        Returns:
            self.assignments_df: (pd.DataFrame) assignments dataframe
        """
        for plan in self.plans:
            saved_assignments_df = pd.DataFrame()
            saved_assignments_df["GEOID"] = self.state_df["GEOID"]
            curr_assignment = self.assignments_df[f"District{plan}"]
            assignment_dict = assignment_ser_to_dict(curr_assignment)
            G_d = load_district_adjacency_graph(self.config, curr_assignment)
            n_tups = get_connected_n_tuples(G_d, n)
            step = 0
            n_saved = 0
            while len(n_tups) > 0:
                step += 1
                chosen_districts = n_tups.pop()
                district_subregions = [
                    assignment_dict[district_id]
                    for district_id in chosen_districts
                ]
                if self.heuristic_func(district_subregions):
                    subregion = [
                        cgu_ix
                        for district_subregion in district_subregions
                        for cgu_ix in district_subregion
                    ]
                    self.config.subregion = subregion
                    self.config.n_districts = n
                    reassignment_dict = assignment_ser_to_dict(
                        self.partition_func(self.config)["District0"]
                    )
                    if n_maj_black(
                        reassignment_dict, self.state_df
                    ) > n_maj_black(
                        {
                            ix: district_subregions[ix]
                            for ix in range(len(district_subregions))
                        },
                        self.state_df,
                    ):
                        for (
                            chosen_district_ix,
                            district_subregion,
                        ) in reassignment_dict.items():
                            curr_assignment.loc[district_subregion] = (
                                chosen_districts[chosen_district_ix]
                            )
                        assignment_dict = assignment_ser_to_dict(
                            curr_assignment
                        )
                        G_d = load_district_adjacency_graph(
                            self.config, curr_assignment
                        )
                        n_tups = get_connected_n_tuples(G_d, n)
                    if self.save_rule_func(step):
                        saved_assignments_df = pd.concat(
                            [
                                saved_assignments_df,
                                pd.DataFrame(
                                    {f"District{n_saved}": curr_assignment}
                                ),
                            ],
                            axis=1,
                        )
                        self.saved_steps[plan].append(step)
                        n_saved += 1
            saved_assignments_df.to_csv(
                os.path.join(
                    self.save_path,
                    "assignments",
                    f"post_process_exp_{self.exp_id}_plan_{plan}.csv",
                ),
                index=False,
            )
            self.assignments_df[f"District{plan}"] = curr_assignment
        return self.assignments_df

def short_bursts(self, n_steps, burst_length):  # TODO
        """
        Args:
            n_steps: (int) Number of repartitioning steps
            burst_length: (int) Number of steps per burst
            plans: (list(int)) Plans to reoptimize
        """
        n_bursts = n_steps // burst_length

        burst_assignments_df = pd.DataFrame()
        zeros = np.zeros(len(self.assignments_df["GEOID"]), dtype=int)
        for burst_step in range(burst_length):
            burst_assignments_df[f"District{burst_step}"] = zeros

        for plan in self.plans:
            curr_best_assignment_ser = self.assignments_df[f"District{plan}"]
            curr_best_n_maj_black = n_maj_black(
                assignment_ser_to_dict(curr_best_assignment_ser)
            )
            step = 0
            for burst in range(n_bursts):
                # Run the burst
                curr_assignment = curr_best_assignment_ser
                # curr_assignment = self.assignments_df[f'District{plan}']
                for burst_step in range(burst_length):
                    subregion = self.local_search_func(curr_assignment)
                    chosen_districts = curr_assignment[subregion].unique()
                    self.config.subregion = subregion
                    self.config.n_districts = len(chosen_districts)
                    reassignment_dict = assignment_ser_to_dict(
                        self.partition_func(self.config)["District0"]
                    )
                    for (
                        chosen_district_ix,
                        district_subregion,
                    ) in reassignment_dict.items():
                        curr_assignment.iloc[district_subregion] = (
                            chosen_districts[chosen_district_ix]
                        )
                    burst_assignments_df[f"District{burst_step}"] = (
                        curr_assignment.copy()
                    )
                    if self.save_rule_func(step):
                        self.saved_assignments_df[step] = curr_assignment
                    step += 1
                # Find the best assignment in the chain
                max_maj_black = 0
                curr_best_assignment_ser = burst_assignments_df["District0"]
                nums_maj_black = majority_black(
                    burst_assignments_df,
                    self.state_df,
                    self.config.n_districts,
                    num_plans=burst_length,
                ).sum(axis=1)
                for burst_step in range(1, burst_length):
                    if nums_maj_black[burst_step] > max_maj_black:
                        max_maj_black_burst_step = burst_step
                self.assignments_df[f"District{plan}"] = burst_assignments_df[
                    f"District{max_maj_black_burst_step}"
                ]
        return self.assignments_df

    def random_two_opt(self, n_steps):
        """
        For each of the chosen plans, for n_steps steps, perform a heuristic
        local search to find a subregion to reoptimize, and then reoptimize it.
        ------------------------------------------------------------------------
        Args:
            n_steps: (int) number of steps for which to iterate this process
        ------------------------------------------------------------------------
        Returns:
            self.assignments_df: (pd.DataFrame) assignments dataframe
        """
        for plan in self.plans:
            curr_assignment = self.assignments_df[f"District{plan}"]
            saved_assignments_df = pd.DataFrame()
            saved_assignments_df["GEOID"] = self.state_df["GEOID"]
            n_saved = 0
            for step in range(n_steps):
                subregion = self.local_search_func(curr_assignment)
                chosen_districts = curr_assignment[subregion].unique()
                self.config.subregion = subregion
                self.config.n_districts = len(chosen_districts)
                reassignment_dict = assignment_ser_to_dict(
                    self.partition_func(self.config)["District0"]
                )
                for (
                    chosen_district_ix,
                    district_subregion,
                ) in reassignment_dict.items():
                    curr_assignment.loc[district_subregion] = chosen_districts[
                        chosen_district_ix
                    ]
                if self.save_rule_func(step):
                    saved_assignments_df[f"District{n_saved}"] = curr_assignment
                    self.saved_steps[plan].append(step)
                    n_saved += 1
            saved_assignments_df.to_csv(
                os.path.join(
                    self.save_path,
                    "assignments",
                    f"post_process_exp_{self.exp_id}_plan_{plan}.csv",
                ),
                index=False,
            )
            self.assignments_df[f"District{plan}"] = curr_assignment
        return self.assignments_df

    def enumerative_two_opt(self):
        """
        For each of the chosen plans, iterate through the list of edges in the
        district adjacency graph and reoptimize if the heuristic_func says to.
        If we have an improvement in the number of maj black districts, start
        over from the beginning.
        ------------------------------------------------------------------------
        Returns:
            self.assignments_df: (pd.DataFrame) assignments dataframe
        """
        for plan in self.plans:
            saved_assignments_df = pd.DataFrame()
            saved_assignments_df["GEOID"] = self.state_df["GEOID"]
            curr_assignment = self.assignments_df[f"District{plan}"]
            assignment_dict = assignment_ser_to_dict(curr_assignment)
            G_d = load_district_adjacency_graph(self.config, curr_assignment)
            edges = list(G_d.edges())
            step = 0
            n_saved = 0
            while len(edges) > 0:
                step += 1
                chosen_districts = edges.pop()
                district_subregions = [
                    assignment_dict[district_id]
                    for district_id in chosen_districts
                ]
                if self.heuristic_func(district_subregions):
                    subregion = [
                        cgu_ix
                        for district_subregion in district_subregions
                        for cgu_ix in district_subregion
                    ]
                    self.config.subregion = subregion
                    self.config.n_districts = 2
                    reassignment_dict = assignment_ser_to_dict(
                        self.partition_func(self.config)["District0"]
                    )
                    if n_maj_black(
                        reassignment_dict, self.state_df
                    ) > n_maj_black(
                        {0: district_subregions[0], 1: district_subregions[1]},
                        self.state_df,
                    ):
                        for (
                            chosen_district_ix,
                            district_subregion,
                        ) in reassignment_dict.items():
                            curr_assignment.loc[district_subregion] = (
                                chosen_districts[chosen_district_ix]
                            )
                        assignment_dict = assignment_ser_to_dict(
                            curr_assignment
                        )
                        G_d = load_district_adjacency_graph(
                            self.config, curr_assignment
                        )
                        edges = list(G_d.edges())
                    if self.save_rule_func(step):
                        saved_assignments_df[f"District{n_saved}"] = (
                            curr_assignment
                        )
                        self.saved_steps[plan].append(step)
                        n_saved += 1
            saved_assignments_df.to_csv(
                os.path.join(
                    self.save_path,
                    "assignments",
                    f"post_process_exp_{self.exp_id}_plan_{plan}.csv",
                ),
                index=False,
            )
            self.assignments_df[f"District{plan}"] = curr_assignment
        return self.assignments_df

    def enumerative_three_opt(self):
        """
        For each of the chosen plans, iterate through all connected triples of
        districts in the district adjacency graph and reoptimize if the
        heuristic_func says to. If we have an improvement in the number of maj
        black districts, start over from the beginning.
        ------------------------------------------------------------------------
        Returns:
            self.assignments_df: (pd.DataFrame) assignments dataframe
        """
        for plan in self.plans:
            saved_assignments_df = pd.DataFrame()
            saved_assignments_df["GEOID"] = self.state_df["GEOID"]
            curr_assignment = self.assignments_df[f"District{plan}"]
            assignment_dict = assignment_ser_to_dict(curr_assignment)
            G_d = load_district_adjacency_graph(self.config, curr_assignment)
            three_tups = get_connected_three_tuples(G_d)
            step = 0
            n_saved = 0
            while len(three_tups) > 0:
                step += 1
                chosen_districts = three_tups.pop()
                district_subregions = [
                    assignment_dict[district_id]
                    for district_id in chosen_districts
                ]
                if self.heuristic_func(district_subregions):
                    subregion = [
                        cgu_ix
                        for district_subregion in district_subregions
                        for cgu_ix in district_subregion
                    ]
                    self.config.subregion = subregion
                    self.config.n_districts = 3
                    reassignment_dict = assignment_ser_to_dict(
                        self.partition_func(self.config)["District0"]
                    )
                    if n_maj_black(
                        reassignment_dict, self.state_df
                    ) > n_maj_black(
                        {
                            ix: district_subregions[ix]
                            for ix in range(len(district_subregions))
                        },
                        self.state_df,
                    ):
                        for (
                            chosen_district_ix,
                            district_subregion,
                        ) in reassignment_dict.items():
                            curr_assignment.loc[district_subregion] = (
                                chosen_districts[chosen_district_ix]
                            )
                        assignment_dict = assignment_ser_to_dict(
                            curr_assignment
                        )
                        G_d = load_district_adjacency_graph(
                            self.config, curr_assignment
                        )
                        three_tups = get_connected_three_tuples(G_d)
                    if self.save_rule_func(step):
                        saved_assignments_df = pd.concat(
                            [
                                saved_assignments_df,
                                pd.DataFrame(
                                    {f"District{n_saved}": curr_assignment}
                                ),
                            ],
                            axis=1,
                        )
                        self.saved_steps[plan].append(step)
                        n_saved += 1
            saved_assignments_df.to_csv(
                os.path.join(
                    self.save_path,
                    "assignments",
                    f"post_process_exp_{self.exp_id}_plan_{plan}.csv",
                ),
                index=False,
            )
            self.assignments_df[f"District{plan}"] = curr_assignment
        return self.assignments_df
'''

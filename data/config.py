from typing import Optional, Union, Any
import os
import json
import sys

sys.path.append(".")

from constants import (
    State,
    Granularity,
    CenterSelectionMethod,
    CapacitiesAssignmentMethod,
    CapacityWeights,
    Mode,
    VapCol,
    STATE_LIST,
    GRANULARITY_LIST,
    RESULTS_PATH,
)


class StateConfig:
    def __init__(
        self,
        state: State,
        year: int,
        granularity: Granularity,
        subregion: Optional[list[int]] = None,
    ):
        if state not in STATE_LIST:
            raise ValueError("Invalid State")
        if granularity not in GRANULARITY_LIST:
            raise ValueError("Invalid granularity")
        self.state = state
        self.year = year
        self.granularity = granularity
        self.subregion = subregion

    def get_dirname(self):
        return "%s_%d_%s" % (self.state, self.year, self.granularity)

    def save_to(self, file_path):
        """file must have extension .json"""
        with open(file_path, "w") as file:
            json.dump(self, file, indent=0)


class SHPConfig(StateConfig):
    def __init__(
        self,
        state: State,
        year: int,
        granularity: Granularity,
        subregion: Optional[list[int]],
        n_root_samples: int,
        n_samples: int,
        max_sample_tries: int,
        parent_resample_tries: int,
        max_n_splits: int,
        min_n_splits: int,
        max_split_population_difference: float,
        n_districts: int,
        population_tolerance: float,
        ideal_pop: float,
        n_beta_reoptimize_steps: int,
        selection_method: CenterSelectionMethod,
        perturbation_scale: int,
        n_random_seeds: int,
        capacities: CapacitiesAssignmentMethod,
        capacity_weights: CapacityWeights,
        col: VapCol,
        IP_gap_tol: float,
        IP_timeout: int,  # TODO: check if this is necessary
        final_partition_range: Optional[list[int]],
        final_partition_ips: list[
            str
        ],  # TODO: consider implementing a more specific type for this. Also consider changing the name to fit something more broad, like leaf_node_partition_IPs or smth
        beta: float,
        epsilon: float,
        use_warm_starts: bool,
        use_time_limit: bool,
        verbose: bool,
        event_logging: bool,  # TODO: check if this is necessary
        debug_file: Union[
            str, Any
        ],  # TODO: think about potentially deleting? Or making this part of the code better. Also figure out the correct type.
        debug_file_2: Union[str, Any],
        callback_time_interval: int,  # TODO: consider getting rid of this
        save_dirname: str,
        save_config: bool,
        save_tree: bool,
        save_cdms: bool,
        save_district_adj_graphs: bool,
        results_path: str,
        assignments_file_name: str,
        linear_objective: bool,  # TODO: consider getting rid of this
        mode: Mode,
        tree_time_str: str,
    ):
        super().__init__(state, year, granularity)
        self.n_root_samples = n_root_samples
        self.n_samples = n_samples
        self.max_sample_tries = max_sample_tries
        self.parent_resample_tries = parent_resample_tries
        self.max_n_splits = max_n_splits
        self.min_n_splits = min_n_splits
        self.max_split_population_difference = max_split_population_difference
        self.n_districts = n_districts
        self.population_tolerance = population_tolerance
        self.ideal_pop = ideal_pop
        self.subregion = subregion
        self.n_beta_reoptimize_steps = n_beta_reoptimize_steps
        self.selection_method = selection_method
        self.perturbation_scale = perturbation_scale
        self.n_random_seeds = n_random_seeds
        self.capacities = capacities
        self.capacity_weights = capacity_weights
        self.col = col
        self.IP_gap_tol = IP_gap_tol
        self.IP_timeout = IP_timeout
        self.final_partition_range = final_partition_range
        self.final_partition_ips = final_partition_ips
        self.beta = beta
        self.epsilon = epsilon
        self.use_warm_starts = use_warm_starts
        self.use_time_limit = use_time_limit
        self.verbose = verbose
        self.event_logging = event_logging
        self.debug_file = debug_file
        self.debug_file_2 = debug_file_2
        self.callback_time_interval = callback_time_interval
        self.save_path = self.get_save_path(save_dirname)
        self.save_config = save_config
        self.save_tree = save_tree
        self.save_cdms = save_cdms
        self.save_district_adj_graphs = save_district_adj_graphs
        self.results_path = results_path
        self.assignments_file_name = assignments_file_name
        self.linear_objective = linear_objective
        self.mode = mode
        self.tree_time_str = tree_time_str

    def get_save_path(self, save_dirname):
        partial_save_path = os.path.join(RESULTS_PATH, self.get_dirname())
        if not os.path.exists(partial_save_path):
            os.mkdir(partial_save_path)
        return os.path.join(partial_save_path, save_dirname)

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
            json.dump(self.__dict__, file, indent=0)


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
        n_beta_reoptimize_steps: int,
        selection_method: CenterSelectionMethod,
        perturbation_scale: int,
        n_random_seeds: int,
        capacities: CapacitiesAssignmentMethod,
        capacity_weights: CapacityWeights,
        col: VapCol,
        ip_gap_tol: float,
        ip_timeout: Optional[
            float
        ],  # If not None, then all ips have a time limit of n_cgus * ip_timeout
        final_partition_range: Optional[list[int]],
        final_partition_ips: list[
            str
        ],  # TODO: consider implementing a more specific type for this. Also consider changing the name to fit something more broad, like leaf_node_partition_IPs or smth
        beta: float,
        epsilon: float,
        use_warm_starts: bool,
        save_dirname: str,
    ):
        super().__init__(state, year, granularity, subregion)
        self.n_root_samples = n_root_samples
        self.n_samples = n_samples
        self.max_sample_tries = max_sample_tries
        self.parent_resample_tries = parent_resample_tries
        self.max_n_splits = max_n_splits
        self.min_n_splits = min_n_splits
        self.max_split_population_difference = max_split_population_difference
        self.n_districts = n_districts
        self.population_tolerance = population_tolerance
        self.subregion = subregion
        self.n_beta_reoptimize_steps = n_beta_reoptimize_steps
        self.selection_method = selection_method
        self.perturbation_scale = perturbation_scale
        self.n_random_seeds = n_random_seeds
        self.capacities = capacities
        self.capacity_weights = capacity_weights
        self.col = col
        self.ip_gap_tol = ip_gap_tol
        self.ip_timeout = ip_timeout
        self.final_partition_range = final_partition_range
        self.final_partition_ips = final_partition_ips
        self.beta = beta
        self.epsilon = epsilon
        self.use_warm_starts = use_warm_starts
        self.save_path = self.get_save_path(save_dirname)

    def get_save_path(self, save_dirname):
        partial_save_path = os.path.join(RESULTS_PATH, self.get_dirname())
        if not os.path.exists(partial_save_path):
            os.mkdir(partial_save_path)
        return os.path.join(partial_save_path, save_dirname)

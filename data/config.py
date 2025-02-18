from typing import Optional, get_args
import os
import json
import numpy as np
import sys

sys.path.append(".")
from constants import (
    State,
    Granularity,
    CenterSelectionMethod,
    CapacitiesAssignmentMethod,
    CapacityWeights,
    VapCol,
    IPStr,
    PriorityFuncStr,
    PartitionFuncStr,
    RESULTS_PATH,
    SIX_COLORS,
)


class StateConfig:
    def __init__(
        self,
        state: State,
        year: int,
        granularity: Granularity,
        subregion: Optional[list[int]] = None,
    ):
        if state not in get_args(State):
            raise ValueError("Invalid state")
        if granularity not in get_args(Granularity):
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


"""
class OldSHPConfig(StateConfig):
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
        ip_timeout: Optional[float],
        # If ip_timeout is not None, then all ips have a time limit of n_cgus * ip_timeout
        final_partition_range: list[int],
        final_partition_ips: list[IPStr],
        beta: float,
        epsilon: float,
        use_warm_starts: bool,
        save_dirname: str,
    ):
        if selection_method not in get_args(CenterSelectionMethod):
            raise ValueError("Invalid center selection method")
        if capacities not in get_args(CapacitiesAssignmentMethod):
            raise ValueError("Invalid capacities assignment method")
        if capacity_weights not in get_args(CapacityWeights):
            raise ValueError("Invalid capacity weights")
        if col not in get_args(VapCol):
            raise ValueError("Invalid vap col")
        ip_strs = get_args(IPStr)
        for ip_str in final_partition_ips:
            if ip_str not in ip_strs:
                raise ValueError(f"Invalid ip str: {ip_str}")
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
"""


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
        ip_timeout: Optional[float],
        # If ip_timeout is not None, then all ips have a time limit of n_cgus * ip_timeout
        final_partition_threshold: int,  # 0 represents no threshold
        final_partition_ips: list[IPStr],
        beta: float,
        epsilon: float,
        use_warm_starts: bool,
        save_dirname: str,
    ):
        if selection_method not in get_args(CenterSelectionMethod):
            raise ValueError("Invalid center selection method")
        if capacities not in get_args(CapacitiesAssignmentMethod):
            raise ValueError("Invalid capacities assignment method")
        if capacity_weights not in get_args(CapacityWeights):
            raise ValueError("Invalid capacity weights")
        if col not in get_args(VapCol):
            raise ValueError("Invalid vap col")
        ip_strs = get_args(IPStr)
        for ip_str in final_partition_ips:
            if ip_str not in ip_strs:
                raise ValueError(f"Invalid ip str: {ip_str}")
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
        self.final_partition_threshold = final_partition_threshold
        self.final_partition_ips = final_partition_ips
        self.beta = beta
        self.epsilon = epsilon
        self.use_warm_starts = use_warm_starts
        self.save_dirname = save_dirname

    def get_save_path(self):
        partial_save_path = os.path.join(RESULTS_PATH, self.get_dirname())
        if not os.path.exists(partial_save_path):
            os.mkdir(partial_save_path)
        return os.path.join(partial_save_path, self.save_dirname)


# TODO: decide if I want to keep this as a StateConfig subclass or make it
# its own thing
class PostProcessorConfig(StateConfig):
    def __init__(
        self,
        state: State,
        year: int,
        granularity: Granularity,
        subregion: Optional[list[int]],
        delta: float,
        priority_func_str: PriorityFuncStr,
        partition_func_str: PartitionFuncStr,
        loaded_partitions_file_name: str,
        saved_partitions_file_name: str,
    ):
        super().__init__(state, year, granularity, subregion)
        self.delta = delta
        self.priority_func_str = priority_func_str
        self.partition_func_str = partition_func_str
        self.loaded_partitions_file_name = loaded_partitions_file_name
        self.saved_partitions_file_name = saved_partitions_file_name


class MapConfig(StateConfig):
    def __init__(
        self,
        state: State,
        year: int,
        granularity: Granularity,
        subregion: Optional[list[int]],
        partitions_file_name: Optional[str] = None,
        colors: list[np.ndarray] = SIX_COLORS,
        cmap_name: Optional[str] = None,
    ):
        super().__init__(state, year, granularity, subregion)
        self.partitions_file_name = partitions_file_name
        self.colors = colors
        self.cmap_name = cmap_name

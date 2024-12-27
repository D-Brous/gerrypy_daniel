import sys

sys.path.append("../gerrypy_daniel")

from constants import (
    State,
    Granularity,
    CenterSelectionMethod,
    CapacitiesAssignmentMethod,
    CapacityWeights,
    Mode,
    STATE_LIST,
    GRANULARITY_LIST,
)
from typing import Optional, Union, Any


class StateConfig:
    def __init__(self, state: State, year: int, granularity: Granularity):
        if state not in STATE_LIST:
            raise ValueError("Invalid State")
        if granularity not in GRANULARITY_LIST:
            raise ValueError("Invalid granularity")
        self.state = state
        self.year = year
        self.granularity = granularity

    def get_dirname(self):
        return "%s_%d_%s" % (self.state, self.year, self.granularity)


class SHPConfig(StateConfig):
    def __init__(
        self,
        state: State,
        year: int,
        granularity: Granularity,
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
        subregion: Optional[list[int]],
        alpha_reoptimize_steps: int,
        selection_method: CenterSelectionMethod,
        perturbation_scale: int,
        n_random_seeds: int,
        capacities: CapacitiesAssignmentMethod,
        capacity_weights: CapacityWeights,
        IP_gap_tol: float,
        IP_timeout: int,  # TODO: check if this is necessary
        exact_partition_range: Optional[list[int]],
        maj_black_partition_IPs: list[
            str
        ],  # TODO: consider implementing a more specific type for this. Also consider changing the name to fit something more broad, like leaf_node_partition_IPs or smth
        alpha: float,
        epsilon: float,
        use_black_maj_warm_start: bool,
        use_time_limit: bool,
        verbose: bool,
        event_logging: bool,  # TODO: check if this is necessary
        debug_file: Union[
            str, Any
        ],  # TODO: think about potentially deleting? Or making this part of the code better. Also figure out the correct type.
        debug_file_2: Union[str, Any],
        callback_time_interval: int,  # TODO: consider getting rid of this
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
        self.alpha_reoptimize_steps = alpha_reoptimize_steps
        self.selection_method = selection_method
        self.perturbation_scale = perturbation_scale
        self.n_random_seeds = n_random_seeds
        self.capacities = capacities
        self.capacity_weights = capacity_weights
        self.IP_gap_tol = IP_gap_tol
        self.IP_timeout = IP_timeout
        self.exact_partition_range = exact_partition_range
        self.maj_black_partition_IPs = maj_black_partition_IPs
        self.alpha = alpha
        self.epsilon = epsilon
        self.use_black_maj_warm_start = use_black_maj_warm_start
        self.use_time_limit = use_time_limit
        self.verbose = verbose
        self.event_logging = event_logging
        self.debug_file = debug_file
        self.debug_file_2 = debug_file_2
        self.callback_time_interval = callback_time_interval
        self.save_config = save_config
        self.save_tree = save_tree
        self.save_cdms = save_cdms
        self.save_district_adj_graphs = save_district_adj_graphs
        self.results_path = results_path
        self.assignments_file_name = assignments_file_name
        self.linear_objective = linear_objective
        self.mode = mode
        self.tree_time_str = tree_time_str

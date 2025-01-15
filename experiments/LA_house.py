import sys

sys.path.append(".")

# import constants
from data.config import SHPConfig
from optimize.shp import SHP

# from functools import partial
# import time
# from optimize.generate import ColumnGenerator

state_config = {
    "state": "LA",
    "year": 2010,
    "granularity": "block_group",
    "subregion": None,
}
tree_config = {
    "n_root_samples": 10,
    "n_samples": 3,  # Should be 3-5
    "max_sample_tries": 5,  # 25 before
    "parent_resample_tries": 5,  # 5 before
    "max_n_splits": 2,
    "min_n_splits": 2,
    "max_split_population_difference": 1.5,
    "n_districts": 105,
    "population_tolerance": 0.045,
    "n_beta_reoptimize_steps": 30,
}
center_selection_config = {
    "selection_method": "uniform_random",
    "perturbation_scale": 1,
    "n_random_seeds": 1,
    "capacities": "match",
    "capacity_weights": "voronoi",
}
ip_config = {
    "ip_gap_tol": 1e-3,
    "ip_timeout": 5e-3,
    "col": "POCVAP",
    "final_partition_range": [2, 3, 4, 5],
    "final_partition_ips": ["maj_cvap_exact"],
    "use_warm_starts": False,
    "beta": 0,
    "epsilon": 0,
}
save_config = {
    "save_dirname": "LA_POCVAP",
}
shp_config = SHPConfig(
    **{
        **state_config,
        **tree_config,
        **center_selection_config,
        **ip_config,
        **save_config,
    }
)

if __name__ == "__main__":
    shp = SHP(shp_config)
    shp.shp(
        save_config=True,
        save_tree=True,
        save_partitions=True,
        logging=True,
        printing=True,
    )

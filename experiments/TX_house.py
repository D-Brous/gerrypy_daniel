import sys

sys.path.append(".")
from constants import SEATS_DICT
from data.config import SHPConfig, PostProcessorConfig
from optimize.generate import ColumnGenerator
from optimize.postprocess import PostProcessor

state = "TX"
col = "HVAP"
n = 4
pop_tol = 0.02


state_config = {
    "state": state,
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
    "n_districts": SEATS_DICT[state]["state_house"],
    "population_tolerance": pop_tol,
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
    "col": col,
    "final_partition_threshold": 5,
    "final_partition_ips": ["maj_cvap_exact"],
    "use_warm_starts": False,
    "beta": 0,
    "epsilon": 0,
}
save_config = {
    "save_dirname": f"state_house_{col}",
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
local_reopt_config = {
    "delta": 0,
    "priority_func_str": "average_geq_threshold_excess",
    "partition_func_str": "maj_cvap_ip",
    "loaded_partitions_file_name": "shp.csv",
    "saved_partitions_file_name": f"priority_{n}_opt_no_br.csv",
}
pp_config = PostProcessorConfig(**state_config, **local_reopt_config)
if __name__ == "__main__":
    from optimize.tree import SHPTree
    import os

    tree = SHPTree.from_file(
        os.path.join(shp_config.get_save_path(), "tree.pickle")
    )
    cg = ColumnGenerator(shp_config, tree=tree)
    # cg.run_shp(
    #     save_config=True,
    #     save_tree=True,
    #     save_partitions=True,
    #     logging=True,
    #     printing=True,
    #     run_beta_reoptimization=True,
    # )
    cg.run_beta_reopt(save_partitions=True, logging=True, printing=True)

    # shp_config.n_beta_reoptimize_steps = 0
    # post_processor = PostProcessor(shp_config, pp_config)
    # plan_ids = list(range(10))
    # for plan_id in plan_ids:
    #     post_processor.priority_n_opt(n, plan_id)

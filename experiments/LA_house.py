import sys

sys.path.append(".")
from constants import SEATS_DICT
from data.config import SHPConfig, PostProcessorConfig
from optimize.generate import ColumnGenerator
from optimize.postprocess import PostProcessor
from optimize.shortbursts import (
    short_burst_run,
    create_seed,
    get_gc_graph,
    load_seed,
    get_my_partition,
)

from gerrychain import (
    Partition,
    Graph,
    MarkovChain,
    proposals,
    updaters,
    constraints,
)

state = "LA"
col = "POCVAP"
n = 4
pop_tol = 0.045


state_config = {
    "state": state,
    "year": 2020,
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
    # from optimize.tree import SHPTree
    # import os

    # cg = ColumnGenerator(shp_config)
    # cg.run_shp(
    #     save_config=True,
    #     save_tree=True,
    #     save_partitions=True,
    #     logging=True,
    #     printing=True,
    #     run_beta_reoptimization=True,
    # )

    # # tree = SHPTree.from_file(
    # #     os.path.join(shp_config.get_save_path(), "tree.pickle")
    # # )
    # # cg = ColumnGenerator(shp_config, tree = tree)
    # # cg.run_beta_reopt(save_partitions=True, logging=True, printing=True)

    # shp_config.n_beta_reoptimize_steps = 0
    # post_processor = PostProcessor(shp_config, pp_config)
    # plan_ids = list(range(10))  # [0, 5, 6]  # [1, 2, 3, 4, 8, 9] # [0,5,6,7]
    # for plan_id in plan_ids:
    #     post_processor.priority_n_opt(n, plan_id)

    seed_ids = list(range(10))

    # for seed_id in seed_ids:
    #     create_seed(f"seed_{seed_id}.json", shp_config)

    from analyze.feasibility import check_plan_feasibility, check_feasibility
    from data.demo_df import DemoDataFrame
    from data.shape_df import ShapeDataFrame
    from data.graph import Graph
    from data.partition import Partitions
    from analyze.maj_min import n_maj_cvap
    import os
    import time
    import math
    import matplotlib.pyplot as plt
    import numpy as np

    # graph = get_gc_graph(shp_config)
    # chain_updaters = {}

    # partitions = Partitions()
    # for seed_id in seed_ids:
    #     partition = get_my_partition(
    #         load_seed(
    #             f"seed_{seed_id}.json", shp_config, graph, chain_updaters
    #         ),
    #         shp_config,
    #     )
    #     partitions.set_plan(seed_id, partition)
    # check_feasibility(shp_config, partitions, demo_df, G)

    demo_df = DemoDataFrame.from_config(shp_config)
    shape_df = ShapeDataFrame.from_config(shp_config)
    G = Graph.from_shape_df(shape_df)

    n_steps = 500000
    cannon_et_al_steps = 100000
    burst_size = 10
    max_partitions = Partitions()
    csv_path = os.path.join(
        shp_config.get_save_path(), "partitions", "short_bursts.csv"
    )
    times = []
    n_maj_cvaps = []
    n_maj_cvaps_cannon_et_al = []
    for seed_id in seed_ids:
        t_init = time.thread_time()
        print(
            f"Running short bursts of length {burst_size} for {n_steps} steps on seed {seed_id}:"
        )
        max_scores_sb, scores_sb, max_part = short_burst_run(
            shp_config, n_steps, burst_size, f"seed_{seed_id}.json"
        )
        times.append(time.thread_time() - t_init)
        print(f"Runtime: {times[-1]} s")
        print(
            f"Number of maj {shp_config.col} districts after {cannon_et_al_steps} steps: {int(max_scores_sb[cannon_et_al_steps-1])}"
        )
        print(
            f"Number of maj {shp_config.col} districts after {n_steps} steps: {int(max_scores_sb[-1])}"
        )
        n_maj_cvaps_cannon_et_al.append(
            int(max_scores_sb[cannon_et_al_steps - 1])
        )
        n_maj_cvaps.append(int(max_scores_sb[-1]))
        max_partition = get_my_partition(max_part, shp_config)
        max_partitions.set_plan(seed_id, max_partition)
        check_feasibility(shp_config, max_partitions, demo_df, G)

        max_partitions.to_csv(csv_path, shp_config)
        np.save(
            os.path.join(
                shp_config.get_save_path(),
                "partitions",
                f"short_bursts_{seed_id}.npy",
            ),
            scores_sb,
        )
        n_maj = n_maj_cvap(shp_config.col, max_partition, demo_df)
        assert n_maj == n_maj_cvaps[-1]
        # n_maj_cvaps.append(n_maj)
        # print(n_maj)
    print(f"Runtimes: {times}")
    print(
        f"Numbers of maj {shp_config.col} districts after {cannon_et_al_steps} steps: {n_maj_cvaps_cannon_et_al}"
    )
    print(
        f"Numbers of maj {shp_config.col} districts after {n_steps} steps: {n_maj_cvaps}"
    )

    # scores_sb = np.load(
    #     os.path.join(
    #         shp_config.get_save_path(),
    #         "partitions",
    #         f"short_bursts_0.npy",
    #     )
    # )
    # plt.plot(list(range(500000)), scores_sb)
    # plt.show()

    # burst_size = 10
    # seed_id = 0
    # max_partitions = Partitions()
    # times = []
    # n_steps_vals = [10, 100, 1000, 10000, 100000]
    # for n_steps in n_steps_vals:
    #     t_init = time.thread_time()
    #     max_scores_sb, scores_sb, max_part = short_burst_run(
    #         shp_config, n_steps, burst_size, f"seed_{seed_id}.json"
    #     )
    #     t_final = time.thread_time()
    #     times.append(t_final)
    #     print(f"Seed {seed_id}: {t_final}")
    #     print(max_scores_sb[-1])
    #     max_partition = get_my_partition(max_part, shp_config)
    #     max_partitions.set_plan(seed_id, max_partition)
    #     check_feasibility(shp_config, max_partitions, demo_df, G)
    #     print(n_maj_cvap(shp_config.col, max_partition, demo_df))
    # log_n_steps_vals = [math.log(n_steps) for n_steps in n_steps_vals]
    # log_times = [math.log(time) for time in times]
    # plt.plot(log_n_steps_vals, log_times)
    # plt.show()

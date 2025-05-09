import networkx as nx
import numpy as np
import pandas as pd
import sys

sys.path.append(".")
from data.config import SHPConfig
from data.demo_df import DemoDataFrame
from data.graph import Graph
from data.partition import Partition, Partitions


def check_feasibility(
    config: SHPConfig, partitions: Partitions, demo_df: DemoDataFrame, G: Graph
):
    # n_plans = config.n_root_samples
    # if len(config.final_partition_ips) > 0:
    #    n_plans *= len(config.final_partition_ips)
    n_districts = config.n_districts
    n_cgus = demo_df.get_n_cgus()
    ideal_pop = demo_df.get_ideal_pop(n_districts)
    district_pop_ub = (1 + config.population_tolerance) * ideal_pop
    district_pop_lb = (1 - config.population_tolerance) * ideal_pop
    for plan_id in partitions.get_plan_ids():
        partition = partitions.get_plan(plan_id)
        error_str = check_plan_feasibility(
            plan_id,
            partition,
            demo_df,
            G,
            n_districts,
            n_cgus,
            district_pop_ub,
            district_pop_lb,
        )
        if error_str is not None:
            raise RuntimeError(error_str)


def check_plan_feasibility(
    plan_id: int,
    partition: Partition,
    demo_df: DemoDataFrame,
    G: Graph,
    n_districts: int,
    n_cgus: int,
    district_pop_ub: float,
    district_pop_lb: float,
):
    subregion = partition.get_region()
    cgus_used = pd.Series(np.full(n_cgus, False, dtype=bool), index=subregion)
    # cgus_used = np.full(n_cgus, False, dtype=bool)
    if len(partition.get_parts()) != n_districts:
        return f"Plan {plan_id} has the wrong number of districts"
    for district_id, district_subregion in partition.get_parts().items():
        # if np.any(cgus_used[district_subregion]):
        cgus_used_district = cgus_used.loc[district_subregion]
        if cgus_used_district.any():
            # repeated = district_subregion[
            #    np.where(cgus_used[district_subregion] == True)
            # ]
            repeated = list(cgus_used_district[cgus_used_district].index)
            return f"Cgus {repeated} are used in more than one district in plan {plan_id}"
        else:
            cgus_used[district_subregion] = True
        if not nx.is_connected(G.get_subgraph(district_subregion)):
            return (
                f"District {district_id} from plan {plan_id} is not connected"
            )
        district_df = demo_df.get_subregion_df(district_subregion)
        district_pop = district_df["POP"].sum()
        if district_pop > district_pop_ub:
            return f"The population of district {district_id} from plan {plan_id} is too large"
        if district_pop < district_pop_lb:
            return f"The population of district {district_id} from plan {plan_id} is too small"
    # if not np.all(cgus_used):
    cgus_unused = ~cgus_used
    if cgus_unused.any():
        # unused = np.where(cgus_used == False)
        unused = list(cgus_unused[cgus_unused].index)
        return f"Cgus {unused} are unused in plan {plan_id}"

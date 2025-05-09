import os
import pandas as pd
import numpy as np
import pickle
from functools import partial
from gerrychain import (
    Partition,
    Graph,
    MarkovChain,
    proposals,
    updaters,
    constraints,
    tree,
)
from gerrychain.optimization import SingleMetricOptimizer, Gingleator
from networkx.readwrite import json_graph
import networkx as nx
import json
import sys

sys.path.append(".")
from constants import DEMO_DATA_PATH, SEED_PATH
from data.config import SHPConfig
from data.shape_df import ShapeDataFrame
from data.demo_df import DemoDataFrame
from data.partition import Partition as P
from data.graph import Graph as G

POPCOL = "POP"


def get_my_partition(partition: Partition, shp_config: SHPConfig) -> P:
    assignment_dict = partition.assignment.to_dict()
    n_cgus = len(assignment_dict)
    assignment_array = np.zeros(n_cgus)
    demo_df = DemoDataFrame.from_config(shp_config)
    geoid_dict = {row["GEOID"]: index for index, row in demo_df.iterrows()}
    # print(geoid_dict)
    # print(partition.graph.nodes[0]["GEOID20"])
    for node in range(n_cgus):
        cgu = geoid_dict[int(partition.graph.nodes[node]["GEOID20"])]
        # print(node, cgu)
        assignment_array[cgu] = assignment_dict[node]
    return P.from_assignment_ser(pd.Series(assignment_array))

    parts = {id: [] for id in ranges(shp_config.n_districts)}
    for node, id in partition.assignment.to_dict():
        parts[id].append(node)
    my_partition = P(shp_config.n_districts)
    my_partition.districts


# def check_feasibility(partition: Partition, shp_config: SHPConfig):
#     parts = {id: [] for id in range(shp_config.n_districts)}
#     for node, id in partition.assignment.to_dict():
#         parts[id].append(node)
#     demo_df = DemoDataFrame.from_config(shp_config)
#     ideal_pop = total_pop / shp_config.n_districts
#     lb = ideal_pop * (1 - shp_config.population_tolerance)
#     ub = ideal_pop * (1 - shp_config.population_tolerance)
#     total_pop = demo_df["POP"].sum()
#     for id in range(shp_config.n_districts):
#         subregion_df = demo_df.loc[parts[id]]
#         district_pop = subregion_df["POP"].sum()
#         ideal


def get_gc_graph(shp_config: SHPConfig):
    graph = Graph.from_json(
        os.path.join(
            DEMO_DATA_PATH,
            shp_config.get_dirname(),
            f"{shp_config.state}_blockgroup.json",
        )
    )
    demo_df = DemoDataFrame.from_config(shp_config)
    demo_df["GEOID"] = demo_df["GEOID"].astype(str)
    # demo_df["POP"] = demo_df["POP"].astype(float)
    # print(demo_df.loc[3128]["GEOID"])
    graph.join(demo_df, left_index="GEOID20", right_index="GEOID")
    graph.to_json(
        os.path.join(
            DEMO_DATA_PATH,
            shp_config.get_dirname(),
            f"{shp_config.state}_blockgroup_appended.json",
        )
    )
    # for i in range(len(graph.nodes)):
    #     print(type(graph.nodes[i]["POP"]), type(graph.nodes[i]["P0010001"]))
    #     print(graph.nodes[i]["POP"], (graph.nodes[i]["P0010001"]))
    #     if graph.nodes[i]["POP"] != graph.nodes[i]["P0010001"]:
    #         print(i, graph.nodes[i]["POP"], graph.nodes[i]["P0010001"])
    return graph


def create_seed(filename: str, shp_config: SHPConfig):
    graph = get_gc_graph(shp_config)
    seed_dir = os.path.join(SEED_PATH, shp_config.get_dirname())
    seed_file = os.path.join(seed_dir, filename)

    n_failures = 0
    incomplete = True
    while incomplete:
        try:
            initial_partition = Partition.from_random_assignment(
                graph=graph,
                n_parts=shp_config.n_districts,
                epsilon=shp_config.population_tolerance,  # shp_config.population_tolerance / 3,
                pop_col=POPCOL,
                method=tree.recursive_seed_part,
            )
            incomplete = False
            if not os.path.exists(seed_dir):
                os.mkdir(seed_dir)
            with open(seed_file, "w") as f:
                json.dump(initial_partition.assignment.to_dict(), f)
            return
        except Exception as exp:
            n_failures += 1
            print(exp)
            print(f"Number of failed attempts so far: {n_failures}")
            pass


def load_seed(
    filename: str, shp_config: SHPConfig, graph: Graph, updaters: updaters
):
    seed_dir = os.path.join(
        SEED_PATH,
        shp_config.get_dirname(),
    )
    seed_file = os.path.join(seed_dir, filename)
    if os.path.exists(seed_file):
        with open(seed_file, "r") as f:
            assignment = json.load(f)
        new_assignment = dict()
        for key, value in assignment.items():
            new_assignment[int(key)] = value
        initial_partition = Partition(
            graph=graph, assignment=new_assignment, updaters=updaters
        )
        return initial_partition
    else:
        raise FileExistsError(f"{filename} is not an existing seed file")


# def get_initial_partition(
#     shp_config: SHPConfig, graph: Graph, updaters: updaters
# ):
#     seed_dir = os.path.join(
#         SEED_PATH,
#         shp_config.get_dirname(),
#     )
#     seed_file = os.path.join(seed_dir, "seed.json")
#     if os.path.exists(seed_file):
#         with open(seed_file, "r") as f:
#             assignment = json.load(f)
#         initial_partition = Partition(
#             graph=graph, assignment=assignment, updaters=updaters
#         )
#     else:
#         if not os.path.exists(seed_dir):
#             os.mkdir(seed_dir)
#         initial_partition = Partition.from_random_assignment(
#             graph=graph,
#             n_parts=shp_config.n_districts,
#             epsilon=shp_config.population_tolerance,
#             pop_col=POPCOL,
#             updaters=updaters,
#         )
#         with open(seed_file, "w") as f:
#             json.dump(initial_partition.assignment, f)
#     return initial_partition


def short_burst_run(
    shp_config: SHPConfig, n_steps: int, burst_size: int, seed_filename: str
):
    # shape_df = ShapeDataFrame.from_config(shp_config)
    # # g = G.get_cgu_adjacency_graph(shp_config)
    # # g = nx.Graph([(1, 2)])
    # # data = json_graph.adjacency_data(g)
    # # json_string = json.dumps(data)
    # json_string = shape_df.to_json()
    # json_path = os.path.join(
    #     DEMO_DATA_PATH,
    #     shp_config.get_dirname(),
    #     f"{shp_config.state}_blockgroup_custom.json",
    # )
    # with open(json_path, "w") as f:
    #     f.write(json_string)
    graph = get_gc_graph(shp_config)

    TOTPOP = sum(graph.nodes()[n][POPCOL] for n in graph.nodes())
    CVAPCOL = shp_config.col
    chain_updaters = {
        "population": updaters.Tally(POPCOL, alias="population"),
        "VAP": updaters.Tally("VAP"),
        CVAPCOL: updaters.Tally(CVAPCOL),
    }
    initial_partition = load_seed(
        seed_filename, shp_config, graph, chain_updaters
    )
    proposal = partial(
        proposals.recom,
        pop_col=POPCOL,
        pop_target=TOTPOP / shp_config.n_districts,
        epsilon=shp_config.population_tolerance,
        node_repeats=1,
    )
    cnstrs = constraints.within_percent_of_ideal_population(
        initial_partition, shp_config.population_tolerance
    )

    # def num_maj_T(partition):
    #     partition
    #
    # optimizer = SingleMetricOptimizer(
    #     proposals=proposal,
    #     constraints=cnstrs,
    #     initial_state=initial_partition,
    #     optimization_metric=num_maj_T,
    #     maximize=True,
    # )

    gingleator = Gingleator(
        proposal=proposal,
        constraints=cnstrs,
        initial_state=initial_partition,
        minority_pop_col=CVAPCOL,
        total_pop_col="VAP",
        score_function=Gingleator.num_opportunity_dists,
    )

    max_scores_sb = np.zeros(n_steps)
    scores_sb = np.zeros(n_steps)
    n_bursts = n_steps // burst_size
    last_burst = dict()
    for i, part in enumerate(
        gingleator.short_bursts(burst_size, n_bursts, with_progress_bar=True)
    ):
        max_scores_sb[i] = gingleator.best_score
        scores_sb[i] = gingleator.score(part)
        if n_steps - burst_size <= i <= n_steps - 1:
            last_burst[i] = part
    max_score = max_scores_sb[-1]
    for i in last_burst:
        if scores_sb[i] == max_score:
            max_part = last_burst[i]
    return max_scores_sb, scores_sb, max_part


# method = partial(
#     tree.recursive_tree_part,
#     node_repeats=100,
#     method=partial(
#         tree.bipartition_tree,
#         allow_pair_reselection=True,
#     ),
# )
# gen_method = partial(
#     tree.recursive_seed_part,
#     node_repeats=100,
#     method=partial(
#         tree.bipartition_tree,
#         allow_pair_reselection=True,
#     ),
# )

# graph = Graph.from_json(
#     os.path.join(
#         DEMO_DATA_PATH,
#         shp_config.get_dirname(),
#         "LA_blockgroup_appended.json",
#     )
# )
# seed_file = "LA_seed.json"

# n_failures = 0
# while True:
#     try:
#         print(shp_config.population_tolerance)
#         initial_partition = Partition.from_random_assignment(
#             graph=graph,
#             n_parts=shp_config.n_districts,
#             epsilon=shp_config.population_tolerance,
#             pop_col=POPCOL,
#             method=tree.recursive_seed_part,
#         )

#         with open(seed_file, "w") as f:
#             json.dump(initial_partition.assignment.to_dict(), f)
#         return
#     except Exception as exp:
#         print(type(exp))
#         print(f"\tFailed to create partition: {exp}")
#         n_failures += 1
#         print(f"\tNumber of failures: {n_failures}")
#         pass

# assingment_dicts = []
# for i in range(10):
#     print(i)
#     good = False
#     failed_attempts = 0
#     while not good:
#         try:
#             initial_partition = Partition.from_random_assignment(
#                 graph=graph,
#                 n_parts=105,
#                 epsilon=0.045,
#                 pop_col="P0010001",
#                 method=tree.recursive_seed_part,
#             )
#             good = True
#             print(f"\tSuccess after {failed_attempts} failed attempts")
#             failed_attempts = 0
#         except Exception:
#             failed_attempts += 1
#             pass
#     assingment_dicts.append(initial_partition.assignment.to_dict())

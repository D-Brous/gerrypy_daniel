from gurobipy import *
from gurobipy import Model, Env, GRB, min_, quicksum
import networkx as nx
import numpy as np
import random
import math

from data.df import DemoDataFrame
from data.graph import Graph
from optimize.tree import SHPNode
from data.config import SHPConfig
from constants import IPStr

env = Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()


class IPSetup:
    def __init__(
        self,
        config: SHPConfig,
        subregion_df: DemoDataFrame,
        G: Graph,
        lengths: np.ndarray,
        node: SHPNode,
        child_centers: dict[int, int],
        ideal_pop: float,
    ):
        self.config = config
        self.subregion_df = subregion_df
        self.G = G.get_subgraph(node.subregion)
        self.lengths = lengths
        self.edge_dists = self.G.get_edge_dists(
            child_centers, config, node.subregion
        )
        self.node = node
        self.child_centers = child_centers
        self.col = config.col
        self.ideal_pop = ideal_pop

        self.beta = self.config.beta
        self.epsilon = self.config.epsilon
        self.pop_dict = self.subregion_df["POP"].to_dict()

        self.costs = self.make_costs()
        self.connectivity_sets = self.make_edge_dists_connectivity_sets()
        self.pop_bounds = self.make_pop_bounds()

    def make_costs(self) -> dict[int, dict[int, float]]:
        """Creates the costs for the compactness part of the objective
        function

        Returns:
            (dict[int, dict[int, float]]): dict of center : cgu : cost
            where cost is the cost coefficient for the variable
            districts[center][cgu]
        """
        population = self.subregion_df["POP"].values
        index = list(self.subregion_df.index)
        costs = self.lengths[np.ix_(list(self.child_centers.keys()), index)] * (
            (population / 1000)
        )

        costs **= self.choose_alpha()
        return {
            center: {
                index[cgu_ix]: cost
                for cgu_ix, cost in enumerate(costs[center_ix])
            }
            for center_ix, center in enumerate(self.child_centers)
        }

    def make_edge_dists_connectivity_sets(
        self,
    ) -> dict[int, dict[int, list[int]]]:
        """
        Creates the dict that stores the information necessary to
        build connectivity constraints in an ip

        Returns:
            (dict[int, dict[int, list[int]]]): dict of center : node :
            constr_set, where constr_set is a list of neighbors of node
            whose edge distance to the center is lower than that of node
        """
        connectivity_set = {}
        for center in self.child_centers:
            connectivity_set[center] = {}
            for node in self.edge_dists[center]:
                constr_set = []
                dist = self.edge_dists[center][node]
                for nbor in self.G[node]:
                    if self.edge_dists[center][nbor] < dist:
                        constr_set.append(nbor)
                connectivity_set[center][node] = constr_set
        return connectivity_set

    def make_pop_bounds(self) -> dict[int, dict[str, int]]:
        """
        Computes the upper and lower population bounds using the centers
        and their capacities

        Returns: (dict[int, dict[str, int]]) of center : info_dict where
            info_dict stores upper and lower population bounds and #
            districts as values keyed by corresponding strings
        """
        pop_deviation = self.ideal_pop * self.config.population_tolerance
        pop_bounds = {}
        # Make the bounds for an area considering # area districts and tree level
        for center, n_child_districts in self.child_centers.items():
            if n_child_districts in self.config.final_partition_range:
                levels_to_leaf = 1
            else:
                levels_to_leaf = max(math.ceil(math.log2(n_child_districts)), 1)
            distr_pop = self.ideal_pop * n_child_districts

            ub = distr_pop + pop_deviation / levels_to_leaf
            lb = distr_pop - pop_deviation / levels_to_leaf

            pop_bounds[center] = {
                "ub": ub,
                "lb": lb,
                "n_districts": n_child_districts,
            }

        return pop_bounds

    def get_ideal_vap(self):
        return self.subregion_df["VAP"].sum() / self.node.n_districts

    def choose_alpha(self):
        return 1 + random.random()

    def get_cvap_dict(self):
        return self.subregion_df[self.col].to_dict()

    def get_vap_dict(self):
        return self.subregion_df["VAP"].to_dict()

    def get_warm_start(self, xs):
        return {
            center: {cgu: var.X for cgu, var in xs["districts"][center].items()}
            for center in xs["districts"]
        }

    def make_partial_ip(self):
        partition_ip = Model("partition", env=env)

        # Create the variables
        districts = {}
        for center, cgus in self.costs.items():
            districts[center] = {}
            for cgu in cgus:
                districts[center][cgu] = partition_ip.addVar(
                    vtype=GRB.BINARY, obj=self.costs[center][cgu]
                )

        # Each tract belongs to exactly one district
        for j in self.pop_dict:
            partition_ip.addLConstr(
                quicksum(
                    districts[i][j] for i in districts if j in districts[i]
                )
                == 1,
                name="exactlyOne",
            )
        # Population tolerances
        for i in districts:
            partition_ip.addLConstr(
                quicksum(
                    districts[i][j] * self.pop_dict[j] for j in districts[i]
                )
                >= self.pop_bounds[i]["lb"],
                name="x%s_minsize" % i,
            )

            partition_ip.addLConstr(
                quicksum(
                    districts[i][j] * self.pop_dict[j] for j in districts[i]
                )
                <= self.pop_bounds[i]["ub"],
                name="x%s_maxsize" % i,
            )

        # Connectivity
        for center, sp_sets in self.connectivity_sets.items():
            for node, sp_set in sp_sets.items():
                if center == node:
                    continue
                partition_ip.addLConstr(
                    districts[center][node]
                    <= quicksum(districts[center][nbor] for nbor in sp_set)
                )

        partition_ip.Params.LogToConsole = 0
        # partition_ip.Params.TimeLimit = len(population) / 200
        partition_ip.Params.MIPGap = self.config.ip_gap_tol
        if self.config.ip_timeout is not None:
            partition_ip.Params.TimeLimit = (
                len(self.pop_dict) * self.config.ip_timeout
            )

        return partition_ip, districts

    def make_base_ip(self):
        partition_ip, districts = self.make_partial_ip()

        partition_ip.setObjective(
            quicksum(
                districts[i][j] * self.costs[i][j]
                for i in self.costs
                for j in self.costs[i]
            ),
            GRB.MINIMIZE,
        )
        partition_ip.update()
        xs = {"districts": districts}
        return partition_ip, xs

    def make_maj_cvap_approx_ip(self):
        partition_ip, districts = self.make_partial_ip()

        # Create the extra variables
        maj_cvap = {}
        for center in self.costs.keys():
            maj_cvap[center] = partition_ip.addVar(vtype=GRB.BINARY)

        # Approximately majority cvap
        cvap = self.get_cvap_dict()
        ideal_vap = self.get_ideal_vap()
        for i in districts:
            partition_ip.addLConstr(
                quicksum(2 * districts[i][j] * cvap[j] for j in districts[i])
                >= (1 + self.epsilon) * ideal_vap * maj_cvap[i]
            )

        # Set objective
        partition_ip.setObjective(
            self.beta
            * quicksum(
                districts[i][j] * self.costs[i][j]
                for i in self.costs
                for j in self.costs[i]
            )
            - (1 - self.beta) * quicksum(maj_cvap[i] for i in self.costs),
            GRB.MINIMIZE,
        )

        partition_ip.update()
        xs = {"districts": districts, "maj_cvap": maj_cvap}
        return partition_ip, xs

    def make_maj_cvap_explicit_ip(self):
        partition_ip, districts = self.make_partial_ip()

        # Create the extra variables
        maj_cvap = {}
        dummy_vars = {}
        prods = {}
        for center, cgus in self.costs.items():
            prods[center] = {}
            dummy_vars[center] = {}
            for cgu in cgus:
                prods[center][cgu] = partition_ip.addVar(
                    vtype=GRB.BINARY,
                )
                dummy_vars[center][cgu] = partition_ip.addVar(
                    vtype=GRB.BINARY,
                )
            maj_cvap[center] = partition_ip.addVar(vtype=GRB.BINARY)

        # Majority cvap
        cvap = self.get_cvap_dict()
        vap = self.get_vap_dict()
        for i in districts:
            partition_ip.addLConstr(
                quicksum(2 * districts[i][j] * cvap[j] for j in districts[i])
                >= quicksum(prods[i][j] * vap[j] for j in districts[i]) + 1
            )
            for j in districts[i]:
                partition_ip.addLConstr(prods[i][j] <= districts[i][j])
                partition_ip.addLConstr(prods[i][j] <= maj_cvap[i])
                partition_ip.addLConstr(
                    prods[i][j] >= districts[i][j] - 1 + dummy_vars[i][j]
                )
                partition_ip.addLConstr(
                    prods[i][j] >= maj_cvap[i] - dummy_vars[i][j]
                )

        # Set objective
        partition_ip.setObjective(
            self.beta
            * quicksum(
                districts[i][j] * self.costs[i][j]
                for i in self.costs
                for j in self.costs[i]
            )
            - (1 - self.beta) * quicksum(maj_cvap[i] for i in self.costs),
            GRB.MINIMIZE,
        )

        partition_ip.update()
        xs = {"districts": districts, "maj_cvap": maj_cvap, "prods": prods}
        return partition_ip, xs

    def make_maj_cvap_exact_ip(self):
        partition_ip, districts = self.make_partial_ip()

        # Create the extra variables
        maj_cvap = {}
        prods = {}
        for center, cgus in self.costs.items():
            prods[center] = {}
            for cgu in cgus:
                prods[center][cgu] = partition_ip.addVar(
                    vtype=GRB.BINARY,
                )
            maj_cvap[center] = partition_ip.addVar(vtype=GRB.BINARY)

        # Majority cvap
        cvap = self.get_cvap_dict()
        vap = self.get_vap_dict()
        for i in districts:
            partition_ip.addConstr(
                quicksum(2 * districts[i][j] * cvap[j] for j in districts[i])
                >= quicksum(prods[i][j] * vap[j] for j in districts[i]) + 1
            )
            for j in districts[i]:
                partition_ip.addConstr(
                    prods[i][j] == min_(districts[i][j], maj_cvap[i])
                )

        # Set objective
        partition_ip.setObjective(
            (self.beta / 1000)
            * quicksum(
                districts[i][j] * self.costs[i][j]
                for i in self.costs
                for j in self.costs[i]
            )
            - (1 - self.beta) * quicksum(maj_cvap[i] for i in self.costs),
            GRB.MINIMIZE,
        )

        partition_ip.update()
        xs = {"districts": districts, "maj_cvap": maj_cvap, "prods": prods}
        return partition_ip, xs

    def make(self, ip: IPStr) -> tuple[Model, dict[str, dict]]:
        if ip == "base":
            return self.make_base_ip()
        elif ip == "maj_cvap_approx":
            return self.make_maj_cvap_approx_ip()
        elif ip == "maj_cvap_explicit":
            return self.make_maj_cvap_explicit_ip()
        elif ip == "maj_cvap_exact":
            return self.make_maj_cvap_exact_ip()
        else:
            raise ValueError(f"{ip}, the given ip string, is not recognized.")

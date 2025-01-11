from gurobipy import *
from gurobipy import quicksum
from gurobipy import Model
from gurobipy import GRB
from gurobipy import min_
import networkx as nx
import numpy as np
import random
import math

from data.df import DemoDataFrame
from data.graph import Graph
from optimize.tree import SHPNode
from data.config import SHPConfig
from constants import IPStr


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
        self.edge_dists = self.G.get_edge_dists(child_centers)
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
        costs = self.lengths[np.ix_(self.child_centers, index)] * (
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
        partition_ip = Model("partition")

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
        partition_ip.Params.MIPGap = self.config.IP_gap_tol
        if self.config.use_time_limit:
            partition_ip.Params.TimeLimit = len(self.pop_dict) / 200

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


'''
def make_partition_IP_maj_cvap_approximate(ip_setup: IPSetup):
    """
    Creates the Gurobi model to partition a region.
    Args:
        costs: (dict) {center: {tract: cost}} The keys of the outer dict
            are the only centers that are considered in the partition.
             I.e. len(lengths) = z.
        edge_dists: (dict) {center: {tract: hop_distance}} Same as lengths but
            value is the shortest path hop distance (# edges between i and j)
        G: (nx.Graph) The block adjacency graph
        population: (dict) {tract int id: population}
        pop_bounds: (dict) {center int id: {'lb': district population lower
            bound, 'ub': tract population upper bound}
        alpha: (int) The exponential cost term

    Returns: (tuple (Gurobi partition model, model variables as a dict)

    """
    partition_ip = Model("partition")
    districts = {}
    maj_black = {}
    # Create the variables
    for center, cgus in ip_setup.costs.items():
        districts[center] = {}
        for cgu in cgus:
            districts[center][cgu] = partition_ip.addVar(
                vtype=GRB.BINARY, obj=ip_setup.costs[center][cgu]
            )
        maj_black[center] = partition_ip.addVar(vtype=GRB.BINARY)

    # Each tract belongs to exactly one district
    for j in ip_setup.population:
        partition_ip.addLConstr(
            quicksum(districts[i][j] for i in districts if j in districts[i])
            == 1,
            name="exactlyOne",
        )
    # Population tolerances
    for i in districts:
        partition_ip.addLConstr(
            quicksum(
                districts[i][j] * ip_setup.population[j] for j in districts[i]
            )
            >= ip_setup.pop_bounds[i]["lb"],
            name="x%s_minsize" % i,
        )

        partition_ip.addLConstr(
            quicksum(
                districts[i][j] * ip_setup.population[j] for j in districts[i]
            )
            <= ip_setup.pop_bounds[i]["ub"],
            name="x%s_maxsize" % i,
        )

    # connectivity
    for center, sp_sets in ip_setup.connectivity_sets.items():
        for node, sp_set in sp_sets.items():
            if center == node:
                continue
            partition_ip.addLConstr(
                districts[center][node]
                <= quicksum(districts[center][nbor] for nbor in sp_set)
            )

    cvap = ip_setup.get_cvap_col()
    ideal_vap = ip_setup.get_ideal_vap()
    # majority black
    for i in districts:
        partition_ip.addLConstr(
            quicksum(2 * districts[i][j] * cvap[j] for j in districts[i])
            >= (1 + ip_setup.epsilon) * ideal_vap * maj_black[i]
        )

    partition_ip.setObjective(
        ip_setup.beta
        * quicksum(
            districts[i][j] * ip_setup.costs[i][j]
            for i in ip_setup.costs
            for j in ip_setup.costs[i]
        )
        - (1 - ip_setup.beta) * quicksum(maj_black[i] for i in ip_setup.costs),
        GRB.MINIMIZE,
    )
    partition_ip.Params.LogToConsole = 0
    # partition_ip.Params.TimeLimit = len(population) / 200
    partition_ip.update()

    xs = {"districts": districts, "maj_black": maj_black}
    return partition_ip, xs


def make_partition_IP_MajBlack_explicit(ip_setup: IPSetup):
    """
    Creates the Gurobi model to partition a region while maximizing the
        number of majority black subregions created.
    Args:
        ip_setup: (IPSetup) a setup configuration for all relevant IPs

    Returns: (tuple (Gurobi partition model, model variables as a dict)

    """
    partition_ip = Model("partition")
    districts = {}
    maj_black = {}
    dummy_vars = {}
    prods = {}
    # Create the variables
    for center, cgus in costs.items():
        districts[center] = {}
        prods[center] = {}
        dummy_vars[center] = {}
        for cgu in cgus:
            districts[center][cgu] = partition_ip.addVar(
                vtype=GRB.BINARY, obj=costs[center][cgu]
            )
            prods[center][cgu] = partition_ip.addVar(
                vtype=GRB.BINARY,
            )
            dummy_vars[center][cgu] = partition_ip.addVar(
                vtype=GRB.BINARY,
            )
        maj_black[center] = partition_ip.addVar(vtype=GRB.BINARY)

    # Each tract belongs to exactly one district
    for j in population:
        partition_ip.addLConstr(
            quicksum(districts[i][j] for i in districts if j in districts[i])
            == 1,
            name="exactlyOne",
        )
    # Population tolerances
    for i in districts:
        partition_ip.addLConstr(
            quicksum(districts[i][j] * population[j] for j in districts[i])
            >= pop_bounds[i]["lb"],
            name="x%s_minsize" % i,
        )

        partition_ip.addLConstr(
            quicksum(districts[i][j] * population[j] for j in districts[i])
            <= pop_bounds[i]["ub"],
            name="x%s_maxsize" % i,
        )

    # connectivity
    for center, sp_sets in connectivity_sets.items():
        for node, sp_set in sp_sets.items():
            if center == node:
                continue
            partition_ip.addLConstr(
                districts[center][node]
                <= quicksum(districts[center][nbor] for nbor in sp_set)
            )

    # majority black
    for i in districts:
        partition_ip.addLConstr(
            quicksum(2 * districts[i][j] * BVAP[j] for j in districts[i])
            >= quicksum(prods[i][j] * VAP[j] for j in districts[i]) + 1
        )
        for j in districts[i]:
            partition_ip.addLConstr(prods[i][j] <= districts[i][j])
            partition_ip.addLConstr(prods[i][j] <= maj_black[i])
            partition_ip.addLConstr(
                prods[i][j] >= districts[i][j] - 1 + dummy_vars[i][j]
            )
            partition_ip.addLConstr(
                prods[i][j] >= maj_black[i] - dummy_vars[i][j]
            )

    partition_ip.setObjective(
        alpha
        * quicksum(
            districts[i][j] * costs[i][j] for i in costs for j in costs[i]
        )
        - (1 - alpha) * quicksum(maj_black[i] for i in costs),
        GRB.MINIMIZE,
    )
    partition_ip.Params.LogToConsole = 0
    # partition_ip.Params.TimeLimit = len(population) / 200
    partition_ip.update()

    xs = {"districts": districts, "maj_black": maj_black, "prods": prods}
    return partition_ip, xs


def make_partition_IP_MajBlack(
    costs, connectivity_sets, population, BVAP, VAP, pop_bounds, alpha
):
    """
    Creates the Gurobi model to partition a region.
    Args:
        costs: (dict) {center: {tract: cost}} The keys of the outer dict
            are the only centers that are considered in the partition.
             I.e. len(lengths) = z.
        edge_dists: (dict) {center: {tract: hop_distance}} Same as lengths but
            value is the shortest path hop distance (# edges between i and j)
        G: (nx.Graph) The block adjacency graph
        population: (dict) {tract int id: population}
        pop_bounds: (dict) {center int id: {'lb': district population lower
            bound, 'ub': tract population upper bound}
        alpha: (int) The exponential cost term

    Returns: (tuple (Gurobi partition model, model variables as a dict)

    """
    partition_ip = Model("partition")
    districts = {}
    maj_black = {}
    prods = {}
    # Create the variables
    for center, cgus in costs.items():
        districts[center] = {}
        prods[center] = {}
        for cgu in cgus:
            districts[center][cgu] = partition_ip.addVar(
                vtype=GRB.BINARY, obj=costs[center][cgu]
            )
            prods[center][cgu] = partition_ip.addVar(
                vtype=GRB.BINARY,
            )
        maj_black[center] = partition_ip.addVar(vtype=GRB.BINARY)

    # Each tract belongs to exactly one district
    for j in population:
        partition_ip.addConstr(
            quicksum(districts[i][j] for i in districts if j in districts[i])
            == 1,
            name="exactlyOne",
        )
    # Population tolerances
    for i in districts:
        partition_ip.addConstr(
            quicksum(districts[i][j] * population[j] for j in districts[i])
            >= pop_bounds[i]["lb"],
            name="x%s_minsize" % i,
        )

        partition_ip.addConstr(
            quicksum(districts[i][j] * population[j] for j in districts[i])
            <= pop_bounds[i]["ub"],
            name="x%s_maxsize" % i,
        )

    # connectivity
    for center, sp_sets in connectivity_sets.items():
        for node, sp_set in sp_sets.items():
            if center == node:
                continue
            partition_ip.addConstr(
                districts[center][node]
                <= quicksum(districts[center][nbor] for nbor in sp_set)
            )

    # majority black
    for i in districts:
        partition_ip.addConstr(
            quicksum(2 * districts[i][j] * BVAP[j] for j in districts[i])
            >= quicksum(prods[i][j] * VAP[j] for j in districts[i]) + 1
        )
        for j in districts[i]:
            partition_ip.addConstr(
                prods[i][j] == min_(districts[i][j], maj_black[i])
            )

    partition_ip.setObjective(
        (alpha / 1000)
        * quicksum(
            districts[i][j] * costs[i][j] for i in costs for j in costs[i]
        )
        - (1 - alpha) * quicksum(maj_black[i] for i in costs),
        GRB.MINIMIZE,
    )
    partition_ip.Params.LogToConsole = 0
    # partition_ip.Params.TimeLimit = len(population) / 200
    partition_ip.update()

    xs = {"districts": districts, "maj_black": maj_black, "prods": prods}
    return partition_ip, xs


def make_partition_IP(costs, connectivity_sets, population, pop_bounds):
    """
    Creates the Gurobi model to partition a region.
    Args:
        costs: (dict) {center: {tract: cost}} The keys of the outer dict
            are the only centers that are considered in the partition.
             I.e. len(lengths) = z.
        edge_dists: (dict) {center: {tract: hop_distance}} Same as lengths but
            value is the shortest path hop distance (# edges between i and j)
        G: (nx.Graph) The block adjacency graph
        population: (dict) {tract int id: population}
        pop_bounds: (dict) {center int id: {'lb': district population lower
            bound, 'ub': tract population upper bound}
        alpha: (int) The exponential cost term

    Returns: (tuple (Gurobi partition model, model variables as a dict)

    """
    partition_ip = Model("partition")
    districts = {}
    # Create the variables
    for center, cgus in costs.items():
        districts[center] = {}
        for cgu in cgus:
            districts[center][cgu] = partition_ip.addVar(
                vtype=GRB.BINARY, obj=costs[center][cgu]
            )
    # Each tract belongs to exactly one district
    for j in population:
        partition_ip.addConstr(
            quicksum(districts[i][j] for i in districts if j in districts[i])
            == 1,
            name="exactlyOne",
        )
    # Population tolerances
    for i in districts:
        partition_ip.addConstr(
            quicksum(districts[i][j] * population[j] for j in districts[i])
            >= pop_bounds[i]["lb"],
            name="x%s_minsize" % i,
        )

        partition_ip.addConstr(
            quicksum(districts[i][j] * population[j] for j in districts[i])
            <= pop_bounds[i]["ub"],
            name="x%s_maxsize" % i,
        )

    # connectivity
    for center, sp_sets in connectivity_sets.items():
        for node, sp_set in sp_sets.items():
            if center == node:
                continue
            partition_ip.addConstr(
                districts[center][node]
                <= quicksum(districts[center][nbor] for nbor in sp_set)
            )

    partition_ip.setObjective(
        quicksum(
            districts[i][j] * costs[i][j] for i in costs for j in costs[i]
        ),
        GRB.MINIMIZE,
    )
    partition_ip.Params.LogToConsole = 0
    # partition_ip.Params.TimeLimit = len(population) / 200
    partition_ip.update()

    xs = {"districts": districts}
    return partition_ip, xs


def edge_distance_connectivity_sets(edge_distance, G):
    connectivity_set = {}
    for center in edge_distance:
        connectivity_set[center] = {}
        for node in edge_distance[center]:
            constr_set = []
            dist = edge_distance[center][node]
            for nbor in G[node]:
                if edge_distance[center][nbor] < dist:
                    constr_set.append(nbor)
            connectivity_set[center][node] = constr_set
    return connectivity_set


def make_partition_IP_vectorized(
    cost_coeffs, spt_matrix, population, pop_bounds
):
    """
    Creates the Gurobi model to partition a region.
    Args:
        cost_coeffs: (dict) {center: {tract: distance}} The keys of the outer dict
            are the only centers that are considered in the partition.
             I.e. len(lengths) = z.
        spt_matrix: (np.array) nonzero elements of row (i * B + j) are
            equal to the set S_ij
        G: (nx.Graph) The block adjacency graph
        population: (dict) {tract int id: population}
        pop_bounds: (dict) {center int id: {'lb': district population lower
            bound, 'ub': tract population upper bound}

    Returns: (tuple (Gurobi partition model, Gurobi MVar)

    """
    partition_ip = Model("partition")
    n_centers, n_blocks = cost_coeffs.shape

    # Assigment variables
    assignment = partition_ip.addMVar(
        shape=n_centers * n_blocks, vtype=GRB.BINARY, obj=cost_coeffs.flatten()
    )

    population_matrix = np.zeros((n_centers, n_blocks * n_centers))
    for i in range(n_centers):
        population_matrix[i, i * n_blocks : (i + 1) * n_blocks] = population

    # Population balance
    partition_ip.addConstr(
        population_matrix @ assignment <= pop_bounds[:, 1]
    )
    partition_ip.addConstr(
        population_matrix @ assignment >= pop_bounds[:, 0]
    )

    # Strict covering
    partition_ip.addConstr(
        np.tile(np.eye(n_blocks), (1, n_centers)) @ assignment == 1
    )

    # Subtree of shortest path tree
    partition_ip.addConstrs(
        spt_matrix[n_blocks * c : n_blocks * (c + 1), :]
        @ assignment[n_blocks * c : n_blocks * (c + 1)]
        >= assignment[n_blocks * c : n_blocks * (c + 1)]
        for c in range(n_centers)
    )

    partition_ip.Params.LogToConsole = 0
    # partition_ip.Params.TimeLimit = len(population) / 20
    partition_ip.update()

    return partition_ip, assignment
'''

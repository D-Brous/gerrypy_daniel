import sys

sys.path.append(".")

from optimize.generate import ColumnGenerator

# from analyze.districts import *
# from optimize.dir_processing import district_df_of_tree_dir
import constants
from copy import deepcopy
import time
import os
import numpy as np
import json

# from optimize.master import *
from gurobipy import GRB

# from data.load import *
# from optimize.improvement import *
import pickle
import scipy as sp
import pandas as pd

# from analyze.maj_black import majority_black, maj_black_logging_info
# from analyze.feasibility import check_feasibility
# from analyze.maj_black import n_maj_black
from data.df import DemoDataFrame
from data.config import SHPConfig
from data.partition import Partition, Partitions
from optimize.logging import Logger
from optimize.tree import SHPTree


def callback(model, where):
    if where == GRB.Callback.MIP:
        time = model.cbGet(GRB.Callback.RUNTIME)
        if model._last_callback_time + model._callback_time_interval <= time:
            model._last_callback_time = time
            num_feasible = model.cbGet(GRB.Callback.MIP_SOLCNT)
            # num_unexplored = model.cbGet(GRB.Callback.MIP_NODLFT)
            best_obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            best_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
            phase = model.cbGet(GRB.Callback.MIP_PHASE)
            print(
                f"{time} s: In phase {phase}, {num_feasible} feasible sols found, curr best obj is {best_obj}, curr best bound is {best_bound}\n"
            )


def save_object(obj, filepath):
    with open(filepath, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def save_matrix(mtx, filepath):
    sparse_mtx = sp.sparse.csc_matrix(mtx)
    sp.sparse.save_npz(filepath, sparse_mtx)


class SHP:
    """
    SHP class to test different generation configurations.
    """

    def __init__(self, config: SHPConfig):
        """
        Initialized with configuration dict
        Args:
            config: (dict) the following are the required keys
                state: (str) 2 letter abbreviation
                n_districts: (int)
                population_tolerance: (float) ideal population +/- factor epsilon
                max_sample_tries: (int) number of attempts at each node
                n_samples: (int) the fan-out split width
                n_root_samples: (int) the split width of the root node w
                max_n_splits: (int) max split size z_max
                min_n_splits: (int) min split size z_min
                max_split_population_difference: (float) maximum
                    capacity difference between 2 sibling nodes
                event_logging: (bool) log events for visualization
                verbose: (bool) print runtime information
                selection_method: (str) seed selection method to use
                perturbation_scale: (float) pareto distribution parameter
                n_random_seeds: (int) number of fixed seeds in seed selection
                capacities: (str) style of capacity matching/computing
                capacity_weights: (str) 'voronoi' or 'fractional'
                IP_gap_tol: (float) partition IP gap tolerance
                IP_timeout: (float) maximum seconds to spend solving IP
                callback_time_interval: (int) interval in seconds between printouts to console with some information during master problem(s) or
                                        (NoneType) None if we don't want master problem console printouts
                'granularity': (str) level of granularity of the census data on which to operate


        """
        self.config = config
        self.save_path = self.config.save_path
        self.demo_df = DemoDataFrame.from_config(config)

        # self.n_cgus = len(self.demo_df["GEOID"])
        self.partitions = Partitions()
        if self.config.n_beta_reoptimize_steps > 0:
            self.beta_reopt_partitions = Partitions()

    def shp(self, save_config=True, save_tree=True, save_partitions=True):
        """Performs generation and finds solutions for each root partition.
        Also performs beta reoptimization if specified.
        """
        config_file_path = os.path.join(self.save_path, "config.json")
        tree_file_path = os.path.join(self.save_path, "tree.pickle")
        partitions_path = os.path.join(self.save_path, "partitions")
        debug_file_path = os.path.join(self.save_path, "debug.txt")
        if save_config or save_tree or save_partitions:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            elif (
                os.path.exists(config_file_path)
                or os.path.exists(tree_file_path)
                or os.path.exists(partitions_path)
            ):
                raise NameError(
                    "The current save path has already been used. Please give the save path an unused name."
                )
        if save_partitions:
            os.mkdir(partitions_path)
        if save_config:
            self.config.save_to(config_file_path)

        tree = SHPTree(self.config)
        logger = Logger(True, True, debug_file_path)
        cg = ColumnGenerator(self.config, tree, logger)

        logger.print_generation_initiation()

        for root_partition_id in range(self.config.n_root_samples):
            internal_nodes, leaf_nodes = cg.generate_root_partition_tree()
            if save_tree:
                tree.save_to(tree_file_path)

            n_final_ips = len(self.config.final_partition_ips)
            for ip_ix, ip_str in enumerate(self.config.final_partition_ips):
                ip_leaf_nodes = tree.get_leaf_nodes_from_ip(
                    ip_str, root_partition_id
                )
                solution_nodes = tree.get_solution_nodes_dp(
                    internal_nodes,
                    ip_leaf_nodes,
                    root_partition_id,
                    self.config.col,
                )
                partition_id = n_final_ips * root_partition_id + ip_ix
                partition = tree.get_partition(solution_nodes)
                self.partitions.set(partition_id, partition)

                if self.config.n_beta_reoptimize_steps > 0:
                    beta_reopt_solution_nodes = cg.beta_reoptimize(
                        solution_nodes,
                        internal_nodes,
                        root_partition_id,
                        ip_str,
                        self.config.col,
                        self.config.n_beta_reoptimize_steps,
                    )
                    beta_reopt_partition = tree.get_partition(
                        beta_reopt_solution_nodes
                    )
                    self.beta_reopt_partitions.set(
                        partition_id, beta_reopt_partition
                    )
                if save_partitions:
                    self.partitions.to_csv(
                        os.path.join(partitions_path, "shp.csv"),
                        self.config,
                    )
                    if self.config.n_beta_reoptimize_steps > 0:
                        self.beta_reopt_partitions.to_csv(
                            os.path.join(
                                partitions_path,
                                "shp_beta_reopt.csv",
                            ),
                            self.config,
                        )

        logger.log_generation_completion(tree)
        logger.print_generation_completion(tree)
        logger.close()
        # TODO: put in a solution feasibility check


'''

def run(self):
        if self.config.mode == "partition" or self.config.mode == "both":
            if self.config.assignments_file_name is not None:
                os.mkdir(self.save_path)
                os.mkdir(os.path.join(self.save_path, "assignments"))
            if self.config.save_tree:
                os.mkdir(os.path.join(self.save_path, "tree"))
            if self.config.save_cdms:
                os.mkdir(os.path.join(self.save_path, "cdms"))
            if self.config.save_district_adj_graphs:
                os.mkdir(os.path.join(self.save_path, "district_adj_graphs"))
            self.shp()
        elif self.config.mode == "master":
            self.save_path = self.get_save_path(
                time_str=self.config.tree_time_str
            )
            self.assignments_df = self.master_solutions()
        else:
            raise ValueError("mode value is invalid")
        if self.config.mode == "both" or self.config.mode == "master":
            G = load_adjacency_graph(self.config)
            if self.config.subregion is not None:
                G = G.subgraph(self.config.subregion)
            check_feasibility(
                self.config,
                self.assignments_df,
                self.demo_df,
                G,
                self.config.verbose,
            )
            if self.config.beta_reoptimize_steps > 0:
                return self.assignments_df, self.beta_reopt_assignments_df
            else:
                return self.assignments_df

def get_sample_trial_leaf_nodes(
        self,
        maj_black_partition_ix,
        root,
        sample_internal_nodes,
        sample_leaf_nodes,
    ):
        maj_black_partitioned_nodes = {}
        sample_trial_leaf_nodes = {}
        for node in sample_leaf_nodes.values():
            parent_id = node.parent_id
            if (
                parent_id == 0
                and root.n_districts in self.config.final_partition_range
            ):
                maj_black_partitioned_nodes[parent_id] = root
            elif (
                sample_internal_nodes[parent_id].n_districts
                in self.config.final_partition_range
            ):
                maj_black_partitioned_nodes[parent_id] = sample_internal_nodes[
                    parent_id
                ]
            else:
                sample_trial_leaf_nodes[node.id] = node
        for node in maj_black_partitioned_nodes.values():
            ix = 0
            for maj_black_partitions_ixs in node.partitions_used:
                for mbp_ix in maj_black_partitions_ixs:
                    if mbp_ix == maj_black_partition_ix:
                        sample_trial_leaf_nodes.update(
                            {
                                id: sample_leaf_nodes[id]
                                for id in node.children_ids[ix]
                            }
                        )
                    ix += 1
        return sample_trial_leaf_nodes

    def get_solution_set_dp(
        self,
        root,
        internal_nodes,
        leaf_nodes,
        root_partition_ix,
        root_partition,
    ):
        if self.config.verbose:
            print(
                "\n-------------Solving master for root sample number %d-------------\n"
                % root_partition_ix
            )
        nodes = {**internal_nodes, **leaf_nodes}
        dp_queue = []
        parent_layer = [root]
        children_layer = [
            internal_nodes[id]
            for id in root_partition
            if id in nodes and nodes[id].n_districts != 1
        ]
        # children_layer = [internal_nodes[id] for partition in root.children_ids for id in partition if nodes[id].n_districts != 1]
        while len(children_layer) > 0:
            dp_queue += children_layer
            parent_layer = children_layer
            children_layer = [
                nodes[id]
                for node in parent_layer
                for partition in node.children_ids
                for id in partition
                if id in nodes and nodes[id].n_districts != 1
            ]

        for node in leaf_nodes.values():
            node.best_subtree = (
                [node.id],
                int(node.is_maj_black(self.demo_df)),
            )

        for i in range(len(dp_queue) - 1, -1, -1):
            current_node = dp_queue[i]
            best_subtree_ids = []
            best_subtree_score = -1
            for partition in current_node.children_ids:
                try:
                    sample_score = sum(
                        nodes[id].best_subtree[1] for id in partition
                    )
                    if best_subtree_score < sample_score:
                        best_subtree_ids = [
                            subtree_id
                            for id in partition
                            for subtree_id in nodes[id].best_subtree[0]
                        ]
                        best_subtree_score = sample_score
                except:
                    continue
            current_node.best_subtree = (best_subtree_ids, best_subtree_score)
        return {
            subtree_id
            for node_id in root_partition
            for subtree_id in nodes[node_id].best_subtree[0]
        }

    def get_solution_ser(self, solution_set, leaf_nodes):
        assignment_ser = pd.Series(index=self.assignments_df.index)
        for district_ix, node_id in enumerate(solution_set):
            assignment_ser.loc[leaf_nodes[node_id].subregion] = district_ix
        return assignment_ser

    def beta_reoptimize(
        self,
        cg,
        solution_set,
        root,
        internal_nodes,
        leaf_nodes,
        root_partition_ix,
        root_partition,
        n_steps,
    ):
        # max_n_maj_black = n_maj_black(assignment_ser_to_dict(assignment_ser), self.demo_df)
        # best_assignment_ser = assignment_ser
        # print(f'Initial n_maj_black: {max_n_maj_black}')
        # lb = 0
        # ub = 1
        cg.config.use_time_limit = False
        self.config.verbose = False
        # Dictionary which stores [parent_id : tuple of information] where the tuple stores the corresponding node,
        # the list of centers of the optimal sample, the current lower and upper bounds of beta on that sample, and
        # the number of majority black districts in the current best sample, and the leaf nodes for the current best
        # sample, respectively.
        beta = cg.config.beta
        parent_information = {}
        new_solution_set = set()
        new_leaf_nodes = {}
        for leaf_id in solution_set:
            parent_id = leaf_nodes[leaf_id].parent_id
            if parent_id in parent_information:
                parent_information[parent_id][1].update(
                    {leaf_nodes[leaf_id].center: 1}
                )
                parent_information[parent_id][5].append(leaf_nodes[leaf_id])
                parent_information[parent_id][4] += int(
                    leaf_nodes[leaf_id].is_maj_black(self.demo_df)
                )
            else:
                parent_node = (
                    root if parent_id == 0 else internal_nodes[parent_id]
                )
                if parent_node.n_districts in self.config.final_partition_range:
                    parent_information[parent_id] = [
                        parent_node,
                        {leaf_nodes[leaf_id].center: 1},
                        0,
                        1,
                        int(leaf_nodes[leaf_id].is_maj_black(self.demo_df)),
                        [leaf_nodes[leaf_id]],
                    ]
                else:
                    new_solution_set.update({leaf_id})
                    new_leaf_nodes.update({leaf_id: leaf_nodes[leaf_id]})

        for parent_id, information in parent_information.items():
            for step in range(n_steps):
                parent_node = information[0]
                beta = (information[2] + information[3]) / 2
                child_nodes, _ = cg.make_partition(
                    self.demo_df.loc[parent_node.subregion],
                    parent_node,
                    information[1],
                    0,
                )
                n_maj_black = sum(
                    int(node.is_maj_black(self.demo_df)) for node in child_nodes
                )
                if n_maj_black < information[4]:
                    parent_information[parent_id][3] = beta
                else:
                    parent_information[parent_id][2] = beta
                    parent_information[parent_id][4] = n_maj_black
                    parent_information[parent_id][5] = child_nodes
            new_solution_set.update(
                {leaf_node.id for leaf_node in parent_information[parent_id][5]}
            )
            new_leaf_nodes.update(
                {
                    leaf_node.id: leaf_node
                    for leaf_node in parent_information[parent_id][5]
                }
            )

        self.config.verbose = True
        return new_solution_set, new_leaf_nodes

    def selected_districts(self, solution_ixs, cdm):
        selected_districts = np.zeros(self.n_cgus, dtype=int)
        for district in range(len(solution_ixs)):
            for cgu, cgu_in_district in enumerate(
                cdm.T[solution_ixs[district]]
            ):
                if cgu_in_district:
                    selected_districts[cgu] = district
        return selected_districts

    def export_solutions(
        self, sol_dict, demo_df, cdm, sol_tree, internal_nodes
    ):
        """
        Creates a dataframe with each block matched to a district based on the IP solution
        Args:
            solutions: (dict) of solutions outputted by IP
            demo_df: (pd DataFrame) with state data
            cdm: (np.array) n x d matrix where a_ij = 1 when block i appears in district j.
            sol_tree: list of SHPNodes representing the leaf nodes and their ancestors #TODO make this work for multiple trials

        Returns: Dataframe mapping GEOID to district assignments

        """
        assignments_df = pd.DataFrame()
        assignments_df["GEOID"] = demo_df["GEOID"]

        print("begin export solutions")

        for sol_idx in range(len(sol_dict)):
            solution_ixs = sol_dict[sol_idx]["solution_ixs"]
            col_title = "District" + str(sol_idx)
            assignments_df[col_title] = self.selected_districts(
                solution_ixs, cdm
            )

        # add a column parent which tells us this block's parent's center IF it is a center for the final tree, or -1 if it is a root for the final tree
        # assignments_df['Parent'] = np.nan
        # assignments_df['ID'] = np.nan
        # for node in sol_tree:
        #     if node.parent_id is not None:
        #         assignments_df.loc[node.center, 'ID']=node.id
        #         parent_center=internal_nodes[node.parent_id].center
        #         assignments_df.loc[node.center, 'Parent']=parent_center
        #         if parent_center is None:
        #             assignments_df.loc[node.center, 'Parent']=-1

        return assignments_df
        
    def master_solutions(self):
        # Would need to reimplement if I want to use again
        try:
            tree = load_tree(self.save_path)
            root = tree[-1]
        except:
            raise FileExistsError(
                "tree_time_str is invalid or given directory has no tree files"
            )
        try:
            cdms = load_cdms(self.save_path)
        except:
            cdms = {}
            for root_partition_ix in range(self.config.n_root_samples):
                cdms[root_partition_ix] = make_cdm(
                    tree[root_partition_ix][1], n_cgus=self.n_cgus
                )
        master_times = np.zeros((self.config.n_root_samples))
        print(len(root.children_ids))
        for root_partition_ix, root_partition in enumerate(root.children_ids):
            master_start_t = time.thread_time()
            col_title = "District" + str(root_partition_ix)
            solution_set = self.get_solution_set_dp(
                root,
                tree[root_partition_ix][0],
                tree[root_partition_ix][1],
                cdms[root_partition_ix],
                root_partition_ix,
                root_partition,
            )
            self.assignments_df[col_title] = (
                1  # self.get_solution_ser(solution_set, leaf_nodes)
            )
            if self.config.assignments_file_name is not None:
                self.assignments_df.to_csv(
                    os.path.join(
                        self.save_path,
                        "assignments",
                        self.config.assignments_file_name,
                    )
                    + ".csv",
                    index=False,
                )
            master_times[root_partition_ix] = (
                time.thread_time() - master_start_t
            )
        print(f"Master solutions times: {np.round(master_times, 2)}")
        print(f"Total master solutions time: {np.sum(master_times):0.2f}")

def get_save_path(self, time_str=str(int(time.time()))):
        """
        If time_str=None, creates directory save_path and returns it, and otherwise
        assembles it using time_str
        Args:
            time_str: (str or None) time string of desired save directory
        """
        return os.path.join(self.config.results_path, "results_%s" % time_str)

        
def master_solutions_nonlinear(self, args=None):
        """
        Solves the master selection problem optimizing for fairness on all root partitions.
        Args:
            leaf_nodes: (SHPNode list) with node capacity equal to 1 (has no child nodes).
            internal_nodes: (SHPNode list) with node capacity >1 (has child nodes).
            district_df: (pd.DataFrame) selected statistics of generated districts.
            state: (str) two letter state abbreviation
            state_vote_share: (float) the expected Republican vote-share of the state.
            lengths: (np.array) Pairwise block distance matrix.
            G: (nx.Graph) The block adjacency graph
        """
        if args is None:
            save_path = self.get_save_path(time_str=self.config.tree_time_str)
            try:
                tree = load_object(os.path.join(save_path, "tree.pkl"))
            except:
                raise FileExistsError("tree_time_str is invalid")
            demo_df, G, lengths, edge_dists = load_opt_data(
                self.config.state, self.config.year, self.config.granularity
            )
            n_census_shapes = len(demo_df["GEOID"])
            cdm = make_cdm(tree[1], n_cgus=n_census_shapes)
        else:
            tree = args["tree"]
            cdm = args["cdm"]
            demo_df = args["demo_df"]
            save_path = args["save_path"]
        internal_nodes = tree[0]
        leaf_nodes = tree[1]
        # district_df_of_tree_dir(save_path)
        # maj_min=majority_minority(cdm, demo_df)
        # print(maj_min)
        # print(sum(maj_min))
        cost_coeffs = majority_black(cdm, demo_df)
        maj_min = np.zeros((len(cdm.T)))
        bb = np.zeros((len(cost_coeffs)))

        root_map, ix_to_id = make_root_partition_to_leaf_map(
            leaf_nodes, internal_nodes
        )
        sol_dict = {}
        initial_t = time.thread_time()
        print("\n-------------Starting master problems-------------\n")
        for partition_ix, leaf_slice in root_map.items():
            """
            relax_start_t = time.thread_time()
            model_relaxed, dvars_relaxed = make_master(self.config.n_districts,
                                        cdm[:, leaf_slice],
                                        cost_coeffs[leaf_slice],
                                        maj_min[leaf_slice],
                                        bb[leaf_slice],
                                        callback_time_interval=self.config.callback_time_interval,
                                        relax=True)
            relax_construction_t = time.thread_time()
            model_relaxed.optimize()
            relax_solve_t = time.thread_time()
            print(f"\nRelaxed construction time: {relax_construction_t-relax_start_t}")
            print(f"Relaxed solve time: {relax_solve_t-relax_construction_t}")
            df = pd.DataFrame()
            xs = [v.X for v in dvars_relaxed.values()]
            df['X'] = xs
            print(f'Number of nonzero elements in xs = {sum([x != 0 for x in xs])}')
            #print([v.X for v in dvars_relaxed.values()])
            probabilities = [v.X * (1.0 >= v.X >= 0.5) + 1.0 * (v.X > 1.0) for v in dvars_relaxed.values()]
            df['probs'] = probabilities
            print(f'Number of nonzero elements in probabilities = {sum([x != 0 for x in probabilities])}')
            included_districts = np.random.binomial(n=1, p=probabilities)
            df['included'] = included_districts
            df.to_csv(os.path.join(save_path, 'test.csv'))
            print(f'Number of nonzero elements in included_districts = {sum([x != 0 for x in included_districts])}')
            #print(included_districts)
            """
            start_t = time.thread_time()
            model, dvars = make_master(
                self.config.n_districts,
                cdm[:, leaf_slice],
                cost_coeffs[leaf_slice],
                maj_min[leaf_slice],
                bb[leaf_slice],
                callback_time_interval=self.config.callback_time_interval,
            )
            construction_t = time.thread_time()

            # model.Params.LogToConsole = 0
            model.Params.MIPGapAbs = 1e-4
            model.Params.TimeLimit = len(leaf_slice) / 10
            print(f"MIPFocus: {model.Params.MIPFocus}")
            print(f"Threads: {model.Params.Threads}")
            if self.config.callback_time_interval is not None:
                model.optimize(callback=callback)
            else:
                model.optimize()
            status = model.status
            if status == GRB.INF_OR_UNBD:
                print("model is infeasible or unbounded")
                model.reset()
                if self.config.callback_time_interval is not None:
                    model.optimize(callback=callback)
                else:
                    model.optimize()
                status = model.status
            if status == GRB.INFEASIBLE:
                print("computing IIS")
                model.computeIIS()
                model.write("model.ilp")
                print("done with IIS")
            else:
                print(f"Model status = {status}")
            opt_cols = [j for j, v in dvars.items() if v.X > 0.5]
            solve_t = time.thread_time()

            sol_dict[partition_ix] = {
                "construction_time": construction_t - start_t,
                "solve_time": solve_t - construction_t,
                "n_leaves": len(leaf_slice),
                "solution_ixs": root_map[partition_ix][opt_cols],
                "optimal_objective": cost_coeffs[leaf_slice][opt_cols],
            }
            print(
                f"\nConstruction time: {sol_dict[partition_ix]['construction_time']}"
            )
            print(f"Solve time: {sol_dict[partition_ix]['solve_time']}")
            # print(opt_cols)
            # print(sol_dict[partition_ix]['solution_ixs'])
            # print(maj_min[sol_dict[partition_ix]['solution_ixs']])
            # constraint = model.getConstrByName("majorityMinority")
            # print(constraint.slack)

            sol_tree = get_solution_tree(
                leaf_nodes,
                internal_nodes,
                ix_to_id,
                sol_dict[partition_ix]["solution_ixs"],
            )
            # TODO fix sol_tree so that it gives us the right data structure and then write a function to print this to an output file, probably json

            if status == 2:
                print("Optimal solution found")
            elif status == 3:
                print(
                    "WARNING: no optimal solution is possible. Solving relaxation."
                )

            # constraintm_slacks=[]
            # for k in range(len(cost_coeffs)):
            #     constraintm=model.getConstrByName('testm_%s' % k)
            #     constraintm_slacks.append(constraintm.slack)
            #     if constraintm.slack!=0:
            #         print(str(k)+": "+str(int(constraintm.slack)))
            #         print(dvars[k])
            assignments_df = self.export_solutions(
                sol_dict, demo_df, cdm, sol_tree, internal_nodes
            )
            results_save_name = "assignments_%s.csv" % str(int(time.time()))
            assignments_df.to_csv(
                os.path.join(save_path, results_save_name), index=False
            )
        print(
            f"\nTotal time for master problems: {time.thread_time()-initial_t}"
        )

'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import Figure
import libpysal
import random
import numpy as np
from typing import Optional, get_args
import sys

sys.path.append(".")
import constants
from constants import (
    IPStr,
    VapCol,
    CguMapFunc,
    DistrictMapFunc,
    ComparativeDistrictMapFunc,
    flatten,
    SIX_COLORS,
    SMOOTH_SIX_COLORS,
)
from data.config import SHPConfig, MapConfig
from data.demo_df import DemoDataFrame
from data.shape_df import ShapeDataFrame
from data.partition import Partitions, Partition
from analyze.maj_min import cvap_props, cvap_props_cgus, is_maj_cvap
from optimize.tree import SHPTree


# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(
                plt.Rectangle(
                    [
                        width / len(orig_handle.colors) * i
                        - handlebox.xdescent,
                        -handlebox.ydescent,
                    ],
                    width / len(orig_handle.colors),
                    height,
                    facecolor=c,
                    edgecolor="k",
                    linewidth=0.2,
                )
            )

        patch = mpl.collections.PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch


"""
def add_layers_above(results_time_str, config, cgus, plans=None):
    shp = SHP(config)
    tree = load_tree(
        os.path.join(
            config.results_path, "results %s" % results_time_str, "tree"
        )
    )
    root = tree[-1]
    for plan in plans:
        (internal_nodes, leaf_nodes) = tree[plan]
        solution_set = shp.get_solution_set_dp(
            root, internal_nodes, leaf_nodes, plan, root.children_ids[plan]
        )
        node_layer_above = set()
        for leaf_id in solution_set:
            parent_node = internal_nodes[leaf_nodes[leaf_id].parent_id]
            if parent_node.n_districts in config.exact_partition_range:
                node_layer_above.add(parent_node)
            else:
                node_layer_above.add(leaf_nodes[leaf_id])
        assignment_ser = pd.Series(index=cgus.index)
        for district_ix, node in enumerate(node_layer_above):
            for leaf_id in node.best_subtree[0]:
                assignment_ser.loc[leaf_nodes[leaf_id].area] = district_ix
        cgus[f"LayerAbove{plan}"] = assignment_ser
    return cgus

def six_coloring(district_adj_mtx):
    def get_min_degree(adj_mtx):
        min_degree_v = 0
        min_degree = adj_mtx[0].sum()
        for i in range(1, len(adj_mtx)):
            curr_degree = adj_mtx[i].sum()
            if curr_degree < min_degree:
                min_degree_v = i
                min_degree = curr_degree
        return min_degree_v

    def remove_v(v, adj_mtx):
        upper = np.append(adj_mtx[:v, :v], adj_mtx[:v, v + 1 :], axis=1)
        lower = np.append(
            adj_mtx[v + 1 :, :v], adj_mtx[v + 1 :, v + 1 :], axis=1
        )
        return np.append(upper, lower, axis=0)

    n = len(district_adj_mtx)
    if n <= 6:
        return list(np.arange(n))
    coloring = list(np.arange(6))

    deletion_order = np.zeros((n - 6), dtype="int")
    adj_mtxs = [district_adj_mtx]
    for i in range(n - 7, -1, -1):
        min_degree_v = get_min_degree(adj_mtxs[0])
        deletion_order[i] = min_degree_v
        adj_mtxs = [remove_v(min_degree_v, adj_mtxs[0])] + adj_mtxs
    adj_mtxs = adj_mtxs[1:]

    for i in range(0, n - 6):
        v = deletion_order[i]
        coloring = coloring[:v] + [-1] + coloring[v:]
        available_colors = np.ones((6))
        for j in range(i + 6):
            if adj_mtxs[i][v][j] != 0:
                available_colors[coloring[j]] = 0
        available_colors_list = []
        for k in range(6):
            if available_colors[k] == 1:
                available_colors_list.append(k)
        random.seed(i)
        coloring[v] = random.choice(available_colors_list)

    return coloring


def cmap_colored(color_list, num_districts, district_adj_mtx):
    coloring = six_coloring(district_adj_mtx)
    return ListedColormap(
        [color_list[coloring[i]] for i in range(num_districts)]
    )


def cmap_colored_shaded(
    color_list, num_districts, district_adj_mtx, shading_factors
):
    coloring = six_coloring(district_adj_mtx)
    return ListedColormap(
        [
            color_list[coloring[i]] * np.array([1, 1, 1, shading_factors[i]])
            for i in range(num_districts)
        ]
    )


def cmap_gradient(cmap_name, num_proportions, proportions):
    return ListedColormap(
        [
            mpl.colormaps[cmap_name](proportions[i])
            for i in range(num_proportions)
        ]
    )

def cgus(ax, state_df, cgus, color_list, cmap_name):
    cgus.plot(ax=ax, color="tab:blue")
    cgus.boundary.plot(ax=ax, edgecolor="black", linewidth=0.05)


def cgus_bm_shaded(ax, state_df, cgus, color_list, cmap_name):
    vap = state_df["VAP"].to_numpy()
    black_proportions = state_df["BVAP"].to_numpy() / np.where(vap == 0, 1, vap)
    cmap = cmap_gradient(cmap_name, len(cgus["GEOID"]), black_proportions)
    cgus.plot(ax=ax, cmap=cmap)
    cgus.boundary.plot(ax=ax, edgecolor="black", linewidth=0.05)
    plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps[cmap_name]), ax=ax)

def cgus_pop_shaded(ax, state_df, cgus, color_list, cmap_name):
    max_pop = state_df["population"].max()
    pop_proportions = state_df["population"].to_numpy() / max_pop
    cmap = cmap_gradient(cmap_name, len(cgus["GEOID"]), pop_proportions)
    cgus.plot(ax=ax, cmap=cmap)
    cgus.boundary.plot(ax=ax, edgecolor="black", linewidth=0.05)
    plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps[cmap_name]), ax=ax)

def cgus_six_colored(
    plan, ax, num_districts, black_proportions, cgus, color_list
):
    cgu_neighbors = libpysal.weights.Queen.from_dataframe(cgus)
    cgu_adj_mtx, _ = cgu_neighbors.full()
    print("before cmap")
    # cmap = cmap_colored(color_list, len(cgu_adj_mtx), cgu_adj_mtx)
    print("after cmap")
    cgus.plot(ax=ax, linewidth=1.0)
    for cgu in cgus.index:
        cgus[cgus.index == cgu].boundary.plot(
            ax=ax, edgecolor="black", linewidth=0.2
        )
def cgus_highlight_disconnected(
    plan, ax, num_districts, black_proportions, cgus, color_list
):
    coloring = np.full((3471, 4), 1, dtype=float)
    geomtypes = cgus.geometry.geom_type.values
    multipolygon_cgus = [
        i for i in range(len(geomtypes)) if geomtypes[i] == "MultiPolygon"
    ]
    coloring[multipolygon_cgus] = np.array([227, 25, 25, 256]) / 256
    cmap = ListedColormap(coloring)
    cgus.plot(ax=ax, cmap=cmap)
    cgus.boundary.plot(ax=ax, edgecolor="black", linewidth=0.05)
    for point, label in zip(cgus.geometry.representative_point(), cgus.index):
        # circle = plt.Circle((point.x, point.y), 0.2, color='black')
        # ax.add_patch(circle)
        if label in multipolygon_cgus:
            ax.annotate(
                label,
                xy=(point.x, point.y),
                xytext=(0.5, 0.5),
                textcoords="offset points",
                fontsize=0.5,
            )

def districts_six_colored_bm_outlined(
    plan, ax, num_districts, black_proportions, cgus, color_list
):
    maj_black = black_proportions > 0.5
    district_shapes = cgus.dissolve(by=f"District{plan}")
    district_neighbors = libpysal.weights.Queen.from_dataframe(district_shapes)
    district_adj_mtx, _ = district_neighbors.full()
    cmap = cmap_colored(color_list, num_districts, district_adj_mtx)
    cgus.plot(ax=ax, column=f"District{plan}", cmap=cmap, linewidth=1.0)

    maj_black_districts = []
    i = 0
    for district in district_shapes.index:
        if not maj_black[int(district)]:
            i += 1
        else:
            maj_black_districts.append(district)
    for district in maj_black_districts:
        district_shapes[district_shapes.index == district].boundary.plot(
            ax=ax, edgecolor="black", linewidth=0.2
        )


def districts_six_colored_bp_shaded(
    plan, ax, num_districts, black_proportions, cgus, color_list, num_shades=3
):
    # print(cgus[cgus[f'District{plan}'] == 0]['GEOID'])
    district_shapes = cgus.dissolve(by=f"District{plan}")
    # print(district_shapes.loc[0])
    district_neighbors = libpysal.weights.Queen.from_dataframe(district_shapes)
    district_adj_mtx, _ = district_neighbors.full()

    def shading_factor(black_proportion):
        if 0 <= black_proportion < 0.5:
            return 1 / 4
        else:
            return 1

    shading_factors = [
        shading_factor(black_proportions[i]) for i in range(num_districts)
    ]
    # shading_factors = [1 / ((num_shades - math.floor(black_proportions[i] * 2 * (num_shades - 1))) * (black_proportions[i] < 0.5) + (black_proportions[i] >= 0.5)) for i in range(num_districts)]
    cmap = cmap_colored_shaded(
        color_list, num_districts, district_adj_mtx, shading_factors
    )
    cgus.plot(ax=ax, column=f"District{plan}", cmap=cmap, linewidth=5.0)
    cgus.boundary.plot(ax=ax, edgecolor="black", linewidth=0.05, linestyle="--")
    for district in district_shapes.index:
        district_shapes[district_shapes.index == district].boundary.plot(
            ax=ax, edgecolor="black", linewidth=0.1
        )

    # assignment_dict = assignment_ser_to_dict(cgus[f'District{plan}'])

    for point, district_id in zip(
        district_shapes.geometry.representative_point(), district_shapes.index
    ):
        # circle = plt.Circle((point.x, point.y), 0.2, color='black')
        # ax.add_patch(circle)
        label = district_id  # bvap_prop(assignment_dict[district_id], state_df)
        ax.annotate(
            label,
            xy=(point.x, point.y),
            xytext=(0.5, 0.5),
            textcoords="offset points",
            fontsize=0.5,
        )

    '''
    for point in district_shapes.geometry.representative_point():
        circle = plt.Circle((point.x, point.y), 0.2, color='black')
        ax.add_patch(circle)
    '''
    handles = [
        MulticolorPatch(color_list),
        MulticolorPatch([c * np.array([1, 1, 1, 1 / 4]) for c in color_list]),
    ]
    labels = ["Majority Black", "Not Majority Black"]
    ax.legend(
        handles=handles,
        labels=labels,
        handler_map={MulticolorPatch: MulticolorPatchHandler()},
        loc="upper right",
        labelspacing=0,
        fontsize=3,
        handlelength=15,
        handleheight=3,
        bbox_to_anchor=(0.95, 0.95),
    )


def districts_bp_grayscale(
    plan, ax, num_districts, black_proportions, cgus, color_list
):
    cmap = cmap_gradient("binary", num_districts, black_proportions)
    cgus.plot(ax=ax, column=f"District{plan}", cmap=cmap, linewidth=1.0)
    plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps["binary"]), ax=ax)

def draw_cgu_maps(config, color_list, cmap_name, cgu_map_func):
    cgus = load_cgus(config)
    state_df = load_state_df(config)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    pdf_path = os.path.join(config.results_path, f"{cgu_map_func.__name__}.pdf")
    cgu_map_func(ax, state_df, cgus, color_list, cmap_name)
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf", dpi=300)


def draw_individual_district_maps(
    shp_config: SHPConfig,
    results_time_str: str,
    assignments_file_name: str,
    color_list: list[list[int]],
    individual_district_map_func,
    plans: Optional[list[int]] = None,
    highlight_diff: bool = False,
    outline_layer_above: bool = False,
    maj_black_partition_ix: Optional[int] = None,
    root_partition_ix: Optional[int] = None,
):
    assignments_df = load_assignments_df(
        shp_config.results_path, results_time_str, assignments_file_name
    )
    cgus = load_cgus(shp_config)
    save_path = os.path.join(
        shp_config.results_path, "results_" + results_time_str
    )
    if not os.path.exists(os.path.join(save_path, "maps")):
        os.mkdir(os.path.join(save_path, "maps"))
    assignments_df["GEOID"] = (
        assignments_df["GEOID"].astype(str).apply(lambda x: x.zfill(11))
    )
    cgus = cgus.merge(assignments_df, on="GEOID", how="left")
    column_name = list(assignments_df.columns)[1]
    num_districts = int(max(assignments_df[column_name].to_list())) + 1
    num_plans = 0
    for column_name in assignments_df.columns.values:
        if column_name[:8] == "District":
            num_plans += 1

    state_df = load_state_df(shp_config)

    if plans is None:
        plans = np.arange(num_plans)

    if outline_layer_above:
        shp = SHP(shp_config)
        tree = load_tree(save_path)
        root = tree[-1]
        for plan in plans:
            if root_partition_ix is None:
                if maj_black_partition_ix is not None:
                    root_partition_ix = math.floor(plan / 2)
                else:
                    root_partition_ix = plan
            (internal_nodes, leaf_nodes) = tree[root_partition_ix]
            if maj_black_partition_ix is not None:
                leaf_nodes = shp.get_sample_trial_leaf_nodes(
                    maj_black_partition_ix, root, internal_nodes, leaf_nodes
                )
            solution_set = shp.get_solution_set_dp(
                root,
                internal_nodes,
                leaf_nodes,
                root_partition_ix,
                root.children_ids[root_partition_ix],
            )
            node_layer_above = set()
            for leaf_id in solution_set:
                parent_node = internal_nodes[leaf_nodes[leaf_id].parent_id]
                if parent_node.n_districts in shp_config.exact_partition_range:
                    node_layer_above.add(parent_node)
                else:
                    node_layer_above.add(leaf_nodes[leaf_id])
            assignment_ser = pd.Series(index=cgus.index)
            for district_ix, node in enumerate(node_layer_above):
                for leaf_id in node.best_subtree[0]:
                    assignment_ser.loc[leaf_nodes[leaf_id].area] = district_ix
            cgus[f"LayerAbove{plan}"] = assignment_ser

    prev_assignment_dict = None
    for plan in plans:
        fig, ax = plt.subplots()
        ax.set_axis_off()
        pdf_path = os.path.join(
            save_path,
            "maps",
            f"districting_{plan}_{assignments_file_name}_{individual_district_map_func.__name__}.pdf",
        )
        curr_assignment_dict = assignment_ser_to_dict(
            assignments_df[f"District{plan}"]
        )
        black_proportions = np.array(bvap_props(curr_assignment_dict, state_df))
        '''
        bvap_per_district = np.zeros(num_districts)
        vap_per_district = np.zeros(num_districts)
        for i in assignments_df.index.tolist():
            district = int(assignments_df.loc[i, f'District{plan}'])
            try:
                bvap_per_district[district] += int(state_df.loc[i, 'BVAP'])
                vap_per_district[district] += int(state_df.loc[i, 'VAP'])
            except IndexError:
                print(f'District indices for plan {plan} are incorrect')
                return
        black_proportions = bvap_per_district / vap_per_district
        '''
        print(black_proportions[5:7])
        if highlight_diff:
            diff = []
            if prev_assignment_dict is not None:
                for (
                    district_id,
                    district_region,
                ) in curr_assignment_dict.items():
                    if prev_assignment_dict[district_id] != district_region:
                        diff.append(district_id)
            prev_assignment_dict = curr_assignment_dict
            individual_district_map_func(
                plan, ax, num_districts, black_proportions, cgus, color_list
            )
            district_shapes = cgus.dissolve(by=f"District{plan}")
            for point in district_shapes.loc[
                diff
            ].geometry.representative_point():
                circle = plt.Circle((point.x, point.y), 0.2, color="black")
                ax.add_patch(circle)
            '''
            for point, district_id in zip(district_shapes.geometry.representative_point(), district_shapes.index):
                #circle = plt.Circle((point.x, point.y), 0.2, color='black')
                #ax.add_patch(circle)
                label = bvap_prop(curr_assignment_dict[district_id], state_df)
                ax.annotate(label, xy=(point.x, point.y), xytext=(0.5, 0.5), textcoords="offset points", fontsize=0.5)
            '''
        else:
            individual_district_map_func(
                plan, ax, num_districts, black_proportions, cgus, color_list
            )
        if outline_layer_above:
            print(cgus[f"LayerAbove{plan}"])
            layer_above_shapes = cgus.dissolve(by=f"LayerAbove{plan}")
            for region in layer_above_shapes.index:
                layer_above_shapes[
                    layer_above_shapes.index == region
                ].boundary.plot(ax=ax, edgecolor="black", linewidth=0.4)
        fig.savefig(pdf_path, bbox_inches="tight", format="pdf", dpi=300)

def draw_comparative_district_maps(
    config,
    results_time_str,
    assignments_file_name,
    color_list,
    comparative_district_map_func,
    plans=None,
    diff_districts=None,
    save_diff_districts_data=False,
):
    assignments_df = load_assignments_df(
        config.results_path, results_time_str, assignments_file_name
    )
    cgus = load_cgus(config)
    save_path = os.path.join(config.results_path, "results_" + results_time_str)
    if not os.path.exists(os.path.join(save_path, "maps")):
        os.mkdir(os.path.join(save_path, "maps"))
    assignments_df["GEOID"] = (
        assignments_df["GEOID"].astype(str).apply(lambda x: x.zfill(11))
    )
    cgus = cgus.merge(assignments_df, on="GEOID", how="left")
    num_districts = int(max(assignments_df["District0"].to_list())) + 1
    if plans is None:
        plans = []
        for column_name in assignments_df.columns.values:
            if column_name[:8] == "District":
                plans.append(column_name[8:])
    num_plans = len(plans)
    state_df = load_state_df(config)

    for i in range(1, num_plans):
        plan_before = plans[i - 1]
        plan_after = plans[i]
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].set_axis_off()
        axes[1].set_axis_off()
        pdf_path = os.path.join(
            save_path,
            "maps",
            f"districting_{plan_before}->{plan_after}_{assignments_file_name}_{comparative_district_map_func.__name__}.pdf",
        )

        assignment_dict_before = assignment_ser_to_dict(
            assignments_df[f"District{plan_before}"]
        )
        assignment_dict_after = assignment_ser_to_dict(
            assignments_df[f"District{plan_after}"]
        )
        if diff_districts is None:
            diff_districts = [
                district_id
                for district_id in assignment_dict_before
                if assignment_dict_before[district_id]
                != assignment_dict_after[district_id]
            ]
        print(diff_districts)
        if save_diff_districts_data:
            diff_districts_df = pd.DataFrame()
            diff_districts_df["BVAP before"] = [
                bvap(assignment_dict_before[district_id], state_df)
                for district_id in diff_districts
            ]
            diff_districts_df["VAP before"] = [
                vap(assignment_dict_before[district_id], state_df)
                for district_id in diff_districts
            ]
            diff_districts_df["BVAP/VAP before"] = (
                diff_districts_df["BVAP before"]
                / diff_districts_df["VAP before"]
            )
            diff_districts_df["BVAP after"] = [
                bvap(assignment_dict_after[district_id], state_df)
                for district_id in diff_districts
            ]
            diff_districts_df["VAP after"] = [
                vap(assignment_dict_after[district_id], state_df)
                for district_id in diff_districts
            ]
            diff_districts_df["BVAP/VAP after"] = (
                diff_districts_df["BVAP after"] / diff_districts_df["VAP after"]
            )
            print(diff_districts_df)
            # diff_districts_df.to_csv(os.path.join(save_path, 'maps', f'districting_{plan_before}->{plan_after}_{assignments_file_name}_{comparative_district_map_func.__name__}.csv'))

        black_proportions = np.array(
            [
                bvap_props(assignment_dict_before, state_df),
                bvap_props(assignment_dict_after, state_df),
            ]
        )
        print(black_proportions)
        comparative_district_map_func(
            plan_before,
            plan_after,
            diff_districts,
            fig,
            axes,
            num_districts,
            black_proportions,
            cgus,
            color_list,
            state_df,
        )
        fig.savefig(pdf_path, bbox_inches="tight", format="pdf", dpi=300)

def districts_six_colored_bp_shaded_comparative(
    plan_before,
    plan_after,
    diff_districts,
    fig,
    axes,
    num_districts,
    black_proportions,
    cgus,
    color_list,
    state_df,
):
    cgus_subregion = cgus[cgus[f"District{plan_before}"].isin(diff_districts)]
    diff_districts_2 = [
        int(district_id)
        for district_id in cgus_subregion[f"District{plan_after}"].unique()
    ]
    diff_districts_2.reverse()
    num_diff_districts = len(diff_districts)
    for district_ix in range(num_diff_districts):
        cgus_subregion.loc[
            cgus_subregion[f"District{plan_before}"]
            == diff_districts[district_ix],
            f"District{plan_before}",
        ] = district_ix
        cgus_subregion.loc[
            cgus_subregion[f"District{plan_after}"]
            == diff_districts_2[district_ix],
            f"District{plan_after}",
        ] = district_ix
    district_shapes_before = cgus_subregion.dissolve(
        by=f"District{plan_before}"
    )
    district_shapes_after = cgus_subregion.dissolve(by=f"District{plan_after}")
    district_neighbors_before = libpysal.weights.Queen.from_dataframe(
        district_shapes_before
    )
    district_neighbors_after = libpysal.weights.Queen.from_dataframe(
        district_shapes_after
    )
    district_adj_mtx_before, _ = district_neighbors_before.full()
    district_adj_mtx_after, _ = district_neighbors_after.full()

    def shading_factor(black_proportion):
        if 0 <= black_proportion < 0.5:
            return 1 / 4
        else:
            return 1

    shading_factors_before = [
        shading_factor(black_proportions[0][district_id])
        for district_id in diff_districts
    ]
    shading_factors_after = [
        shading_factor(black_proportions[1][district_id])
        for district_id in diff_districts_2
    ]
    cmap_before = cmap_colored_shaded(
        color_list,
        num_diff_districts,
        district_adj_mtx_before,
        shading_factors_before,
    )
    cmap_after = cmap_colored_shaded(
        color_list,
        num_diff_districts,
        district_adj_mtx_after,
        shading_factors_after,
    )
    cgus_subregion.plot(
        ax=axes[0],
        column=f"District{plan_before}",
        cmap=cmap_before,
        linewidth=5.0,
    )
    cgus_subregion.plot(
        ax=axes[1],
        column=f"District{plan_after}",
        cmap=cmap_after,
        linewidth=5.0,
    )
    cgus_subregion.boundary.plot(ax=axes[0], edgecolor="black", linewidth=0.05)
    cgus_subregion.boundary.plot(ax=axes[1], edgecolor="black", linewidth=0.05)
    for district in district_shapes_before.index:
        district_shapes_before[
            district_shapes_before.index == district
        ].boundary.plot(ax=axes[0], edgecolor="black", linewidth=0.2)
    for district in district_shapes_after.index:
        district_shapes_after[
            district_shapes_after.index == district
        ].boundary.plot(ax=axes[1], edgecolor="black", linewidth=0.2)
    handles = [
        MulticolorPatch(color_list),
        MulticolorPatch([c * np.array([1, 1, 1, 1 / 4]) for c in color_list]),
    ]
    labels = ["Majority Black", "Not Majority Black"]
    fig.legend(
        handles=handles,
        labels=labels,
        handler_map={MulticolorPatch: MulticolorPatchHandler()},
        loc="upper right",
        labelspacing=0,
        fontsize=3,
        handlelength=15,
        handleheight=3,
        bbox_to_anchor=(0.95, 0.95),
    )

def districts_bp_shaded_cgus_comparative(
    plan_before,
    plan_after,
    diff_districts,
    fig,
    axes,
    num_districts,
    black_proportions,
    cgus,
    color_list,
    state_df,
    cmap_name,
):
    state_df_subregion = state_df[
        cgus[f"District{plan_before}"].isin(diff_districts)
    ]
    black_proportions_subregion = (
        state_df_subregion["BVAP"].to_numpy()
        / state_df_subregion["VAP"].to_numpy()
    )
    cgus_subregion = cgus[cgus[f"District{plan_before}"].isin(diff_districts)]
    cmap = cmap_gradient(
        cmap_name, len(cgus_subregion["GEOID"]), black_proportions_subregion
    )
    cgus_subregion.plot(ax=axes[0], cmap=cmap)
    # cgus_subregion.boundary.plot(ax=axes[0], edgecolor='yellow', linewidth=0.05)
    cgus_subregion.plot(ax=axes[1], cmap=cmap)
    # cgus_subregion.boundary.plot(ax=axes[1], edgecolor='yellow', linewidth=0.05)
    district_shapes_before = cgus_subregion.dissolve(
        by=f"District{plan_before}"
    )
    district_shapes_after = cgus_subregion.dissolve(by=f"District{plan_after}")
    for district in district_shapes_before.index:
        if black_proportions[0][district] > 0.5:
            district_shapes_before[
                district_shapes_before.index == district
            ].boundary.plot(ax=axes[0], edgecolor="green", linewidth=0.2)
        else:
            district_shapes_before[
                district_shapes_before.index == district
            ].boundary.plot(ax=axes[0], edgecolor="red", linewidth=0.2)
    for district in district_shapes_after.index:
        if black_proportions[1][district] > 0.5:
            district_shapes_after[
                district_shapes_after.index == district
            ].boundary.plot(ax=axes[1], edgecolor="green", linewidth=0.2)
        else:
            district_shapes_after[
                district_shapes_after.index == district
            ].boundary.plot(ax=axes[1], edgecolor="red", linewidth=0.2)
    fig.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps["binary"]), ax=axes)

"""


class ColorMapGenerator:
    def __init__(self, colors: list[np.ndarray]):
        self.colors = colors

    @staticmethod
    def six_coloring(district_adj_mtx: np.ndarray) -> list[int]:
        def get_min_degree(adj_mtx):
            min_degree_v = 0
            min_degree = adj_mtx[0].sum()
            for i in range(1, len(adj_mtx)):
                curr_degree = adj_mtx[i].sum()
                if curr_degree < min_degree:
                    min_degree_v = i
                    min_degree = curr_degree
            return min_degree_v

        def remove_v(v, adj_mtx):
            upper = np.append(adj_mtx[:v, :v], adj_mtx[:v, v + 1 :], axis=1)
            lower = np.append(
                adj_mtx[v + 1 :, :v], adj_mtx[v + 1 :, v + 1 :], axis=1
            )
            return np.append(upper, lower, axis=0)

        n = len(district_adj_mtx)
        if n <= 6:
            return list(np.arange(n))
        coloring = list(np.arange(6))

        deletion_order = np.zeros((n - 6), dtype="int")
        adj_mtxs = [district_adj_mtx]
        for i in range(n - 7, -1, -1):
            min_degree_v = get_min_degree(adj_mtxs[0])
            deletion_order[i] = min_degree_v
            adj_mtxs = [remove_v(min_degree_v, adj_mtxs[0])] + adj_mtxs
        adj_mtxs = adj_mtxs[1:]

        for i in range(0, n - 6):
            v = deletion_order[i]
            coloring = coloring[:v] + [-1] + coloring[v:]
            available_colors = np.ones((6))
            for j in range(i + 6):
                if adj_mtxs[i][v][j] != 0:
                    available_colors[coloring[j]] = 0
            available_colors_list = []
            for k in range(6):
                if available_colors[k] == 1:
                    available_colors_list.append(k)
            random.seed(i)
            coloring[v] = random.choice(available_colors_list)

        return coloring

    def make_cmap_coloring(self, adj_mtx: np.ndarray) -> ListedColormap:
        coloring = self.six_coloring(adj_mtx)
        return ListedColormap(
            [self.colors[coloring[i]] for i in range(len(adj_mtx))]
        )

    def make_cmap_coloring_shaded(
        self, adj_mtx: np.ndarray, weights: list[float]
    ) -> ListedColormap:
        coloring = self.six_coloring(adj_mtx)
        return ListedColormap(
            [
                self.colors[coloring[i]] * np.array([1, 1, 1, weights[i]])
                for i in range(len(adj_mtx))
            ]
        )

    def make_cmap_gradient(
        self, cmap_name: str, weights: list[float]
    ) -> ListedColormap:
        return ListedColormap(
            [mpl.colormaps[cmap_name](weights[i]) for i in range(len(weights))]
        )

    def make_cmap_coloring_cgus_shaded(
        self, adj_mtx: np.ndarray, partition: Partition, weights: list[float]
    ):
        coloring = self.six_coloring(adj_mtx)
        colors = np.zeros((len(weights), 4), dtype=float)
        for i in range(len(adj_mtx)):
            for j in partition.get_part(i):
                colors[j] = self.colors[coloring[i]] * np.array(
                    [1, 1, 1, weights[j]]
                )
        return ListedColormap(colors)

    def make_cmap_transparent(self, length: int) -> ListedColormap:
        return ListedColormap(np.zeros((length, 4), dtype=float))


class Map:
    def __init__(
        self,
        fig: Figure,
        ax: Axes,
        shape_df: ShapeDataFrame,
        demo_df: DemoDataFrame,
        colors: list[np.ndarray],
    ):
        self.fig = fig
        self.ax = ax
        self.shape_df = shape_df
        self.demo_df = demo_df
        self.colors = colors  # TODO maybe delete?
        self.cmg = ColorMapGenerator(colors)

    # TODO: check if linewidth parameter is necessary for plot_shapes
    def plot_shapes(
        self,
        shape_df: Optional[ShapeDataFrame] = None,
        column=None,
        cmap=None,
        color=None,
        linewidth=1.0,
        hatch=None,
    ):
        if shape_df is None:
            shape_df = self.shape_df
        shape_df.plot(
            ax=self.ax,
            column=column,
            cmap=cmap,
            color=color,
            linewidth=linewidth,
            hatch=hatch,
        )

    def plot_boundaries(
        self,
        shape_df: Optional[ShapeDataFrame] = None,
        edgecolor: str = "black",
        linewidth: float = 0.05,
        linestyle: str = "-",
    ):
        if shape_df is None:
            shape_df = self.shape_df
        shape_df.boundary.plot(
            ax=self.ax,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
        )

    def plot_colorbar(self, cmap_name: str, axes=None):
        if axes is None:
            plt.colorbar(
                mpl.cm.ScalarMappable(cmap=mpl.colormaps[cmap_name]),
                ax=self.ax,
            )
        else:
            # TODO: fix the issue where the colorbar resizes the right map
            # cbar_ax = self.fig.add_axes("right", [0.1, 0.1, 0.05, 0.8])
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes("right", size="0%", pad=0.05)
            self.fig.colorbar(
                mpl.cm.ScalarMappable(cmap=mpl.colormaps[cmap_name]),
                ax=self.ax,
                cax=cax,
            )

    def get_adj_mtx(self, shape_df: ShapeDataFrame) -> np.ndarray:
        neighbors = libpysal.weights.Queen.from_dataframe(shape_df)
        return neighbors.full()[0]


class CguMap(Map):
    def __init__(
        self,
        fig: Figure,
        ax: Axes,
        shape_df: ShapeDataFrame,
        demo_df: DemoDataFrame,
        colors: list[np.ndarray],
    ):

        super().__init__(fig, ax, shape_df, demo_df, colors)

    # TODO: potentially make this take in a cmap as well or smth? Or leave it as is
    def solid_colored(self, color: str = "tab:blue"):
        self.plot_shapes(color=color)
        self.plot_boundaries()

    # TODO: seems to be broken
    def six_colored(self):
        adj_mtx = self.get_adj_mtx(self.shape_df)
        cmap = self.cmg.make_cmap_coloring(adj_mtx)
        self.plot_shapes(cmap=cmap, linewidth=1.0)
        self.plot_boundaries()

    def shaded(self, weights: list[float], cmap_name: Optional[str]):
        if cmap_name is None:
            raise ValueError("User did not provide cmap name in map config")
        cmap = self.cmg.make_cmap_gradient(cmap_name, weights)
        self.plot_shapes(cmap=cmap)
        self.plot_boundaries()
        self.plot_colorbar(cmap_name)

    def show_disconnected(self):
        coloring = np.full((3471, 4), 1, dtype=float)
        geomtypes = self.shape_df.geometry.geom_type.values
        multipolygon_cgus = [
            i for i in range(len(geomtypes)) if geomtypes[i] == "MultiPolygon"
        ]
        coloring[multipolygon_cgus] = np.array([227, 25, 25, 256]) / 256
        cmap = ListedColormap(coloring)
        self.plot_shapes(cmap=cmap)
        self.plot_boundaries()
        for point, label in zip(
            self.shape_df.geometry.representative_point(),
            self.shape_df.index,
        ):
            # circle = plt.Circle((point.x, point.y), 0.2, color='black')
            # ax.add_patch(circle)
            if label in multipolygon_cgus:
                self.ax.annotate(
                    label,
                    xy=(point.x, point.y),
                    xytext=(0.5, 0.5),
                    textcoords="offset points",
                    fontsize=0.5,
                )


class DistrictMap(Map):
    def __init__(
        self,
        fig: Figure,
        ax: Axes,
        shape_df: ShapeDataFrame,
        demo_df: DemoDataFrame,
        colors: list[np.ndarray],
        partition: Partition,
    ):
        super().__init__(fig, ax, shape_df, demo_df, colors)
        self.partition = partition
        self.district_shape_df = self.shape_df.get_district_shape_df(partition)

    def mark_diff(self, diff_district_ids: list[int]):
        for point in self.district_shape_df.loc[
            diff_district_ids
        ].geometry.representative_point():
            circle = plt.Circle((point.x, point.y), 0.2, color="black")
            self.ax.add_patch(circle)
        """
        for point, district_id in zip(district_shapes.geometry.representative_point(), district_shapes.index):
            #circle = plt.Circle((point.x, point.y), 0.2, color='black')
            #ax.add_patch(circle)
            label = bvap_prop(curr_assignment_dict[district_id], state_df)
            ax.annotate(label, xy=(point.x, point.y), xytext=(0.5, 0.5), textcoords="offset points", fontsize=0.5)
        """

    """
    def plot_district_boundaries(
        self, district_shape_df: ShapeDataFrame, linewidth=0.4
    ):
        district_shape_df.boundary.plot(
            ax=self.ax, edgecolor="black", linewidth=linewidth
        )
    """

    def plot_colored_maj_min_legend(self, col: VapCol, on_fig: bool = False):
        handles = [
            MulticolorPatch(self.colors),
            MulticolorPatch(
                [c * np.array([1, 1, 1, 0.25]) for c in self.colors]
            ),
        ]
        labels = [f"Majority {col}", f"Not Majority {col}"]
        obj = self.ax
        if on_fig:
            obj = self.fig
        obj.legend(
            handles=handles,
            labels=labels,
            handler_map={MulticolorPatch: MulticolorPatchHandler()},
            loc="upper right",
            labelspacing=0,
            fontsize=3,
            handlelength=15,
            handleheight=3,
            bbox_to_anchor=(0.95, 0.95),
        )

    def districts_colored_outlined(self, outlined_district_ids: list[int]):
        district_adj_mtx = self.get_adj_mtx(self.district_shape_df)
        cmap = self.cmg.make_cmap_coloring(district_adj_mtx)
        assignment = self.partition.get_assignment()
        self.plot_shapes(column=assignment, cmap=cmap, linewidth=1.0)
        self.plot_boundaries(
            shape_df=self.district_shape_df.get_subregion_df(
                outlined_district_ids
            ),
            linewidth=0.2,
        )

    def districts_colored_shaded(
        self,
        weights: list[float],
        col: VapCol,
        legend=True,
    ):
        district_adj_mtx = self.get_adj_mtx(self.district_shape_df)
        cmap = self.cmg.make_cmap_coloring_shaded(district_adj_mtx, weights)
        assignment = self.partition.get_assignment()
        self.plot_shapes(column=assignment, cmap=cmap, linewidth=5.0)
        # self.plot_boundaries(linestyle="--")
        # self.plot_boundaries(
        #    shape_df=self.district_shape_df, linewidth=0.05, linestyle="--"
        # )

        """
        for point, district_id in zip(
            self.district_shape_df.geometry.representative_point(),
            self.district_shape_df.index,
        ):
            # circle = plt.Circle((point.x, point.y), 0.2, color='black')
            # ax.add_patch(circle)
            label = (
                district_id  # bvap_prop(assignment_dict[district_id], state_df)
            )
            self.ax.annotate(
                label,
                xy=(point.x, point.y),
                xytext=(0.5, 0.5),
                textcoords="offset points",
                fontsize=0.5,
            )
        """

        """
        for point in district_shapes.geometry.representative_point():
            circle = plt.Circle((point.x, point.y), 0.2, color='black')
            ax.add_patch(circle)
        """
        if legend:
            self.plot_colored_maj_min_legend(col, on_fig=True)

    def districts_grayscale_shaded(
        self,
        weights: list[float],
    ):
        cmap = self.cmg.make_cmap_gradient("binary", weights)
        assignment = self.partition.get_assignment()
        self.plot_shapes(column=assignment, cmap=cmap)
        self.plot_colorbar("binary")

    def districts_outlined_cgus_grayscale_shaded(
        self, district_weights: list[float], cgu_weights: list[float]
    ):
        cmap = self.cmg.make_cmap_gradient("binary", cgu_weights)
        self.plot_shapes(cmap=cmap)
        n_districts = len(district_weights)
        maj_district_ids = [
            id for id in range(n_districts) if district_weights[id] > 0.5
        ]
        min_district_ids = [
            id for id in range(n_districts) if district_weights[id] <= 0.5
        ]
        self.plot_boundaries(
            self.district_shape_df.get_subregion_df(min_district_ids),
            edgecolor="red",
            linewidth=0.2,
        )
        self.plot_boundaries(
            self.district_shape_df.get_subregion_df(maj_district_ids),
            edgecolor="green",
            linewidth=0.2,
        )

    def districts_colored_hashed_cgus_shaded(
        self,
        district_weights: list[float],
        cgu_weights: list[float],
        col: VapCol,
    ):
        district_adj_mtx = self.get_adj_mtx(self.district_shape_df)
        cmap = self.cmg.make_cmap_coloring_cgus_shaded(
            district_adj_mtx, self.partition, cgu_weights
        )
        self.plot_shapes(cmap=cmap)
        self.plot_boundaries(linestyle="--")
        self.plot_boundaries(shape_df=self.district_shape_df, linewidth=0.1)
        maj_district_ids = [
            id
            for id in range(self.partition.n_districts)
            if district_weights[id] > 0.5
        ]
        subregion_df = self.district_shape_df.get_subregion_df(maj_district_ids)
        subregion_df.plot(
            ax=self.ax,
            cmap=cmap,
            linewidth=0.5,
            hatch="////////",
            facecolor=None,
            edgecolor="black",
        )
        # self.plot_shapes(
        #     shape_df=self.district_shape_df.get_subregion_df(maj_district_ids),
        #     hatch="//",
        #     linewidth=0.3
        # )
        self.plot_colored_maj_min_legend(col, on_fig=True)

    def districts_colored_hashed(
        self,
        district_weights: list[float],
        col: VapCol,
    ):
        district_adj_mtx = self.get_adj_mtx(self.district_shape_df)
        cmap = self.cmg.make_cmap_coloring(district_adj_mtx)
        # cmap = self.cmg.make_cmap_coloring_shaded(district_adj_mtx, district_weights)
        assignment = self.partition.get_assignment()
        self.plot_shapes(column=assignment, cmap=cmap, linewidth=5.0)
        # self.plot_boundaries(linestyle="--")
        self.plot_boundaries(shape_df=self.district_shape_df, linewidth=0.1)
        maj_district_ids = [
            id
            for id in range(self.partition.n_districts)
            if district_weights[id] > 0.5
        ]
        subregion_df = self.district_shape_df.get_subregion_df(maj_district_ids)
        subregion_df.plot(
            ax=self.ax,
            cmap=self.cmg.make_cmap_transparent(len(subregion_df)),
            linewidth=0.5,
            hatch="////////",
            facecolor=None,
            edgecolor="black",
        )


class MapGenerator:
    def __init__(
        self,
        shp_config: SHPConfig,
        map_config: MapConfig,
        partitions: Optional[Partitions] = None,
    ):
        self.save_path = shp_config.get_save_path()
        self.partitions = partitions
        if partitions is None and map_config.partitions_file_name is not None:
            self.partitions = Partitions.from_csv(
                os.path.join(
                    self.save_path,
                    "partitions",
                    map_config.partitions_file_name,
                )
            )
        self.shp_config = shp_config
        self.map_config = map_config
        self.shape_df = ShapeDataFrame.from_config(shp_config)
        self.demo_df = DemoDataFrame.from_config(shp_config)
        # self.cmg = ColorMapGenerator(self, map_config.colors)

    def make_empty_plot(self) -> tuple[Figure, Axes]:
        fig, ax = plt.subplots()
        ax.set_axis_off()
        return fig, ax

    def make_empty_comparative_plot(self) -> tuple[Figure, tuple[Axes]]:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].set_axis_off()
        axes[1].set_axis_off()
        return fig, axes

    @staticmethod
    def proceed_ok(pdf_path: str):
        if os.path.exists(pdf_path):
            while True:
                response = input(
                    f"The map file {pdf_path} already exists. Do you want"
                    " to replace it? (y/n)\n"
                )
                if response == "y":
                    return True
                elif response == "n":
                    return False
        else:
            return True

    def save_fig(self, fig: Figure, pdf_path: str):
        fig.savefig(
            pdf_path, bbox_inches="tight", pad_inches=0, format="pdf", dpi=300
        )

    def get_diff_district_ids(
        self, prev_partition: Optional[Partition], curr_partition: Partition
    ) -> list[int]:
        if prev_partition is None:
            return []
        return [
            district_id
            for (
                district_id,
                district_subregion,
            ) in curr_partition.get_parts().items()
            if sorted(prev_partition.get_part(district_id))
            != sorted(district_subregion)
        ]
        """
        for (
            district_id,
            district_region,
        ) in curr_partition.get_parts().items():
            if prev_assignment_dict[district_id] != district_region:
                diff.append(district_id)
        """

    """
    def get_district_shape_df(self, partition: Partition) -> ShapeDataFrame:
        shape_df_copy = self.shape_df.copy()
        shape_df_copy["Plan"] = partition.get_assignment()
        return shape_df_copy.dissolve(by="Plan")
    """

    def get_layer_above_shape_df(
        self, plan_id: int, ip_str: IPStr, tree: SHPTree
    ):
        root_partition_id = plan_id // len(tree.config.final_partition_ips)
        layer_above_nodes = tree.get_layer_above_solution_nodes(
            ip_str, root_partition_id
        )
        layer_above_partition = tree.get_partition(layer_above_nodes)
        return self.shape_df.get_district_shape_df(layer_above_partition)

    def get_cvap_prop_shading_weights(self, partition: Partition):
        props = cvap_props(self.shp_config.col, partition, self.demo_df)

        def shading_weight(cvap_prop):
            return 0.25 if 0 <= cvap_prop < 0.5 else 1

        return [shading_weight(cvap_prop) for cvap_prop in props]

    def draw_cgu_map(self, cgu_map_func: CguMapFunc):
        # TODO: possibly allow for the pdf path to include the cmap_name in it, if it's not None?
        pdf_path = os.path.join(
            constants.RESULTS_PATH,
            self.shp_config.get_dirname(),
            f"{cgu_map_func}.pdf",
        )
        if not self.proceed_ok(pdf_path):
            return
        fig, ax = self.make_empty_plot()
        cgu_map = CguMap(
            fig, ax, self.shape_df, self.demo_df, self.map_config.colors
        )
        if cgu_map_func == "solid_colored":
            cgu_map.solid_colored()
        elif cgu_map_func == "cvap_prop_shaded":
            weights = cvap_props_cgus(self.shp_config.col, self.demo_df)
            cgu_map.shaded(weights, self.map_config.cmap_name)
        elif cgu_map_func == "pop_prop_shaded":
            max_pop = self.demo_df["POP"].max()
            weights = self.demo_df["POP"].to_numpy() / max_pop
            cgu_map.shaded(weights, self.map_config.cmap_name)
        elif cgu_map_func == "six_colored":
            cgu_map.six_colored()
        elif cgu_map_func == "show_disconnected":
            cgu_map.show_disconnected()
        else:
            raise ValueError("Invalid Cgu Map Function Name")
        self.save_fig(fig, pdf_path)

    def draw_district_maps(
        self,
        district_map_func: DistrictMapFunc,
        plan_ids: Optional[list[int]] = None,
        highlight_diff: bool = False,
        outline_layer_above: bool = False,
        ip_str: Optional[IPStr] = None,
    ):
        if self.partitions is None:
            raise ValueError(
                "User did not provide partitions or partitions file name"
            )
        map_path = os.path.join(self.save_path, "maps")
        if not os.path.exists(map_path):
            os.mkdir(map_path)
        """
        assignments_df["GEOID"] = (
            assignments_df["GEOID"].astype(str).apply(lambda x: x.zfill(11))
        )
        """
        tree = SHPTree.from_file(os.path.join(self.save_path, "tree.pickle"))
        if plan_ids is None:
            plan_ids = self.partitions.get_plan_ids()
        prev_partition = None
        if ip_str is None:
            ip_str = self.shp_config.final_partition_ips[0]
        for plan_id in plan_ids:
            pdf_path = os.path.join(
                map_path,
                "_".join(
                    (
                        self.map_config.partitions_file_name[:-4],
                        f"p{plan_id}",
                        f"{district_map_func}.pdf",
                    )
                ),
            )
            if not self.proceed_ok(pdf_path):
                continue
            curr_partition = self.partitions.get_plan(plan_id)
            fig, ax = self.make_empty_plot()
            district_map = DistrictMap(
                fig,
                ax,
                self.shape_df,
                self.demo_df,
                self.map_config.colors,
                curr_partition,
            )
            if district_map_func == "colored_maj_cvap_outlined":
                col = self.shp_config.col
                maj_cvap_district_ids = [
                    district_id
                    for district_id, district_subregion in curr_partition.get_parts().items()
                    if is_maj_cvap(col, district_subregion, self.demo_df)
                ]
                district_map.districts_colored_outlined(maj_cvap_district_ids)
            elif district_map_func == "colored_cvap_prop_shaded":
                weights = self.get_cvap_prop_shading_weights(curr_partition)
                district_map.districts_colored_shaded(
                    weights, self.shp_config.col, legend=False
                )
            elif district_map_func == "cvap_prop_grayscale":
                weights = cvap_props(
                    self.shp_config.col, curr_partition, self.demo_df
                )
                district_map.districts_grayscale_shaded(weights)
            elif district_map_func == "districts_colored_hashed_cgus_shaded":
                cgu_weights = cvap_props_cgus(self.shp_config.col, self.demo_df)
                cgu_weights = [(weight + 0.3) / 2 for weight in cgu_weights]
                district_weights = cvap_props(
                    self.shp_config.col, curr_partition, self.demo_df
                )
                district_map.districts_colored_hashed_cgus_shaded(
                    district_weights, cgu_weights, self.shp_config.col
                )
            elif district_map_func == "districts_colored_hashed":
                district_weights = cvap_props(
                    self.shp_config.col, curr_partition, self.demo_df
                )
                district_map.districts_colored_hashed(
                    district_weights, self.shp_config.col
                )
            else:
                raise ValueError("Invalid District Map Function Name")

            if highlight_diff:
                diff_district_ids = self.get_diff_district_ids(
                    prev_partition, curr_partition
                )
                district_map.mark_diff(diff_district_ids)
                prev_partition = curr_partition

            if outline_layer_above:
                layer_above_shape_df = self.get_layer_above_shape_df(
                    plan_id, ip_str, tree
                )
                district_map.plot_boundaries(
                    shape_df=layer_above_shape_df, linewidth=1.0  # 0.6
                )
            self.save_fig(fig, pdf_path)

    def draw_comparative_district_maps(
        self,
        comparative_district_map_func: ComparativeDistrictMapFunc,
        plan_ids: Optional[list[int]] = None,
        print_diff_districts_data: bool = False,
    ):
        if self.partitions is None:
            raise ValueError(
                "User did not provide partitions or partitions file name"
            )
        map_path = os.path.join(self.save_path, "maps")
        if not os.path.exists(map_path):
            os.mkdir(map_path)
        """
        assignments_df["GEOID"] = (
            assignments_df["GEOID"].astype(str).apply(lambda x: x.zfill(11))
        )
        """
        if plan_ids is None:
            plan_ids = self.partitions.get_plan_ids()
        n_plans = len(plan_ids)
        for i in range(1, n_plans):
            plan_id_i = plan_ids[i - 1]
            plan_id_f = plan_ids[i]
            pdf_path = os.path.join(
                map_path,
                "_".join(
                    (
                        self.map_config.partitions_file_name[:-4],
                        f"p{plan_id_i}-{plan_id_f}",
                        f"{comparative_district_map_func}.pdf",
                    )
                ),
            )
            if not self.proceed_ok(pdf_path):
                continue
            fig, axes = self.make_empty_comparative_plot()
            partition_i = self.partitions.get_plan(plan_id_i)
            partition_f = self.partitions.get_plan(plan_id_f)
            diff_district_ids = self.get_diff_district_ids(
                partition_i, partition_f
            )
            print(len(diff_district_ids))
            if len(diff_district_ids) == 0:
                print(f"Plans {plan_id_i} and {plan_id_f} are the same")
                continue
            partitions = [
                partition_i.get_subpartition(diff_district_ids),
                partition_f.get_subpartition(diff_district_ids),
            ]
            subregion = partitions[0].get_region()

            # subregion = flatten(
            #     [
            #         partition_i.get_part(district_id)
            #         for district_id in diff_district_ids
            #     ]
            # )
            # diff_district_ids_f = (
            #    partition_f.get_assignment().loc[subregion].unique()
            # )

            district_maps = [
                DistrictMap(
                    fig,
                    axes[j],
                    self.shape_df.get_subregion_df(subregion),
                    self.demo_df.get_subregion_df(subregion),
                    self.map_config.colors,
                    partitions[j],
                )
                for j in range(2)
            ]
            """
            if print_diff_districts_data:
                # TODO update
                diff_districts_df = pd.DataFrame()
                diff_districts_df["BVAP before"] = [
                    bvap(assignment_dict_before[district_id], state_df)
                    for district_id in diff_districts
                ]
                diff_districts_df["VAP before"] = [
                    vap(assignment_dict_before[district_id], state_df)
                    for district_id in diff_districts
                ]
                diff_districts_df["BVAP/VAP before"] = (
                    diff_districts_df["BVAP before"]
                    / diff_districts_df["VAP before"]
                )
                diff_districts_df["BVAP after"] = [
                    bvap(assignment_dict_after[district_id], state_df)
                    for district_id in diff_districts
                ]
                diff_districts_df["VAP after"] = [
                    vap(assignment_dict_after[district_id], state_df)
                    for district_id in diff_districts
                ]
                diff_districts_df["BVAP/VAP after"] = (
                    diff_districts_df["BVAP after"]
                    / diff_districts_df["VAP after"]
                )
                print(diff_districts_df)
            """
            if comparative_district_map_func == "colored_cvap_prop_shaded":
                for j in range(2):
                    weights = self.get_cvap_prop_shading_weights(partitions[j])
                    district_maps[j].districts_colored_shaded(
                        weights, self.shp_config.col, legend=False
                    )
            elif comparative_district_map_func == "districts_colored_hashed":
                for j in range(2):
                    district_weights = cvap_props(
                        self.shp_config.col, partitions[j], self.demo_df
                    )
                    district_maps[j].districts_colored_hashed(
                        district_weights, self.shp_config.col
                    )
            elif (
                comparative_district_map_func
                == "maj_cvap_outlined_cvap_prop_shaded_cgus"
            ):
                cgu_weights = cvap_props_cgus(self.shp_config.col, self.demo_df)
                for j in range(2):
                    district_weights = cvap_props(
                        self.shp_config.col, partitions[j], self.demo_df
                    )
                    district_maps[j].districts_outlined_cgus_grayscale_shaded(
                        district_weights, cgu_weights
                    )
                district_maps[1].plot_colorbar("binary", axes=axes)
            else:
                raise ValueError(
                    "Invalid Comparative District Map Function Name"
                )
            self.save_fig(fig, pdf_path)


if __name__ == "__main__":
    color_map_names = {
        "binary"  # grayscale
        "Blues"  # bluescale
    }

    from experiments.LA_house import shp_config

    shp_config.col = "POCVAP"
    shp_config.save_dirname = "state_house_POCVAP"

    plan_id = 0
    # partitions_shp = Partitions.from_csv(
    #     os.path.join(shp_config.get_save_path(), "partitions", "shp.csv")
    # )
    # partitions_old = Partitions.from_csv(
    #     os.path.join(
    #         shp_config.get_save_path(),
    #         "partitions",
    #         f"shp_p{plan_id}_priority_4_opt_no_br.csv",
    #     )
    # )
    # partitions_new = Partitions()
    # partitions_new.set_plan(0, partitions_shp.get_plan(plan_id))
    # for id in partitions_old.get_plan_ids():
    #     partitions_new.set_plan(id + 1, partitions_old.get_plan(id))
    # partitions_new.to_csv(
    #     os.path.join(
    #         shp_config.get_save_path(),
    #         "partitions",
    #         f"shp_p{plan_id}_priority_4_opt_no_br_with_init.csv",
    #     ),
    #     shp_config,
    # )

    map_config = MapConfig(
        "LA",
        2020,
        "block_group",
        None,
        partitions_file_name=f"shp_p{plan_id}_priority_4_opt_no_br_with_init.csv",  # "shp_br.csv",
        # colors=SMOOTH_SIX_COLORS,
        cmap_name="Purples",
    )

    map_generator = MapGenerator(shp_config, map_config)

    # map_generator.draw_cgu_map("six_colored")

    # map_generator.draw_district_maps(
    #     "districts_colored_hashed",  # "colored_cvap_prop_shaded",
    #     plan_ids=[0],
    #     highlight_diff=False,
    #     outline_layer_above=True,
    # )

    map_generator.draw_comparative_district_maps(
        "districts_colored_hashed"  # "colored_cvap_prop_shaded",  # plan_ids=[0, 1]
    )

    # TODO: add a boolean in the config for adding numbers or representative points to districts or cgus

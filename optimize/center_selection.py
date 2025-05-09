import math
import random
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from spatial_utils import *
import sys

sys.path.append(".")
from data.demo_df import DemoDataFrame
from data.shape_df import ShapeDataFrame
from data.config import SHPConfig


# TODO: finish updating this file
def uniform_random(region_df, n_centers):
    """
    Uniform-random center selection function.
    Args:
        region_df: (pd.DataFrame) state_df subset of the node region
        n_centers: (int) split size

    Returns: (np.array) of size [n_centers] with block indices in [region_df]

    """
    return np.random.choice(
        np.array(region_df.index), size=n_centers, replace=False
    )


def iterative_random(region_df, n_centers, pdists):
    blocks = list(region_df.index)
    centers = [np.random.choice(blocks)]
    while len(centers) < n_centers:
        weights = np.prod(pdists[np.ix_(centers, region_df.index)], axis=0)
        centers.append(np.random.choice(blocks, p=weights / weights.sum()))
    return centers


def capacitated_iterative_random(region_df, capacities, pdists):
    """
    Iterative-random center selection method.
    Args:
        region_df: (pd.DataFrame) state_df subset of the node region
        capacities: (int list) of center capacities
        pdists: (np.array) pairwise distance matrix of region centroids

    Returns: (list) of block indices of the sampled centers

    """
    unassigned_blocks = list(region_df.index)
    np.random.shuffle(capacities)

    centers = []
    child_ix = 0
    block_seed = random.choice(unassigned_blocks)
    while child_ix < len(capacities):
        block_seed_sq_dist = pdists[block_seed, unassigned_blocks] ** 2
        center_p = block_seed_sq_dist / np.sum(block_seed_sq_dist)
        center_seed = np.random.choice(unassigned_blocks, p=center_p)

        if child_ix < len(capacities) - 1:
            assignment_order = np.argsort(
                pdists[center_seed, unassigned_blocks]
            )

            blocks_assigned_to_center = []
            population_assigned_to_center = 0
            target_population = capacities[child_ix]
            assignment_ix = 0
            while population_assigned_to_center < target_population:
                block = unassigned_blocks[assignment_order[assignment_ix]]
                blocks_assigned_to_center.append(block)
                population_assigned_to_center += region_df.loc[block][
                    "population"
                ]
                assignment_ix += 1

            for block in blocks_assigned_to_center:
                unassigned_blocks.remove(block)

        centers.append(center_seed)
        block_seed = center_seed
        child_ix += 1
    return centers


def kmeans_seeds(
    region_df, split_size, n_random_seeds=0, perturbation_scale=None
):
    """
    K-means based center selection methods.

    Implements fixed-center and/or pareto pertubation.

    Args:
        region_df: (pd.DataFrame) state_df subset of the node region
        split_size: (int) number of centers to sample
        n_random_seeds: (int) number of fixed centers
        perturbation_scale: Pareto pertubation scale parameter

    Returns: (list) of block indices of the sampled centers

    """

    weights = region_df.values + 1
    if perturbation_scale:
        weights = weight_perturbation(weights, perturbation_scale)
    if n_random_seeds:
        weights = rand_seed_reweight(weights, n_random_seeds)

    pts = region_df[["x", "y"]].values

    kmeans = (
        KMeans(n_clusters=split_size)
        .fit(pts, sample_weight=weights)
        .cluster_centers_
    )

    dists = cdist(kmeans, pts)
    centers = [
        region_df.index[i].item()  # Convert to native int for jsonability
        for i in list(np.argmin(dists, axis=1))
    ]

    return centers


def rand_seed_reweight(weights, n_seeds):
    """
    Utility function for assigning weights for fixed-center selection method.
    Args:
        weights: (np.array) of region block weights for k-means
        n_seeds: (float) number of fixed random seeds (fractional allowed)

    Returns: (np.array) modified weight vector

    """
    n_seeds = int(n_seeds // 1 + (random.random() < n_seeds % 1))
    total_weight = weights.sum()
    for _ in range(n_seeds):
        rand_seed = random.randint(0, len(weights) - 1)
        weights[rand_seed] = total_weight
    return weights


def weight_perturbation(weights, scale):
    """Pareto perturbation"""
    return weights * np.random.pareto(scale, len(weights))


def get_capacities(
    centers: list[int],
    child_sizes: list[int],
    subregion_df: DemoDataFrame,
    shape_subregion_df: ShapeDataFrame,
    config: SHPConfig,
) -> dict[int, int]:
    """
    Implements capacity assigment methods (both computing and matching)

    Args:
        centers: (list[int]) of cgu ids of the centers
        child_sizes: (list) of integers of the child node capacities
        region_df: (pd.DataFrame) state_df subset of the node region
        config: (dict) ColumnGenerator configuration

    Returns: (dict) {block index of center: capacity}

    """
    n_children = len(child_sizes)
    total_seats = int(sum(child_sizes))
    centroids = pd.DataFrame(
        data={
            "x": shape_subregion_df.centroid.x,
            "y": shape_subregion_df.centroid.y,
        }
    )
    center_locs = centroids.loc[centers][["x", "y"]].values
    locs = centroids[["x", "y"]].values
    pop = subregion_df["POP"].values

    dist_mat = cdist(locs, center_locs)
    if config.capacity_weights == "fractional":
        dist_mat **= -2
        weights = dist_mat / np.sum(dist_mat, axis=1)[:, None]
    elif config.capacity_weights == "voronoi":
        assignment = np.argmin(dist_mat, axis=1)
        weights = np.zeros((len(locs), len(centers)))
        weights[np.arange(len(assignment)), assignment] = 1

    center_assignment_score = np.sum(weights * pop[:, None], axis=0)
    center_assignment_score /= center_assignment_score.sum()
    center_fractional_caps = center_assignment_score * total_seats

    if config.capacities == "compute":
        cap_constraint = config.get("capacity_constraint", None)
        if cap_constraint:
            lb = max(1, math.floor(total_seats / (n_children * cap_constraint)))
            ub = min(
                total_seats,
                math.ceil((total_seats * cap_constraint) / n_children),
            )
        else:
            lb = 1
            ub = total_seats
        center_caps = np.ones(n_children).astype(int) * lb
        while center_caps.sum() != total_seats:
            disparity = center_fractional_caps - center_caps
            at_capacity = center_caps >= ub
            disparity[at_capacity] = -total_seats  # enforce upperbound
            center_caps[np.argmax(disparity)] += 1

        return {
            center: capacity for center, capacity in zip(centers, center_caps)
        }

    elif config.capacities == "match":
        center_order = center_assignment_score.argsort()
        capacities_order = child_sizes.argsort()

        return {
            centers[cen_ix]: child_sizes[cap_ix]
            for cen_ix, cap_ix in zip(center_order, capacities_order)
        }
    else:
        raise ValueError("Invalid capacity domain")

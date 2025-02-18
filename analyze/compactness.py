import numpy as np
import pandas as pd
import math
import sys

sys.path.append(".")
from data.partition import Partition
from data.shape_df import ShapeDataFrame
from data.config import SHPConfig


def polsby_popper(subregion: list[int], shape_df: ShapeDataFrame) -> float:
    subregion_df = shape_df.get_subregion_df(subregion)
    subregion_shape = subregion_df.geometry.union_all()
    return 4 * math.pi * subregion_shape.area / subregion_shape.length**2


def polsby_poppers(
    partition: Partition, shape_df: ShapeDataFrame
) -> np.ndarray:
    scores_arr = np.zeros(partition.n_districts, dtype=float)
    for id, subregion in partition.get_parts().items():
        scores_arr[id] = polsby_popper(subregion, shape_df)
    return scores_arr


"""
from constants import RESULTS_PATH
import os
from data.partition import Partitions
from data.config import StateConfig

config = StateConfig("VA", 2010, "block_group")
shape_df = ShapeDataFrame.from_config(config)
partitions = Partitions.from_csv(
    os.path.join(
        RESULTS_PATH,
        config.get_dirname(),
        "state_house_POCVAP",
        "partitions",
        "shp.csv",
    )
)
pps = list(polsby_poppers(partitions.get_plan(0), shape_df).values())
print(pps)
print(sum(pps) / len(pps))
"""

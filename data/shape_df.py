import pandas as pd
import geopandas as gpd
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
import sys

sys.path.append(".")
import constants
from data.config import StateConfig
from data.partition import Partition


class ShapeDataFrame(gpd.GeoDataFrame):
    @classmethod
    def from_gdf(cls, gdf: gpd.GeoDataFrame) -> "ShapeDataFrame":
        return cls(gdf)

    @classmethod
    def from_config(cls, state_config: StateConfig) -> "ShapeDataFrame":
        shape_df = gpd.read_file(
            os.path.join(
                constants.CENSUS_SHAPE_PATH,
                state_config.get_dirname(),
            )
        )
        shape_df = shape_df.to_crs("EPSG:3078")  # meters
        # cgus = cgus[cgus.ALAND > 0] TODO: decide whether this is needed anywhere or not?
        if "GEOID10" in shape_df.columns:
            shape_df.rename(columns={"GEOID10": "GEOID"}, inplace=True)
        shape_df = shape_df.sort_values(by="GEOID").reset_index(drop=True)
        if state_config.subregion is not None:
            shape_df = shape_df.loc[state_config.subregion]
        return cls.from_gdf(shape_df)

    def get_subregion_df(self, subregion):
        return self.from_gdf(self.loc[subregion])

    def get_district_shape_df(self, partition: Partition) -> "ShapeDataFrame":
        shape_df_copy = self.copy()
        shape_df_copy["Plan"] = partition.get_assignment()
        return self.from_gdf(shape_df_copy.dissolve(by="Plan"))

    def get_lengths(self, state_config: StateConfig) -> np.ndarray:
        optimization_cache_path = os.path.join(
            constants.OPT_DATA_PATH, state_config.get_dirname()
        )
        lengths_path = os.path.join(optimization_cache_path, "lengths.npy")
        if os.path.exists(lengths_path):
            return np.load(lengths_path)
        else:
            centroids = pd.DataFrame(
                data={"x": self.centroid.x, "y": self.centroid.y}
            )
            lengths = squareform(pdist(centroids))
            if not os.path.exists(optimization_cache_path):
                os.mkdir(optimization_cache_path)
            np.save(lengths_path, lengths)
            return lengths

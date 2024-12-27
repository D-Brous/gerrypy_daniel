import pandas as pd
import geopandas as gpd
import os

import constants
from data.config import StateConfig


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
        # cgus = cgus[cgus.ALAND > 0]
        if "GEOID10" in shape_df.columns:
            shape_df.rename(columns={"GEOID10": "GEOID"}, inplace=True)
        return cls.from_gdf(
            shape_df.sort_values(by="GEOID").reset_index(drop=True)
        )


class DemoDataFrame(pd.DataFrame):
    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "DemoDataFrame":
        return cls(df)

    @classmethod
    def from_config(cls, state_config: StateConfig) -> "DemoDataFrame":
        acs_df = pd.read_csv(
            os.path.join(
                constants.DEMO_DATA_PATH,
                state_config.get_dirname(),
                "pops.csv",
            ),
            low_memory=False,
        )
        return cls.from_df(acs_df.sort_values("GEOID").reset_index(drop=True))


"""
class ACSDataFrame(pd.DataFrame):
    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "ACSDataFrame":
        return cls(df)

    @classmethod
    def from_config(cls, state_config: StateConfig) -> "ACSDataFrame":
        acs_df = pd.read_csv(
            os.path.join(
                constants.ACS_DATA_PATH,
                state_config.granularity,
                state_config.state,
                str(state_config.year),
                "acs5.csv",
            ),
            low_memory=False,
        )
        return cls.from_df(acs_df.sort_values("GEOID").reset_index(drop=True))
"""

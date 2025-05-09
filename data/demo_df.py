import pandas as pd
import os
import sys

sys.path.append(".")
import constants
from data.config import StateConfig


class DemoDataFrame(pd.DataFrame):
    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "DemoDataFrame":
        return cls(df)

    @classmethod
    def from_config(
        cls,
        state_config: StateConfig,
        interpretation: constants.Interpretation = "any_part",
    ) -> "DemoDataFrame":
        demo_df = (
            pd.read_csv(
                os.path.join(
                    constants.DEMO_DATA_PATH,
                    state_config.get_dirname(),
                    f"pops_{interpretation}.csv",
                ),
                low_memory=False,
            )
            .sort_values("GEOID")
            .reset_index(drop=True)
        )
        if state_config.subregion is not None:
            demo_df = demo_df.loc[state_config.subregion]
        return cls.from_df(demo_df)

    def get_n_cgus(self) -> int:
        return len(self)

    def get_subregion_df(self, subregion):
        """_summary_

        Args:
            subregion (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.from_df(self.loc[subregion])

    def get_ideal_pop(self, n_districts: int) -> float:
        """Returns the ideal district population given that we want to
        have n_districts many districts

        Args:
            n_districts (int): Number of districts
        """
        return self["POP"].sum() / n_districts

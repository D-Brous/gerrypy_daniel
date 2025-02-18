import numpy as np
import pandas as pd
import sys

sys.path.append(".")
from constants import VapCol
from data.demo_df import DemoDataFrame
from data.partition import Partition


def cvap(col: VapCol, subregion: list[int], demo_df: DemoDataFrame) -> int:
    """Returns the size of the voting age population of citizens from
    VapCol col in the subregion in demo_df.

    Args:
        col (VapCol): Name of citizen voting age population column in
            the demo_df
        subregion (list[int]): List of cgu indices of the subregion in
            the demo_df
        demo_df (DemoDataFrame): Dataframe of demographic data
    """
    return demo_df.loc[subregion, col].sum()


def cvap_prop(
    col: VapCol, subregion: list[int], demo_df: DemoDataFrame
) -> float:
    """Returns the proportion of the voting age population of citizens
    from VapCol col in the subregion in demo_df.

    Args:
        col (VapCol): Name of citizen voting age population column in
            the demo_df
        subregion (list[int]): List of cgu indices of the subregion in
            the demo_df
        demo_df (DemoDataFrame): Dataframe of demographic data
    """
    return cvap(col, subregion, demo_df) / cvap("VAP", subregion, demo_df)


def is_maj_cvap(
    col: VapCol, subregion: list[int], demo_df: DemoDataFrame
) -> bool:
    """Returns whether or not the proportion of the voting age population
    of citizens from VapCol col in the subregion in demo_df is greater
    than 1/2

    Args:
        col (VapCol): Name of citizen voting age population column in
            the demo_df
        subregion (list[int]): List of cgu indices of the subregion in
            the demo_df
        demo_df (DemoDataFrame): Dataframe of demographic data
    """
    return cvap_prop(col, subregion, demo_df) > 0.5


def cvap_props(
    col: VapCol, partition: Partition, demo_df: DemoDataFrame
) -> list[float]:
    """Returns the proportions of the voting age population of citizens
    from VapCol col in the districts of partition.

    Args:
        col (VapCol): Name of citizen voting age population column in
            the demo_df
        partition (Partition): Partition of the state
        demo_df (DemoDataFrame): Dataframe of demographic data
    """
    return [
        cvap_prop(col, district_subregion, demo_df)
        for district_subregion in partition.get_parts().values()
    ]


# TODO: Maybe merge these two props functions so they both check for zeros in the denominator


def cvap_props_cgus(col: VapCol, demo_df: DemoDataFrame) -> list[float]:
    """Returns the proportions of the voting age population of citizens
    from VapCol col in the cgus of demo_df. For any cgus with zero
    voting age population, the proportion returned is 0.

    Args:
        col (VapCol): Name of citizen voting age population column in
            the demo_df
        demo_df (DemoDataFrame): Dataframe of demographic data
    """
    vap = demo_df["VAP"].to_numpy()
    return list(demo_df[col].to_numpy() / np.where(vap == 0, 1, vap))


def n_maj_cvap(
    col: VapCol, partition: Partition, demo_df: DemoDataFrame
) -> int:
    """Returns the total number of districts in partition for which the
    citizens of VapCol col form a majority.

    Args:
        col (VapCol): Name of citizen voting age population column in
            the demo_df
        partition (Partition): Partition of the state
        demo_df (DemoDataFrame): Dataframe of demographic data
    """
    return sum(
        int(is_maj_cvap(col, district_subregion, demo_df))
        for district_subregion in partition.get_parts().values()
    )

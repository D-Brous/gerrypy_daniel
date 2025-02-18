import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(".")
from data.config import SHPConfig
from data.demo_df import DemoDataFrame
from data.shape_df import ShapeDataFrame
from data.partition import Partitions
from analyze.maj_min import n_maj_cvap
from analyze.compactness import polsby_poppers
from constants import VapCol, SHORTBURSTS_MAXIMUMS

"""
def n_maj_cvap_arr(
    col: VapCol, partitions: Partitions, demo_df: DemoDataFrame
) -> np.ndarray:
    plan_ids = partitions.get_plan_ids()
    arr = np.zeros(len(plan_ids), dtype=int)
    for plan_id in plan_ids:
        arr[plan_id] = n_maj_cvap(col, partitions.get_plan(plan_id), demo_df)
    return arr


def avg_ppc_arr(partitions: Partitions, shape_df: ShapeDataFrame) -> np.ndarray:
    plan_ids = partitions.get_plan_ids()
    arr = np.zeros(len(plan_ids), dtype=float)
    for plan_id in plan_ids:
        polsby_popper_scores = list(
            polsby_poppers(partitions.get_plan(plan_id), shape_df).values()
        )
        arr[plan_id] = sum(polsby_popper_scores) / len(polsby_popper_scores)
    return arr


def get_partitions(
    config: SHPConfig,
) -> tuple[Partitions, Partitions, Partitions]:
    partitions_path = os.path.join(config.get_save_path(), "partitions")
    partitions_shp = Partitions.from_csv(
        os.path.join(partitions_path, "shp.csv")
    )
    partitions_shp_br = Partitions.from_csv(
        os.path.join(partitions_path, "shp_br.csv")
    )
    partitions_pp = Partitions()
    for plan_id in range(10):
        partitions_pp_plan = Partitions.from_csv(
            os.path.join(
                partitions_path, f"shp_p{plan_id}_priority_4_opt_no_br.csv"
            )
        )
        ids = partitions_pp_plan.get_plan_ids()
        if len(ids) == 0:
            partition = partitions_shp.get_plan(plan_id)
        else:
            partition = partitions_pp_plan.get_plan(max(ids))
        partitions_pp.set_plan(plan_id, partition)
    return partitions_shp, partitions_shp_br, partitions_pp
"""


class PlotGenerator:
    WHISKER_FIGSIZE = (7, 3.5)
    SCATTER_FIGSIZE = (7, 3.5)
    WHISKER_FILE_NAME = "whisker.pdf"
    SCATTER_FILE_NAME = "scatter.pdf"

    def __init__(self, config: SHPConfig):
        self.config = config
        save_path = config.get_save_path()
        self.plot_path = os.path.join(save_path, "plots")
        self.scores_path = os.path.join(save_path, "compactness_scores")
        self.demo_df = DemoDataFrame.from_config(config)
        self.shape_df = ShapeDataFrame.from_config(config)

    def get_partitions(
        self,
    ) -> tuple[Partitions, Partitions, Partitions]:
        partitions_path = os.path.join(
            self.config.get_save_path(), "partitions"
        )
        partitions_shp = Partitions.from_csv(
            os.path.join(partitions_path, "shp.csv")
        )
        partitions_shp_br = Partitions.from_csv(
            os.path.join(partitions_path, "shp_br.csv")
        )
        partitions_pp = Partitions()
        for plan_id in range(10):
            partitions_pp_plan = Partitions.from_csv(
                os.path.join(
                    partitions_path, f"shp_p{plan_id}_priority_4_opt_no_br.csv"
                )
            )
            ids = partitions_pp_plan.get_plan_ids()
            if len(ids) == 0:
                partition = partitions_shp.get_plan(plan_id)
            else:
                partition = partitions_pp_plan.get_plan(max(ids))
            partitions_pp.set_plan(plan_id, partition)
        return partitions_shp, partitions_shp_br, partitions_pp

    def n_maj_cvap_arr(self, partitions: Partitions) -> np.ndarray:
        plan_ids = partitions.get_plan_ids()
        arr = np.zeros(len(plan_ids), dtype=int)
        for plan_id in plan_ids:
            arr[plan_id] = n_maj_cvap(
                self.config.col, partitions.get_plan(plan_id), self.demo_df
            )
        return arr

    def avg_ppc_arr(self, partitions: Partitions, file_name: str) -> np.ndarray:
        plan_ids = partitions.get_plan_ids()
        n_districts = self.config.n_districts
        scores_file_path = os.path.join(self.scores_path, file_name)
        try:
            scores_arr = np.load(scores_file_path)
            avgs_arr = np.mean(scores_arr, axis=1)
        except:
            if not os.path.exists(self.scores_path):
                os.mkdir(self.scores_path)
            scores_arr = np.zeros((len(plan_ids), n_districts), dtype=float)
            avgs_arr = np.zeros(len(plan_ids), dtype=float)
            for plan_id in plan_ids:
                scores_arr[plan_id] = polsby_poppers(
                    partitions.get_plan(plan_id), self.shape_df
                )
                avgs_arr[plan_id] = np.mean(scores_arr[plan_id])
            np.save(scores_file_path, scores_arr)
        return avgs_arr

    def save_fig(self, fig: Figure, pdf_path: str):
        fig.savefig(pdf_path, bbox_inches="tight", format="pdf", dpi=300)

    def set_title(self, ax: Axes):
        ax.set_title(
            f"{self.config.state} State House {self.config.col} ({self.config.n_districts} seats)"
        )

    def set_xticks(self, ax: Axes):
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    def set_xlabel(self, ax: Axes):
        ax.set_xlabel(f"Number of Majority {self.config.col} Districts")

    def proceed_ok(self, pdf_path):
        if not os.path.exists(self.plot_path):
            os.mkdir(self.plot_path)
        else:
            if os.path.exists(pdf_path):
                while True:
                    response = input(
                        f"The plot file {pdf_path} already exists. Do you want"
                        " to replace it? (y/n)\n"
                    )
                    if response == "y":
                        return True
                    elif response == "n":
                        return False
            else:
                return True

    def plot_whisker_bar(
        self,
        points: np.ndarray,
        ax: Axes,
        color: np.ndarray,
        y: int,
        label: str,
    ):
        avg = np.mean(points)
        ax.errorbar(
            x=avg,
            y=y,
            xerr=[
                [avg - min(points)],
                [max(points) - avg],
            ],
            fmt="none",
            ecolor=color,
            capsize=5,
            label=label,
        )
        ax.scatter(
            avg,
            y,
            color=color,
            s=50,
            zorder=10,
        )

    def make_whisker_plot(self):
        pdf_path = os.path.join(self.plot_path, PlotGenerator.WHISKER_FILE_NAME)
        if not self.proceed_ok(pdf_path):
            return
        partitions_shp, partitions_shp_br, partitions_pp = self.get_partitions()
        n_maj_cvaps_shp = self.n_maj_cvap_arr(partitions_shp)
        n_maj_cvaps_shp_br = self.n_maj_cvap_arr(partitions_shp_br)
        n_maj_cvaps_pp = self.n_maj_cvap_arr(partitions_pp)
        fig, ax = plt.subplots(figsize=PlotGenerator.WHISKER_FIGSIZE)
        # Plot vertical bar
        sb_max = SHORTBURSTS_MAXIMUMS[self.config.state][self.config.col]
        ax.axvline(
            x=sb_max,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Short Bursts Maximum",
        )

        # Plot horizontal bars with whiskers
        self.plot_whisker_bar(n_maj_cvaps_shp, ax, "red", 3, "Base")
        self.plot_whisker_bar(
            n_maj_cvaps_shp_br, ax, "green", 2, "Beta Reoptimized"
        )
        self.plot_whisker_bar(
            n_maj_cvaps_pp, ax, "blue", 1, "Local Reoptimized"
        )

        # Finish up plot
        self.set_xlabel(ax)
        self.set_xticks(ax)
        ax.set_ylabel("Set of Plans")
        # ax.set_yticks(
        #     [3, 2, 1], ["Base", "Beta Reoptimized", "Local Reoptimized"]
        # )
        ax.set_yticks([])
        self.set_title(ax)
        ax.legend(loc="upper right")
        self.save_fig(fig, pdf_path)

    def make_scatter_plot(self):
        pdf_path = os.path.join(self.plot_path, PlotGenerator.SCATTER_FILE_NAME)
        if not self.proceed_ok(pdf_path):
            return
        partitions_shp, partitions_shp_br, partitions_pp = self.get_partitions()
        n_maj_cvaps_shp = self.n_maj_cvap_arr(partitions_shp)
        n_maj_cvaps_shp_br = self.n_maj_cvap_arr(partitions_shp_br)
        n_maj_cvaps_pp = self.n_maj_cvap_arr(partitions_pp)
        ppcs_shp = self.avg_ppc_arr(partitions_shp, "polsby_poppers_shp.npy")
        ppcs_shp_br = self.avg_ppc_arr(
            partitions_shp_br, "polsby_poppers_shp_br.npy"
        )
        ppcs_pp = self.avg_ppc_arr(partitions_pp, "polsby_poppers_pp_pr.npy")
        fig, ax = plt.subplots(figsize=PlotGenerator.SCATTER_FIGSIZE)
        ax.scatter(
            n_maj_cvaps_shp, ppcs_shp, color="red", marker="s", label="Base"
        )
        ax.scatter(
            n_maj_cvaps_shp_br,
            ppcs_shp_br,
            color="green",
            marker="o",
            label="Beta Reoptimized",
        )
        ax.scatter(
            n_maj_cvaps_pp,
            ppcs_pp,
            color="blue",
            marker="d",
            label="Local Reoptimized",
        )
        self.set_xticks(ax)
        ax.legend(loc="upper right")
        self.set_xlabel(ax)
        ax.set_ylabel("Average Polsby Popper Compactness")
        self.set_title(ax)
        self.save_fig(fig, pdf_path)


if __name__ == "__main__":
    from experiments.LA_house import shp_config as la_shp_config
    from experiments.TX_house import shp_config as tx_shp_config
    from experiments.VA_house import shp_config as va_shp_config
    from experiments.NM_house import shp_config as nm_shp_config

    for col in ["BVAP", "POCVAP"]:
        la_shp_config.col = col
        la_shp_config.save_dirname = f"state_house_{col}"
        la_pg = PlotGenerator(la_shp_config)
        la_pg.make_whisker_plot()
        la_pg.make_scatter_plot()
    for col in ["BVAP", "HVAP"]:
        tx_shp_config.col = col
        tx_shp_config.save_dirname = f"state_house_{col}"
        tx_pg = PlotGenerator(tx_shp_config)
        tx_pg.make_whisker_plot()
        tx_pg.make_scatter_plot()
    for col in ["BVAP", "POCVAP"]:
        va_shp_config.col = col
        va_shp_config.save_dirname = f"state_house_{col}"
        va_pg = PlotGenerator(va_shp_config)
        va_pg.make_whisker_plot()
        va_pg.make_scatter_plot()
    for col in ["HVAP", "POCVAP"]:
        nm_shp_config.col = col
        nm_shp_config.save_dirname = f"state_house_{col}"
        nm_pg = PlotGenerator(nm_shp_config)
        nm_pg.make_whisker_plot()
        nm_pg.make_scatter_plot()

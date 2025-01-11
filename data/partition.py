import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import pickle
import sys

sys.path.append(".")
from data.df import DemoDataFrame
from data.config import SHPConfig, StateConfig


class Partition:
    def __init__(self, n_districts: int, n_cgus: Optional[int] = None):
        self.n_districts = n_districts
        self.n_cgus = n_cgus
        self.districts = dict()
        for district_id in range(n_districts):
            self.districts[district_id] = []

    @classmethod
    def from_assignment_ser(cls, assignment_ser: pd.Series):
        n_districts = assignment_ser.nunique()
        n_cgus = len(assignment_ser)
        partition = cls(n_districts, n_cgus=n_cgus)
        partition.districts = {
            district_id: assignment_ser[
                assignment_ser == district_id
            ].index.tolist()
            for district_id in range(n_districts)
        }
        return partition

    def get_assignment(self) -> np.ndarray[int]:
        if self.n_cgus is None:
            self.n_cgus = sum(len(cgus) for cgus in self.districts.values())
        assignment = np.zeros((self.n_cgus), dtype=int)
        for district, cgus in self.districts.items():
            assignment[cgus] = district
        return assignment

    def get_parts(self) -> Dict[int, List[int]]:
        return self.districts

    def get_part(self, district: int) -> List[int]:
        if district not in self.districts:
            raise ValueError(
                f"Expected int in the interval [0, {self.n_districts - 1}] but got {district}"
            )
        return self.districts[district]

    def set_part(self, district: int, cgus: List[int]) -> None:
        if district < 0 or district >= self.n_districts:
            raise ValueError(
                f"Expected int in the interval [0, {self.n_districts - 1}] but got {district}"
            )
        self.districts[district] = cgus


class Partitions:
    def __init__(self):
        self.dict = dict()

    @classmethod
    def from_object_file(cls, obj_path: str) -> "Partitions":
        return pickle.load(open(obj_path, "rb"))

    def set(self, id: int, partition: Partition):
        self.dict[id] = partition

    def get(self, id: int) -> Partition:
        if id not in self.dict:
            raise ValueError(f"{id} is not a valid id in this partition dict")
        return self.dict[id]

    def to_object_file(self, obj_path: str):
        pickle.dump(self, open(obj_path, "wb"))

    def to_csv(self, csv_path: str, config: StateConfig):
        partitions_df = pd.DataFrame()
        demo_df = DemoDataFrame.from_config(config)
        partitions_df["GEOID"] = demo_df["GEOID"]
        for id, partition in self.dict.items():
            partitions_df[f"Plan {id}"] = partition.get_assignment()
        partitions_df = partitions_df.set_index("GEOID")
        partitions_df.to_csv(csv_path)

    @classmethod
    def from_csv(cls, csv_path: str) -> "Partitions":
        partitions = cls()
        df = pd.read_csv(csv_path)
        cols = df.columns.to_list()[1:]
        for col in cols:
            partition = Partition.from_assignment_ser(df[col])
            partitions.set(int(col[5:]), partition)
        return partitions


if __name__ == "__main__":

    import time

    config = StateConfig("LA", 2010, "block_group")
    n_trials = 10
    t_init = time.thread_time()
    for i in range(n_trials):
        partitions = Partitions.from_csv("test/assignments.csv")
    print(f"csv load time: {time.thread_time() - t_init}")

    for i in range(n_trials):
        partitions_2 = Partitions.from_object_file("test/assignments.p")
    print(f"object load time: {time.thread_time() - t_init}")

    partitions = Partitions.from_object_file("test/assignments.p")
    for i in range(n_trials):
        partitions.to_csv("test/assignments_2.csv", config)
    print(f"csv save time: {time.thread_time() - t_init}")

    for i in range(n_trials):
        partitions.to_object_file("test/assignments_2.p")
    print(f"object save time: {time.thread_time() - t_init}")

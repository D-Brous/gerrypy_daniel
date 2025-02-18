import pandas as pd
from typing import Optional
import pickle
import copy
import sys

sys.path.append(".")
from constants import flatten
from data.demo_df import DemoDataFrame
from data.config import StateConfig


class Partition:
    """Stores partition as a dict, where the keys are the district ids
    and the values are the district subregions, which are themselves
    lists of ints representing cgu ids. District ids must be integers
    and if there are n districts in the partition, then the district ids
    must be the integers from 0 to n-1 inclusive.
    """

    def __init__(self, n_districts: int, n_cgus: Optional[int] = None):
        self.n_districts = n_districts
        self.n_cgus = n_cgus
        self.districts = dict()
        for district_id in range(n_districts):
            self.districts[district_id] = []

    @classmethod
    def from_assignment_ser(cls, assignment_ser: pd.Series) -> "Partition":
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

    def get_subpartition(self, district_ids: tuple[int]) -> "Partition":
        n_districts = len(district_ids)
        subpartition = Partition(n_districts=n_districts)
        for district_ix in range(n_districts):
            subpartition.set_part(
                district_ix, self.get_part(district_ids[district_ix])
            )
        return subpartition

    def get_assignment(self) -> pd.Series:
        sorted_region = sorted(self.get_region())
        assignment = pd.Series(index=sorted_region)
        for district_id, district_subregion in self.districts.items():
            assignment.loc[district_subregion] = district_id
        return assignment
        """
        if self.n_cgus is None:
            self.n_cgus = sum(len(cgus) for cgus in self.districts.values())
        assignment = np.zeros((self.n_cgus), dtype=int)
        for district_id, district_subregion in self.districts.items():
            assignment[district_subregion] = district_id
        return assignment
        """

    def get_parts(self) -> dict[int, list[int]]:
        return self.districts

    def get_part(self, district_id: int) -> list[int]:
        if district_id not in self.districts:
            raise ValueError(
                f"Expected int in the interval [0, {self.n_districts - 1}] but got {district_id}"
            )
        return self.districts[district_id]

    def set_part(self, district_id: int, district_subregion: list[int]) -> None:
        if district_id < 0 or district_id >= self.n_districts:
            raise ValueError(
                f"Expected int in the interval [0, {self.n_districts - 1}] but got {district_id}"
            )
        self.districts[district_id] = district_subregion

    def get_region(self) -> list[int]:
        return flatten(list(self.get_parts().values()))

    def update_via_subpartition(
        self, subpartition: "Partition", district_ids: tuple[int]
    ):
        for (
            district_ix,
            district_subregion,
        ) in subpartition.get_parts().items():
            self.set_part(district_ids[district_ix], district_subregion)

    def __deepcopy__(self, memo):
        partition_copy = Partition(self.n_districts, self.n_cgus)
        partition_copy.districts = copy.deepcopy(self.districts)
        return partition_copy


class Partitions:
    """Stores dict of partitions (plans), where the keys are plan ids.
    The plan ids can be arbitrary integers, and the dict is unorded.
    """

    def __init__(self):
        self.dict = dict()

    @classmethod
    def from_object_file(cls, obj_path: str) -> "Partitions":
        return pickle.load(open(obj_path, "rb"))

    def set_plan(self, id: int, partition: Partition):
        self.dict[id] = partition

    def get_plan(self, id: int) -> Partition:
        if id not in self.dict:
            raise ValueError(f"{id} is not a valid id in this partition dict")
        return self.dict[id]

    def get_plan_ids(self) -> list[int]:
        return list(self.dict.keys())

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
            partitions.set_plan(int(col[5:]), partition)
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

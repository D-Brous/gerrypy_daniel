import numpy as np
from typing import Optional, Dict, List


class Partition:
    def __init__(self, n_districts: int, n_cgus: Optional[int] = None):
        self.n_districts = n_districts
        self.n_cgus = n_cgus
        self.districts = dict()
        for district in range(n_districts):
            self.districts[district] = []

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

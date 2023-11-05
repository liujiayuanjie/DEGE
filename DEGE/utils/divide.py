from typing import List
import numpy as np
from numpy import ndarray

class Divide:
    def __init__(self, seed: int = 0) -> None:
        self.__rdm = np.random.RandomState(seed)

    def __call__(self, items: ndarray, part_nums: List[int]) -> List[ndarray]:
        idx = np.arange(len(items))
        self.__rdm.shuffle(idx)

        bounds = [0] + np.cumsum(part_nums, axis = 0).tolist()
        parts = [items[idx[l: r]] for l, r in zip(bounds[: -1], bounds[1:])]

        return parts
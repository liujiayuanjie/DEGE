from typing import Any
import math
import numpy as np
from numpy import ndarray

class Batcher:
    def __init__(self, items: ndarray, batch_size: int) -> None:
        self.__items = items
        self.__batch_size = batch_size

    def __shuffle(self) -> None:
        total_size = len(self.__items)

        self.__batch_num = math.ceil(total_size // self.__batch_size)
        self.__cur = 0
        self.__idx = np.arange(total_size)

        np.random.shuffle(self.__idx)
    
    def __iter__(self) -> Any:
        self.__shuffle()
        return self
    
    def __next__(self) -> ndarray:
        if self.__cur >= self.__batch_num:
            raise StopIteration

        cur = self.__cur
        size = self.__batch_size
        idx = self.__idx[cur * size: (cur + 1) * size]
        self.__cur += 1

        items = self.__items[idx]

        return items
from typing import List

class Averager:
    def __init__(self) -> None:
        self.__total = 0.0
        self.__num = 0
    
    def __call__(self, *vals: List[float]) -> None:
        self.__total += sum(vals)
        self.__num += len(vals)

    @property
    def val(self) -> float:
        return 0.0 if self.__num == 0 else self.__total / self.__num
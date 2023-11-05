from typing import Callable, Any
import time

class Timer:
    def __init__(self) -> None:
        self.__dur = 0.0
    
    def __call__(self, fn: Callable, *args: Any, **kwds: Any) -> Any:
        start = time.time()
        res = fn(*args, **kwds)
        self.__dur += time.time() - start

        return res
    
    @property
    def dur(self) -> float:
        return self.__dur
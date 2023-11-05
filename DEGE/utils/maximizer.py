class Maximizer:
    def __init__(self) -> None:
        self.__max = 0
    
    def __call__(self, val: float) -> bool:
        if self.__max == None:
            self.__max = val
            return True

        elif val > self.__max:
            self.__max = val
            return True

        else:
            return False

    @property
    def val(self) -> float:
        return self.__max
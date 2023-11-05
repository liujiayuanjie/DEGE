from typing import Any
import torch
from torch import Tensor
from numpy import ndarray

class Device:
    def __init__(self, dev_name: str) -> None:
        self.__device = torch.device(dev_name)
    
    def attach(self, e: Tensor) -> Tensor:
        return e.to(self.__device)
    
    def tolist(self, e: Tensor) -> Any:
        return e.cpu().detach().numpy().tolist()
    
    def numpy(self, e: Tensor) -> ndarray:
        return e.cpu().detach().numpy()

    def tensor(self, *args: Any, **kargs: Any) -> Tensor:
        return torch.tensor(*args, **kargs).to(self.__device)

    def arange(self, *args: Any, **kargs: Any) -> Tensor:
        return torch.arange(*args, **kargs).to(self.__device)

    def zeros(self, *args: Any, **kargs: Any) -> Tensor:
        return torch.zeros(*args, **kargs).to(self.__device)

    def ones(self, *args: Any, **kargs: Any) -> Tensor:
        return torch.ones(*args, **kargs).to(self.__device)

    def randint(self, *args: Any, **kargs: Any) -> Tensor:
        return torch.randint(*args, **kargs).to(self.__device)

    def rand(self, *args: Any, **kargs: Any) -> Tensor:
        return torch.rand(*args, **kargs).to(self.__device)
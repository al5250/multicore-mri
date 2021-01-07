from abc import abstractmethod, ABC
from typing import List

from numpy import ndarray


class Metric(ABC):

    @abstractmethod
    def __call__(self, pred: List[ndarray], targ: List[ndarray]) -> float:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

from abc import abstractmethod, ABC
from typing import List

from numpy import ndarray


class KSampler(ABC):

    @abstractmethod
    def __call__(self, kspaces_full: List[ndarray]) -> List[ndarray]:
        pass

    @property
    @abstractmethod
    def masks(self) -> List[ndarray]:
        pass

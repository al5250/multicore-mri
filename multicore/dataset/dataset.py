from abc import abstractmethod, ABC
from typing import List, Tuple

from numpy import ndarray


class MRIDataset(ABC):

    @property
    @abstractmethod
    def imgs(self) -> List[ndarray]:
        pass

    @property
    @abstractmethod
    def kspaces(self) -> List[ndarray]:
        pass

    @property
    @abstractmethod
    def kmasks(self) -> List[ndarray]:
        pass

    @property
    @abstractmethod
    def img_size(self) -> Tuple[int]:
        pass

    @property
    @abstractmethod
    def names(self) -> List[str]:
        pass

from abc import abstractmethod, ABC
from typing import Dict, List, Optional

from numpy import ndarray

from reconstruction.dataset import MRIDataset
from reconstruction.logger import Logger


class ReconstructionAlgorithm(ABC):

    @abstractmethod
    def reconstruct(self, dataset: MRIDataset, logger: Logger) -> List[ndarray]:
        pass

    def __str__(self):
        return self.__class__.__name__

from abc import abstractmethod, ABC
from typing import Dict, List, Optional

from numpy import ndarray

from multicore.dataset import MRIDataset
from multicore.logger import Logger


class ReconstructionAlgorithm(ABC):

    @abstractmethod
    def reconstruct(self, dataset: MRIDataset, logger: Logger) -> List[ndarray]:
        pass

    def __str__(self):
        return self.__class__.__name__

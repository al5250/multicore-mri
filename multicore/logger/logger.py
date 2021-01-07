from abc import abstractmethod, ABC
from typing import Dict, Optional

from numpy import ndarray


class Logger(ABC):

    @abstractmethod
    def log_vals(
        self,
        tag: str,
        vals: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        pass

    @abstractmethod
    def log_imgs(
        self,
        tag: str,
        imgs: Dict[str, ndarray],
        step: Optional[int] = None
    ) -> None:
        pass

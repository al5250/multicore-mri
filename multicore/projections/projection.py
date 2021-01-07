from abc import abstractmethod, ABC

from torch import Tensor


class Projection(ABC):

    @abstractmethod
    def apply(self, data: Tensor) -> Tensor:
        pass

    @abstractmethod
    def T_apply(self, data: Tensor) -> Tensor:
        pass

    def __call__(self, data: Tensor) -> Tensor:
        return self.apply(data)

    def T(self, data: Tensor) -> Tensor:
        return self.T_apply(data)

    def __str__(self) -> str:
        return self.__class__.__name__

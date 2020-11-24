import torch
from torch import Tensor

from reconstruction.projections.projection import Projection


class GradientTransform(Projection):

    def __init__(self, dim: int, zeros_first: bool = True) -> None:
        self.dim = dim
        if not zeros_first:
            raise NotImplementedError()
        self.zeros_first = zeros_first

    def apply(self, x: Tensor) -> Tensor:
        N = x.size(self.dim)
        y1 = -x.narrow(self.dim, N-1, 1)
        y2 = x.narrow(self.dim, 1, N-1) - x.narrow(self.dim, 0, N-1)
        y = torch.cat([y1, y2], dim=self.dim)
        return y

    def T_apply(self, y: Tensor) -> Tensor:
        size = list(y.size())
        N = size[self.dim]
        size[self.dim] = 1
        x1 = torch.zeros(size).to(y)
        x2 = y.narrow(self.dim, 1, N-1).cumsum(dim=self.dim)
        x = torch.cat([x1, x2], dim=self.dim)
        return x

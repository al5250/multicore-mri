import torch
from torch import Tensor
import numpy as np

from multicore.projections.projection import Projection


class GradientTransform(Projection):

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def apply(self, x: Tensor) -> Tensor:
        # N = x.size(self.dim)
        # y1 = -x.narrow(self.dim, N-1, 1)
        # y2 = x.narrow(self.dim, 1, N-1) - x.narrow(self.dim, 0, N-1)
        # y = torch.cat([y1, y2], dim=self.dim)
        # return y
        # y = x - torch.roll(x, shifts=1, dims=self.dim)
        y = torch.flip(torch.cumsum(torch.flip(x, dims=[self.dim]), dim=self.dim), dims=[self.dim])
        return y

    def T_apply(self, y: Tensor) -> Tensor:
        # size = list(y.size())
        # N = size[self.dim]
        # size[self.dim] = 1
        # x1 = torch.zeros(size).to(y)
        # x2 = y.narrow(self.dim, 1, N-1).cumsum(dim=self.dim)
        # x = torch.cat([x1, x2], dim=self.dim)
        # x = y - torch.roll(y, shifts=-1, dims=self.dim)
        x = torch.cumsum(y, dim=self.dim)
        return x


class FourierGradientTransform(Projection):

    def __init__(self, vertical: bool = True) -> None:
        self.vertical = vertical

    def apply(self, x: Tensor) -> Tensor:
        _, _, H, W = x.size()
        if self.vertical:
            k = torch.arange(H, device=x.device).view(1, 1, -1, 1)
            y = (1 - torch.exp(-2 * np.pi * 1j * k / H)) * x
        else:
            k = torch.arange(W, device=x.device).view(1, 1, -1, 1)
            y = (1 - torch.exp(-2 * np.pi * 1j * k / W)) * x
        return y

    def T_apply(self, y: Tensor) -> Tensor:
        _, _, H, W = y.size()
        if self.vertical:
            k = torch.arange(H, device=y.device).view(1, 1, -1, 1)
            x = torch.conj(1 - torch.exp(-2 * np.pi * 1j * k / H)) * y
        else:
            k = torch.arange(W, device=y.device).view(1, 1, -1, 1)
            x = torch.conj(1 - torch.exp(-2 * np.pi * 1j * k / W)) * y
        return x

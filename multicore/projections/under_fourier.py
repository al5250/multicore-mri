from typing import Tuple

import torch
from torch import Tensor
from torch.fft import fftn, ifftn
import numpy as np
import pdb

from multicore.projections.projection import Projection


class UndersampledFourier2D(Projection):

    def __init__(self, mask: Tensor, real_input: bool = True, apply_grad=False) -> None:
        self.mask = mask
        self.real_input = real_input
        self.apply_grad = apply_grad

    def apply(self, x: Tensor) -> Tensor:
        y = fftn(x, dim=(-2, -1), norm='ortho')
        if self.apply_grad:
            _, _, H, W = y.size()
            k = torch.arange(H, device=y.device).view(1, 1, -1, 1)
            denom = torch.abs(k) ** 2
            denom[:, :, 0, :] = 1
            y = torch.conj(1 - torch.exp(-2 * np.pi * 1j * k / H)) / denom * y
        y[..., ~self.mask] = 0
        return y

    def T_apply(self, y: Tensor) -> Tensor:
        if not torch.all(y[..., ~self.mask] == 0):
            pdb.set_trace()
        if self.apply_grad:
            _, _, H, W = y.size()
            k = torch.arange(H, device=y.device).view(1, 1, -1, 1)
            denom = torch.abs(k) ** 2
            denom[:, :, 0, :] = 1
            y = (1 - torch.exp(-2 * np.pi * 1j * k / H)) / denom * y
        x = ifftn(y, dim=(-2, -1), norm='ortho')
        if self.real_input:
            x = x.real
        return x

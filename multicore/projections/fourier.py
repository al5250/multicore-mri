from typing import Tuple

import torch
from torch import Tensor
from torch.fft import fftn, ifftn

from multicore.projections.projection import Projection


class Undersampled2DFastFourierTransform(Projection):

    def __init__(self, index: Tensor, size: Tuple[int, int], real_input: bool = True) -> None:
        self.index = index
        self.size = size
        self.N = size[0] * size[1]
        self.K = index.size(-1)
        self.real_input = real_input

    def apply(self, x: Tensor) -> Tensor:
        x_ = x.unflatten(dim=-1, sizes=self.size)
        y_ = fftn(x_, dim=(-2, -1), norm='ortho')
        y_ = y_.flatten(start_dim=-2)
        out_size = y_.size()[:-1] + (self.K,)
        y = torch.gather(y_, dim=-1, index=self.index.expand(out_size))
        return y

    def T_apply(self, y: Tensor) -> Tensor:
        out_size = y.size()[:-1] + (self.N,)
        y_ = torch.zeros(out_size, device=y.device, dtype=y.dtype)
        y_.scatter_(dim=-1, index=self.index.expand_as(y), src=y)
        y_ = y_.unflatten(dim=-1, sizes=self.size)
        x_ = ifftn(y_, dim=(-2, -1), norm='ortho')
        if self.real_input:
            x_ = x_.real
        x = x_.flatten(start_dim=-2)
        return x

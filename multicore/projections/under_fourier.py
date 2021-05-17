from typing import Tuple

import torch
from torch import Tensor
from torch.fft import fftn, ifftn
import numpy as np
import pdb

from multicore.projections.projection import Projection


class UndersampledFourier2D(Projection):

    def __init__(self, mask: Tensor) -> None:
        self.mask = mask

    def apply(self, x: Tensor) -> Tensor:
        y = fftn(x, dim=(-2, -1), norm='ortho')
        y[..., ~self.mask] = 0
        return y

    def T_apply(self, y: Tensor) -> Tensor:
        if not torch.all(y[..., ~self.mask] == 0):
            raise ValueError('Invalid input to UndersampledFourier2D disobeys mask.')
        x = ifftn(y, dim=(-2, -1), norm='ortho')
        return x

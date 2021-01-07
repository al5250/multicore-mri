from typing import List

import torch
from torch import Tensor

from multicore.projections.projection import Projection


class Sequential(Projection):

    def __init__(self, projs: List[Projection], fwd_apply: List[bool]) -> None:
        self.projs = projs
        self.fwds = fwd_apply

    def apply(self, x: Tensor) -> Tensor:
        for proj, fwd in zip(self.projs, self.fwds):
            if fwd:
                x = proj(x)
            else:
                x = proj.T(x)
        return x

    def T_apply(self, x: Tensor) -> Tensor:
        for proj, fwd in zip(self.projs[::-1], self.fwds[::-1]):
            if fwd:
                x = proj.T(x)
            else:
                x = proj(x)
        return x

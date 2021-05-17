from typing import Tuple

import torch
from torch import Tensor

from multicore.projections.projection import Projection


class MultiCoilProjection(Projection):

    def __init__(self, sens: Tensor, real_input: bool = False) -> None:
        self.sens = sens
        self.real_input = real_input
        self.num_coils = sens.size(dim=0)

    def apply(self, x: Tensor) -> Tensor:
        if x.size(dim=-3) != 1:
            raise ValueError(
                'The third-to-last dimension of the input tensor must be 1.'
            )
        y = self.sens * x
        return y

    def T_apply(self, y: Tensor) -> Tensor:
        if y.size(dim=-3) != self.num_coils:
            raise ValueError(
                f'The third-to-last dimension of the input tensor must be equal to '
                f'the number of coils, i.e. {self.num_coils}.'
            )
        x = torch.sum(self.sens.conj() * y, dim=-3, keepdim=True)
        if self.real_input:
            x = x.real
        return x

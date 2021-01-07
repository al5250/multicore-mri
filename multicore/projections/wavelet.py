from typing import Optional, Tuple, List

import torch
from torch import Tensor

from pytorch_wavelets import DWTForward, DWTInverse

from multicore.projections.projection import Projection

import pdb


class Wavelet2D(Projection):

    def __init__(
        self,
        num_levels: int = 1,
        wave: str = 'db2',
        mode: str = 'periodization',
        device: Optional[str] = None,
    ) -> None:
        self.num_levels = num_levels
        self.wave = wave
        self.mode = mode

        self.xfm = DWTForward(J=num_levels, wave=wave, mode=mode).to(device)
        self.ifm = DWTInverse(wave=wave, mode=mode).to(device)

    def apply(self, x: Tensor) -> Tensor:
        x, shape = self.flatten4d(x)
        H, W = shape[-2:]
        assert H == W # Check that height is equal to width
        assert (H & (H-1) == 0) and H != 0 # Check power of 2

        y_low, y_high = self.xfm(x)

        y = self.coeffs_to_tensor(y_low, y_high)
        y = self.unflatten4d(y, shape)
        return y

    def T_apply(self, y: Tensor) -> Tensor:
        y, shape = self.flatten4d(y)
        y_low, y_high = self.tensor_to_coeffs(y, self.num_levels)
        x = self.ifm((y_low, y_high))
        x = self.unflatten4d(x, shape)
        return x

    @staticmethod
    def coeffs_to_tensor(y_low: Tensor, y_high: List[Tensor]) -> Tensor:
        out = y_low
        for yh in y_high[::-1]:
            ylh = yh[..., 0, :, :]
            yhl = yh[..., 1, :, :]
            yhh = yh[..., 2, :, :]
            out = torch.cat(
                [torch.cat([out, ylh], dim=-1), torch.cat([yhl, yhh], dim=-1)],
                dim=-2
            )
        return out

    @staticmethod
    def tensor_to_coeffs(y: Tensor, num_levels: int) -> Tuple[Tensor, List[Tensor]]:
        base_dim = y.size(dim=-1)
        dim = base_dim // (2 ** num_levels)
        y_low = y[..., :dim, :dim]
        y_high = []
        for i in range(num_levels):
            ylh = y[..., :dim, dim:2*dim]
            yhl = y[..., dim:2*dim, :dim]
            yhh = y[..., dim:2*dim, dim:2*dim]
            y_high.append(torch.stack([ylh, yhl, yhh], dim=2))
            dim *= 2
        y_high = y_high[::-1]
        return y_low, y_high

    @staticmethod
    def flatten4d(x: Tensor) -> Tuple[Tensor, Tuple[int, ...]]:
        shape = x.size()
        if len(shape) > 4:
            x = x.flatten(start_dim=0, end_dim=-4)
        return x, shape

    @staticmethod
    def unflatten4d(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
        if len(shape) > 4:
            x = x.unflatten(dim=0, sizes=shape[:-3])
        return x

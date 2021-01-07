from abc import abstractmethod
from typing import List, Optional

from numpy import ndarray
import numpy as np
from numpy.fft import ifftshift, ifft2, fftshift

from multicore.ksampler.ksampler import KSampler


class PowerRuleSampler(KSampler):

    def __init__(
        self,
        n_kspace: int,
        dim_x: 10,
        dim_y: 10,
        samp_type: 'y',
        power: int = 5,
        samp_factor: float = 0.5,
        norm_type: str = 'l2',
        keep_radius: float = 0.0,
        single_mask: bool = False,
        n_iter: int = 2,
        samp_tol: int = 0,
        symmetric: bool = False,
        seed: Optional[int] = None
    ) -> None:
        self.n_kspace = n_kspace

        self.dim_x = dim_x
        self.dim_y = dim_y

        if samp_type not in ['x', 'y', 'xy']:
            raise ValueError('Argument `samp_type` must be in ["x", "y", "xy"].')
        self.samp_type = samp_type

        if power < 0:
            raise ValueError('Argument `power` must be greater than 0.')
        self.power = power

        if samp_factor < 0 or samp_factor > 1:
            raise ValueError('Argument `samp_factor` must be between [0, 1].')
        self.samp_factor = samp_factor

        if norm_type not in ['l2', 'linf']:
            raise ValueError('Argument `norm_type` must be in ["l2", "linf"].')
        self.norm_type = norm_type

        self.keep_radius = keep_radius
        self.single_mask = single_mask
        self.n_iter = n_iter
        self.samp_tol = samp_tol
        self.symmetric = symmetric
        self.seed = seed

        self.probs = self._compute_probs()
        self.reset_masks()

    @property
    def masks(self) -> ndarray:
        return self._masks

    def _compute_probs(self) -> ndarray:
        x = np.zeros((1,)) if self.samp_type == 'y' else np.linspace(-1, 1, self.dim_x)
        y = np.zeros((1,)) if self.samp_type == 'x' else np.linspace(-1, 1, self.dim_y)
        [X, Y] = np.meshgrid(y, x)
        if self.norm_type == 'l2':
            R = np.sqrt(X**2 + Y**2)
            R = R / np.max(np.abs(R))
        elif self.norm_type == 'linf':
            R = np.max(X, Y)
        else:
            raise ValueError(f"Invalid attribute norm_type={self.norm_type}.")

        n_samps = np.floor(self.samp_factor * len(x) * len(y))
        probs = (1 - R) ** self.power
        keep_mask = (R < self.keep_radius)
        probs[keep_mask] = 1
        if np.floor(probs.sum()) > n_samps:
            raise ValueError('Infeasible without undersampling dc, increase p.')

        # Begin bisection
        minval = 0
        maxval = 1
        val = 0.5
        while True:
            val = minval / 2 + maxval / 2
            probs = (1 - R) ** self.power + val
            probs[probs > 1] = 1
            probs[keep_mask] = 1
            total_mass = np.floor(probs.sum())
            if total_mass > n_samps: # infeasible
                maxval = val
            elif total_mass < n_samps: # feasible but not optimal
                minval = val
            else:
                return probs

    def reset_masks(self) -> None:
        if self.seed is not None:
            np.random.seed(self.seed)
        if self.single_mask:
            m = self._sample_mask()
            self._masks = [m] * self.n_kspace
        else:
            self._masks = [self._sample_mask() for _ in range(self.n_kspace)]

    def _sample_mask(self) -> ndarray:
        n_samps = np.floor(self.samp_factor * self.probs.size)
        min_intr = float('inf')
        best_mask = np.zeros_like(self.probs)

        for _ in range(self.n_iter):
            mask = np.zeros_like(self.probs)
            while np.abs(mask.sum() - n_samps) > self.samp_tol:
                mask = (np.random.rand(*self.probs.shape) < self.probs)
                if self.symmetric:
                    sym_mask = np.zeros_like(mask)
                    img_size = np.max(mask.shape)
                    half_size = img_size // 2
                    if self.samp_type == 'x':
                        sym_mask[half_size:, :] = mask[half_size:, :]
                        sym_mask = (sym_mask + np.roll(np.flipud(sym_mask), (1, 0), axis=(0, 1))) > 0
                        mask = sym_mask
                    elif self.samp_type == 'y':
                        sym_mask[:, half_size:] = mask[:, half_size:]
                        sym_mask = (sym_mask + np.roll(np.fliplr(sym_mask), (0, 1), axis=(0, 1))) > 0
                        mask = sym_mask
                    elif self.samp_type == 'xy':
                        sym_mask[:, half_size:] = mask[:, half_size:]
                        sym_mask = (sym_mask + np.roll(np.flipud(np.fliplr(sym_mask)), (1, 1), axis=(0, 1))) > 0
                        mask = sym_mask

            mask_ifft = ifft2(mask / self.probs)
            intr = np.max(np.abs(mask_ifft))
            if intr < min_intr:
                min_intr = intr
                best_mask = mask

        if self.samp_type == 'x':
            out_mask = np.repeat(best_mask, self.dim_y, axis=1)
        elif self.samp_type == 'y':
            out_mask = np.repeat(best_mask, self.dim_x, axis=0)
        else:
            out_mask = best_mask

        out_mask = ifftshift(out_mask)
        return out_mask

    def __call__(self, kspaces_full: List[ndarray]) -> List[ndarray]:
        if len(kspaces_full) != self.n_kspace:
            raise ValueError(
                f'KSampler expects {self.n_kspace} items, but only '
                f'{len(kspaces_full)} were provided.'
            )
        return [m * k for k, m in zip(kspaces_full, self._masks)]

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

from numpy import ndarray
from numpy.fft import fftshift
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt

import pdb
import time
from hydra.utils import instantiate

from multicore.dataset import MRIDataset
from multicore.logger import Logger
from multicore.algorithm.algorithm import ReconstructionAlgorithm
from multicore.projections import (
    Projection,
    UndersampledFourier2D,
    MultiCoilProjection,
    Sequential
)
from multicore.utils import conj_grad, get_torch_device, get_torch_dtypes
from multicore.metric import RootMeanSquareError


class SENSE(ReconstructionAlgorithm):

    def __init__(
        self,
        cg_tol: float = 1e-5,
        max_cg_iters: int = 1e3,
        reg_penalty: float = 0.,
        compute_variances: bool = False,
        num_probes_for_variances: int = 30,
        device: Optional[str] = None,
        precision: str = 'double'
    ) -> None:
        # Set device and precision
        self.device = get_torch_device(device)
        dtype, self.cdtype = get_torch_dtypes(precision)
        torch.set_default_dtype(dtype)

        # Set algorithm parameters
        self.cg_tol = cg_tol
        self.max_cg_iters = max_cg_iters
        self.reg_penalty = reg_penalty
        self.compute_variances = compute_variances

        if self.compute_variances:
            self.bernoulli = Bernoulli(probs=torch.tensor(0.5, device=device))
            self.num_probes = num_probes_for_variances

    @torch.no_grad()
    def reconstruct(self, dataset: MRIDataset, logger: Logger) -> ndarray:
        kspaces = torch.tensor(dataset.kspaces, device=self.device, dtype=self.cdtype)
        sens = torch.tensor(dataset.coil_sens, device=self.device, dtype=self.cdtype)
        kmasks = torch.tensor(dataset.kmasks, device=self.device, dtype=torch.bool)

        fourier = UndersampledFourier2D(mask=kmasks)
        coil_proj = MultiCoilProjection(sens=sens, real_input=True)
        Phi = Sequential(projs=[coil_proj, fourier], fwd_apply=[True, True])

        imgs, variances, cg_iters = self._run_cg(kspaces, Phi)
        imgs = imgs.cpu().numpy()

        logger.log_vals(f"{str(self)}/num_cg_iters", {'num_cg_iters': cg_iters})

        error_maps = np.abs(imgs - dataset.imgs)
        logger.log_imgs(
            f"{str(self)}/ErrorMaps", error_maps, scale_individually=True
        )

        if variances is not None:
            variances = variances.cpu().numpy()
            logger.log_imgs(
                f"{str(self)}/Variances", variances, scale_individually=True
            )

            stds = np.sqrt(variances)
            logger.log_imgs(
                f"{str(self)}/StanDevs", stds, scale_individually=True
            )

        return imgs

    def _run_cg(
        self,
        kspaces: Tensor,
        Phi: Projection
    ) -> Tuple[Tensor, Optional[Tensor], int]:
        sparse_zfill = Phi.T(kspaces).unsqueeze(dim=0)
        _, N, _, H, W = sparse_zfill.size()

        if self.compute_variances:
            probes = self._samp_probes((self.num_probes, N, 1, H, W))
            b = torch.cat([sparse_zfill, probes], dim=0)
        else:
            b = sparse_zfill

        A = lambda x: Phi.T(Phi(x)) + self.reg_penalty * x
        x, converge_iter = conj_grad(
            A, b, dim=(-2, -1), max_iters=self.max_cg_iters, tol=self.cg_tol
        )

        img = x[0].squeeze(dim=1)
        if self.compute_variances:
            variances =  (probes * x[1:]).mean(dim=0).clamp(min=0)
            variances = variances.squeeze(dim=1)
        else:
            variances = None

        cg_iters = self.max_cg_iters if converge_iter is None else converge_iter
        return img, variances, cg_iters

    def _samp_probes(self, size: Tuple[int, ...]):
        return 2 * self.bernoulli.sample(size) - 1

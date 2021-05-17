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


class MultiSENSE(ReconstructionAlgorithm):

    def __init__(
        self,
        sparse_proj: Projection,
        num_em_iters: int = 40,
        num_probes: int = 30,
        cg_tol: float = 1e-5,
        max_cg_iters: int = 1e3,
        alpha0: float = 1e10,
        alpha_init: float = 1.,
        log_imgs_interval: Optional[int] = 5,
        log_rmses: bool = False,
        log_final_variances: bool = False,
        device: Optional[str] = None,
        precision: str = 'double'
    ) -> None:
        # Set device and precision
        self.device = get_torch_device(device)
        dtype, self.cdtype = get_torch_dtypes(precision)
        torch.set_default_dtype(dtype)

        # Create sparse projection object
        self.sparse_proj = instantiate(sparse_proj)

        # Create Bernoulli sampler for probe vectors
        self.bernoulli = Bernoulli(probs=torch.tensor(0.5, device=device))

        # Set algorithm parameters
        self.num_em_iters = num_em_iters
        self.num_probes = num_probes
        self.cg_tol = cg_tol
        self.max_cg_iters = max_cg_iters
        self.alpha0 = alpha0
        self.alpha_init = alpha_init

        # Set logging parameters
        self.log_imgs_interval = log_imgs_interval
        self.log_rmses = log_rmses
        self.log_final_variances = log_final_variances

    @torch.no_grad()
    def reconstruct(self, dataset: MRIDataset, logger: Logger) -> ndarray:
        kspaces = torch.tensor(dataset.kspaces, device=self.device, dtype=self.cdtype)
        sens = torch.tensor(dataset.coil_sens, device=self.device, dtype=self.cdtype)
        kmasks = torch.tensor(dataset.kmasks, device=self.device, dtype=torch.bool)

        fourier = UndersampledFourier2D(mask=kmasks)
        coil_proj = MultiCoilProjection(sens=sens, real_input=True)
        Phi = Sequential(
            projs=[self.sparse_proj, coil_proj, fourier],
            fwd_apply=[False, True, True]
        )

        alpha_init = self.alpha_init * torch.ones(dataset.img_size, device=self.device)
        alpha, mu = self._fastem(kspaces, Phi, alpha_init, dataset, logger)
        imgs = self._compute_imgs(mu)

        error_maps = np.abs(imgs - dataset.imgs)
        logger.log_imgs(f"{str(self)}/ErrorMaps", error_maps)

        if self.log_final_variances:
            variances = self._compute_variances(alpha, Phi, N=kspaces.size(dim=0))
            logger.log_imgs(f"{str(self)}/Variances", variances)

            stds = np.sqrt(variances)
            logger.log_imgs(f"{str(self)}/StanDevs", stds)

        return imgs

    def _fastem(
        self,
        kspaces: Tensor,
        Phi: Projection,
        alpha_init: Tensor,
        dataset: MRIDataset,
        logger: Logger
    ) -> Tuple[Tensor, Tensor]:
        alpha = alpha_init
        sparse_zfill = Phi.T(kspaces)
        mu, sigma_diag, cg_converge_iter = self._estep(alpha, Phi, sparse_zfill)
        self._log_iteration(0, mu, cg_converge_iter, dataset, logger)

        for i in range(self.num_em_iters):
            alpha_new = self._mstep(mu, sigma_diag)
            mu_new, sigma_diag_new, cg_converge_iter = self._estep(
                alpha_new, Phi, sparse_zfill
            )

            if cg_converge_iter is None:
                break
            alpha, mu, sigma_diag = alpha_new, mu_new, sigma_diag_new
            self._log_iteration(i + 1, mu, cg_converge_iter, dataset, logger)

        return alpha, mu

    def _estep(
        self,
        alpha: Tensor,
        Phi: Projection,
        sparse_zfill: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[int]]:
        N, _, H, W = sparse_zfill.size()
        sparse_zfill = sparse_zfill.unsqueeze(dim=0)
        probes = self._samp_probes((self.num_probes, N, 1, H, W))

        b = torch.cat([sparse_zfill, probes], dim=0)
        A = lambda x: Phi.T(Phi(x)) + alpha / self.alpha0 * x
        x, converge_iter = conj_grad(
            A, b, dim=(-2, -1), max_iters=self.max_cg_iters, tol=self.cg_tol
        )

        mu = x[0]
        sigma_diag =  1 / self.alpha0 * (probes * x[1:]).mean(dim=0).clamp(min=0)
        mu = mu.squeeze(dim=1)
        sigma_diag = sigma_diag.squeeze(dim=1)
        return mu, sigma_diag, converge_iter

    def _mstep(self, mu: Tensor, sigma_diag: Tensor):
        alpha = 1 / (mu ** 2 + sigma_diag).mean(dim=0)
        return alpha

    def _log_iteration(
        self,
        _iter: int,
        mu: Tensor,
        cg_converge_iter: int,
        dataset: MRIDataset,
        logger: Logger
    ) -> None:
        logger.log_vals(
            f"{str(self)}/num_cg_iters", {'num_cg_iters': cg_converge_iter}, _iter
        )
        imgs = self._compute_imgs(mu)
        sparsity = np.mean(np.abs(imgs) < 1e-3, axis=(-2, -1))
        logger.log_vals(
            f"{str(self)}/sparsity_level", dict(zip(dataset.names, sparsity)), _iter
        )
        if self.log_rmses:
            metric = RootMeanSquareError(percentage=True)
            rmses = metric(imgs, dataset.imgs)
            combined_metric = RootMeanSquareError(percentage=True, combine=True)
            combined_rmse = combined_metric(imgs, dataset.imgs).item()
            logger.log_vals(
                f"{str(self)}/{str(metric)}", dict(zip(dataset.names, rmses)), _iter
            )
            logger.log_vals(
                f"{str(self)}/{str(metric)}", {'Combined': combined_rmse}, _iter
            )
        if self.log_imgs_interval is not None and _iter % self.log_imgs_interval == 0:
            logger.log_imgs(f"{str(self)}/Reconstruction", imgs, _iter)
            logger.log_imgs(f"{str(self)}/SparseTransform", mu.cpu().numpy(), _iter)

    def _samp_probes(self, size: Tuple[int, ...]):
        return 2 * self.bernoulli.sample(size) - 1

    def _compute_imgs(self, mu: Tensor) -> ndarray:
        imgs = self.sparse_proj.T(mu).cpu().numpy()
        return imgs

    def _compute_variances(self, alpha: Tensor, Phi: Projection, N: int) -> ndarray:
        H, W = alpha.size()
        probes = self._samp_probes((self.num_probes, N, 1, H, W))
        b = self.sparse_proj(probes)
        A = lambda x: Phi.T(Phi(x)) + alpha / self.alpha0 * x
        x, _ = conj_grad(
            A, b, dim=(-2, -1), max_iters=self.max_cg_iters, tol=self.cg_tol
        )
        x = self.sparse_proj.T(x)
        variances =  1 / self.alpha0 * (probes * x).mean(dim=0).clamp(min=0)
        variances = variances.squeeze(dim=1).cpu().numpy()
        return variances

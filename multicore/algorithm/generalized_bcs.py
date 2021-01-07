from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from numpy import ndarray
from numpy.fft import fftshift
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Bernoulli
from torch.fft import fftn, ifftn
import matplotlib.pyplot as plt
from scipy.io import savemat

import pdb
import time

from hydra.utils import instantiate

from multicore.dataset import MRIDataset
from multicore.logger import Logger
from multicore.algorithm.algorithm import ReconstructionAlgorithm
from multicore.projections import (
    Projection,
    UndersampledFourier2D,
    Sequential
)
from multicore.utils import conjugate_gradient
from multicore.metric import RootMeanSquareError


class GeneralizedBCS(ReconstructionAlgorithm):

    def __init__(
        self,
        sparse_proj: Projection,
        num_em_iters: int = 40,
        max_alpha_diff: Optional[float] = None,
        max_alpha_ratio: float = 1.,
        num_probes: int = 30,
        num_init_cg_iters: int = 32,
        cg_tol: float = 1e-10,
        alpha0: float = 1e10,
        alpha_init: float = 1.,
        log_imgs_interval: Optional[int] = 1,
        log_variances: bool = False,
        log_rmses: bool = True,
        complex_imgs: bool = False,
        normalize: bool = True,
        save_img: bool = False,
        device: Optional[str] = None,
        dtype: str = 'float'
    ) -> None:
        if dtype == 'float':
            self.dtype = torch.float
            self.cdtype = torch.cfloat
        elif dtype == 'double':
            self.dtype = torch.double
            self.cdtype = torch.cdouble
        else:
            raise ValueError()
        torch.set_default_dtype(self.dtype)

        self.sparse_proj = instantiate(sparse_proj)
        self.num_em_iters = num_em_iters
        if max_alpha_diff is None:
            max_alpha_diff = float('inf')
        self.max_alpha_ratio = max_alpha_ratio
        self.max_alpha_diff = max_alpha_diff
        self.num_probes = num_probes
        self.num_init_cg_iters = num_init_cg_iters
        self.cg_tol = cg_tol
        self.alpha0 = alpha0
        self.alpha_init = alpha_init
        self.log_imgs_interval = log_imgs_interval
        self.log_variances = log_variances
        self.log_rmses = log_rmses
        if complex_imgs:
            raise NotImplementedError('Current code does not support complex images.')
        self.complex_imgs = complex_imgs

        self.normalize = normalize
        self.save_img = save_img

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.bernoulli = Bernoulli(probs=torch.tensor(0.5, device=device))

    @torch.no_grad()
    def reconstruct(self, dataset: MRIDataset, logger: Logger) -> List[ndarray]:

        start_time = time.time()

        kspaces = torch.tensor(dataset.kspaces, device=self.device, dtype=self.cdtype)
        kmasks = torch.tensor(dataset.kmasks, device=self.device, dtype=torch.bool)

        print('Num observed', torch.sum(kspaces == 0))

        if self.normalize:
            kspaces, bias, scale = self._normalize_kspaces(kspaces)
        else:
            bias = 0
            scale = 1

        fourier = UndersampledFourier2D(mask=kmasks, apply_grad=True)
        # Phi = fourier
        Phi = Sequential(projs=[self.sparse_proj, fourier], fwd_apply=[False, True])
        # Phi = Sequential(projs=[self.sparse_proj, fourier], fwd_apply=[True, True])
        # Phi = fourier

        C, H, W = kspaces.size()
        alpha_init = self.alpha_init * torch.ones(H, W, device=self.device)

        alpha, mu, sigma_diag, imgs = self._fastem(
            kspaces, Phi, alpha_init, kspaces, dataset, logger, bias, scale
        )

        end_time = time.time()
        print(f'Total Bayesian CS Time: {end_time - start_time}')

        if self.save_img:
            _pred_imgs = imgs.transpose(1, 2, 0)
            _kmasks = np.array(dataset.kmasks).transpose(1, 2, 0)
            savemat('bcs.mat', {'pred_img': _pred_imgs, 'kmasks': _kmasks})

        return imgs

    def _normalize_kspaces(
        self,
        kspaces: Tensor
    ) -> Tuple[Tensor, Union[complex, float], float]:
        zero_fill = ifftn(kspaces, dim=(-2, -1), norm='ortho')

        if self.complex_imgs:
            bias = zero_fill.real.min().item() + 1j * zero_fill.imag.min().item()
            max_val = max(zero_fill.real.max(), zero_fill.imag.max())
            scale = max_val - np.min([bias.real, bias.imag])
        else:
            zero_fill = zero_fill.real
            bias = zero_fill.min().item()
            max_val = zero_fill.max().item()
            scale = max_val - bias

        _, H, W = kspaces.size()
        kspaces[:, 0, 0] -= (bias * np.sqrt(H * W))
        kspaces /= scale
        return kspaces, bias, scale


    def _fastem(
        self,
        y: Tensor,
        Phi: Projection,
        alpha_init: Tensor,
        kspaces: Tensor,
        dataset: MRIDataset,
        logger: Logger,
        bias: Union[complex, float] = 0.,
        scale: float = 1.
    ) -> Tuple[Tensor, Tensor, Tensor, List[ndarray]]:
        C, H, W = kspaces.size()

        alpha = alpha_init
        # alpha[:64, :64] /= 10
        # alpha[:32, :32] /= 10
        # imgs_wave = self.sparse_proj(torch.tensor(dataset.imgs).unsqueeze(dim=0))
        # im = imgs_wave[0, 0]
        # threshold = torch.quantile(torch.abs(im), 0.5)
        # wave_mask = (torch.abs(im) < threshold)
        # alpha[wave_mask] *= 100

        mu = torch.zeros(size=(C, H, W), device=self.device, dtype=self.dtype)
        sigma_diag = torch.zeros_like(mu)

        num_cg_iters = self.num_init_cg_iters

        for i in range(self.num_em_iters):
            t = time.time()

            alpha_diff = float('inf')

            mu_new, sigma_diag_new, converged = self._estep(alpha, y, Phi, num_cg_iters)
            print('Sparsity', torch.sum(torch.abs(mu_new) < 1e-5) / mu_new.numel())
            if not converged:
                break
            alpha_new = self._mstep(mu_new, sigma_diag_new)
            alpha_diff = (torch.abs(alpha_new - alpha).mean()).item()
            alpha_ratio = alpha_diff / (torch.abs(alpha).mean()).item()

            alpha = alpha_new
            mu = mu_new
            sigma_diag = sigma_diag_new

            logger.log_vals(f"{str(self)}/alpha_diff", {'alpha_diff': alpha_diff}, i)
            logger.log_vals(f"{str(self)}/alpha_ratio", {'alpha_ratio': alpha_ratio}, i)
            logger.log_vals(
                f"{str(self)}/num_cg_iters", {'num_cg_iters': num_cg_iters}, i
            )

            logger.log_imgs(
                f"{str(self)}/Sparsity", mu, i
            )

            imgs = self._compute_imgs(mu, bias, scale)
            imgs = imgs.cpu().numpy()

            if self.log_rmses:
                metric = RootMeanSquareError(percentage=True)
                rmses = metric(imgs, dataset.imgs)
                combined_metric = RootMeanSquareError(percentage=True, combine=True)
                combined_rmse = combined_metric(imgs, dataset.imgs).item()
                logger.log_vals(
                    f"{str(self)}/{str(metric)}", dict(zip(dataset.names, rmses)), i
                )
                logger.log_vals(
                    f"{str(self)}/{str(metric)}", {'Combined': combined_rmse}, i
                )
            if self.log_imgs_interval is not None and i % self.log_imgs_interval == 0:
                logger.log_imgs(
                    f"{str(self)}/Reconstruction", imgs, i
                )
            print(i, time.time() - t, combined_rmse)
            print('Wave error', metric(self.sparse_proj(torch.tensor(dataset.imgs).unsqueeze(dim=0)).squeeze(dim=0).numpy(), mu.numpy()))
            # pdb.set_trace()

        return alpha, mu, sigma_diag, imgs

    def _estep(
        self,
        alpha: Tensor,
        y: Tensor,
        Phi: Projection,
        num_cg_iters: int
    ) -> Tuple[Tensor, Tensor, bool]:
        C, H, W = y.size()
        b0 = self.alpha0 * Phi.T(y.unsqueeze(dim=0))
        # b0 = self.alpha0 * self.sparse_proj(ifftn(y.unsqueeze(dim=0), dim=(-2, -1), norm='ortho').real)
        # b0 = self.alpha0 * self.sparse_proj.T(ifftn(y.unsqueeze(dim=0), dim=(-2, -1), norm='ortho').real)
        # b0 = self.alpha0 * ifftn(y.unsqueeze(dim=0), dim=(-2, -1), norm='ortho').real
        # b1 = self._samp_probes(self.num_probes, C, H, W)

        alpha = alpha.unsqueeze(dim=0).unsqueeze(dim=0)

        # b0 = Phi.T(y.unsqueeze(dim=0))
        b1 = self._samp_probes(self.num_probes, C, H, W)
        b2 = alpha * b0
        b = torch.cat([b0, b1, b2], dim=0)
        # A = lambda x: (Phi.T(Phi(x))) + alpha * x

        # d = 1 / (self.diag_estimate.unsqueeze(dim=0) + alpha)
        # print('Diag size', d.size())
        A = lambda x: (self.alpha0 * (Phi.T(Phi(x))) + alpha * x)
        # b = d * b
        #
        # Phi_sub = Sequential(
        #     [self.sparse_proj, UndersampledFourier2D(mask=Phi.projs[1].mask[0, 0])],
        #     [False, True]
        # )
        # A_sub = lambda x: d[0, 0, 0] * (self.alpha0 * (Phi_sub.T(Phi_sub(x))) + alpha.squeeze(dim=0).squeeze(dim=1) * x)
        # X = torch.eye(128 * 128).view(128 * 128, 128, 128)
        # matrix = A_sub(X.unsqueeze(dim=0))
        # matrix = matrix.view(128 * 128, 128 * 128)
        # cond_num = torch.lobpcg(matrix, k=1, largest=True)[0] / torch.lobpcg(matrix, k=1, largest=False)[0]
        # print('Condition Number: ', cond_num)

        def stop_criterion(x):
            # mu_new = x[0]
            # sigma_diag_new = (b1 * x[1:]).mean(dim=0).clamp(min=0)
            # alpha_new = self._mstep(mu_new, sigma_diag_new)
            # alpha_diff = torch.abs(alpha_new - alpha).mean().item()
            # alpha_ratio = alpha_diff / (torch.abs(alpha).mean()).item()
            # print(mu_new.mean(), sigma_diag_new.mean())
            # print(alpha_ratio)
            resid = torch.abs(b - A(x)).max()
            print('resid', resid)
            # return alpha_ratio < self.max_alpha_ratio
            return resid < 1

        x, converged = conjugate_gradient(
            A, b, (-2, -1), num_cg_iters, self.cg_tol, stop_criterion=stop_criterion
        )
        # x = b / (self.alpha0 + alpha)
        # converged = True

        mu = x[0]
        sigma_diag = (b1 * x[1:-1]).mean(dim=0).clamp(min=0)
        # self.factor = (b0[0] * b2[0]).sum(dim=(-2, -1), keepdim=True) - (x[-1] * b2[0]).sum(dim=(-2, -1), keepdim=True)
        # sigma_diag = 1 / (self.alpha0 + alpha.squeeze(dim=0))
        # self.alpha0 = ((alpha.squeeze(dim=0) * sigma_diag).sum()) / (torch.abs(y - Phi(mu.unsqueeze(dim=0)).squeeze(dim=0)) ** 2).sum()
        # print('alpha0', self.alpha0)
        return mu, sigma_diag, converged

    def _mstep(self, mu: Tensor, sigma_diag: Tensor):
        alpha = 1 / (mu ** 2 + sigma_diag).mean(dim=0)
        # factor = (0.25 * 128 * 128 + 2 * 1e5) / (self.factor + 2 * 1)
        # alpha = 1 / (mu ** 2 * factor + sigma_diag).mean(dim=0)
        return alpha

    def _samp_probes(self, *size: int):
        return 2 * self.bernoulli.sample(size) - 1

    def _compute_imgs(
        self,
        mu: Tensor,
        bias: Union[complex, float],
        scale: float
    ) -> Tensor:
        # img = self.sparse_proj.T(mu.unsqueeze(dim=0)).squeeze(dim=0)
        img = torch.cumsum(mu, dim=-2)
        # img = self.sparse_proj(mu.unsqueeze(dim=0)).squeeze(dim=0)
        # img = mu

        # if self.normalize:
        #     img = img * scale + bias
        #
        # if not self.complex_imgs and torch.is_complex(img):
        #     img = img.real

        return img

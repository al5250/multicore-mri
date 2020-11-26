from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

from numpy import ndarray
from numpy.fft import fftshift
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Bernoulli
from torch.fft import fftn, ifftn

import pdb
import time

from reconstruction.dataset import MRIDataset
from reconstruction.logger import Logger
from reconstruction.algorithm.algorithm import ReconstructionAlgorithm
from reconstruction.projections import (
    Projection,
    Undersampled2DFastFourierTransform,
    GradientTransform
)
from reconstruction.utils import conjugate_gradient
from reconstruction.metric import RootMeanSquareError


class BayesianCompressedSensing(ReconstructionAlgorithm):

    def __init__(
        self,
        grad_dim: str = 'y',
        max_iters: int = 40,
        max_alpha_diff: float = 1e6,
        num_probes: int = 30,
        max_cg_iters: int = 32,
        cg_tol: float = 1e-10,
        alpha0: float = 3e5,
        log_imgs_interval: Optional[int] = 5,
        log_rmses: bool = False,
        device: Optional[str] = None
    ) -> None:
        if grad_dim not in ['x', 'y', 'xy']:
            raise ValueError('Argument `grad_dim` must be one of ["x", "y", "xy"].')
        self.grad_dim = grad_dim
        self.num_grads = len(grad_dim)

        self.max_iters = max_iters
        self.max_alpha_diff = max_alpha_diff
        self.num_probes = num_probes
        self.max_cg_iters = max_cg_iters
        self.cg_tol = cg_tol
        self.alpha0 = alpha0
        self.log_imgs_interval = log_imgs_interval
        self.log_rmses = log_rmses
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.bernoulli = Bernoulli(probs=torch.tensor(0.5, device=device))

    @torch.no_grad()
    def reconstruct(self, dataset: MRIDataset, logger: Logger) -> List[ndarray]:
        kspaces = torch.tensor(dataset.kspaces, device=self.device)
        self.kspaces = kspaces
        kmasks = torch.tensor(dataset.kmasks, device=self.device)

        size_x, size_y = dataset.img_size

        data_lst = []
        if 'x' in self.grad_dim:
            k = torch.arange(size_x, device=self.device).view(1, -1, 1)
            data_x = (1 - torch.exp(-2 * np.pi * 1j * k / size_x)) * kspaces
            data_lst.append(data_x)

        if 'y' in self.grad_dim:
            k = torch.arange(size_y, device=self.device).view(1, 1, -1)
            data_y = (1 - torch.exp(-2 * np.pi * 1j * k / size_y)) * kspaces
            data_lst.append(data_y)

        data = torch.stack(data_lst, dim=0)
        data = data.flatten(start_dim=-2)

        kmasks = kmasks.flatten(start_dim=-2)
        indices = torch.cat([torch.nonzero(m).view(1, -1) for m in kmasks], dim=0)
        indices = indices.expand((self.num_grads, -1, indices.size(-1)))

        data = torch.gather(data, dim=-1, index=indices)
        Phi = Undersampled2DFastFourierTransform(index=indices, size=(size_x, size_y))
        alpha_init = torch.ones(self.num_grads, size_x * size_y, device=self.device)

        alpha, mu, sigma_diag = self._fastem(data, Phi, alpha_init, logger, dataset)

        imgs = self._grad_to_img(mu, size_x, size_y)
        imgs = imgs.clamp(0, 1)

        return imgs.cpu().numpy()

    def _fastem(
        self,
        y: Tensor,
        Phi: Projection,
        alpha_init: Tensor,
        logger: Logger,
        dataset: MRIDataset
    ) -> Tuple[Tensor, Tensor, Tensor]:
        size_x, size_y = dataset.img_size
        num_contrasts = y.size(dim=1)

        alpha = alpha_init
        mu = torch.zeros(
            size=(self.num_grads, num_contrasts, size_x * size_y),
            device=alpha.device,
            dtype=alpha.dtype
        )
        sigma_diag = torch.zeros_like(mu)

        for i in range(self.max_iters):
            t = time.time()
            mu_new, sigma_diag_new = self._estep(alpha, y, Phi)
            alpha_new = self._mstep(mu_new, sigma_diag_new)

            alpha_diff = torch.abs(alpha_new - alpha).mean().item()
            if alpha_diff > self.max_alpha_diff:
                print('Got Here')
                self.max_cg_iters *= 2
                continue
            else:
                alpha = alpha_new
                mu = mu_new
                sigma_diag = sigma_diag_new

            logger.log_vals(f"{str(self)}/alpha_diff", {'alpha_diff': alpha_diff}, i)
            logger.log_vals(f"{str(self)}/max_cg_iters", {'max_cg_iters': self.max_cg_iters}, i)

            imgs = self._grad_to_img(mu, size_x, size_y)
            imgs = imgs.clamp(0, 1).cpu().numpy()

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
            print(time.time() - t, combined_rmse)

        return alpha, mu, sigma_diag

    def _estep(
        self,
        alpha: Tensor,
        y: Tensor,
        Phi: Projection,
    ) -> Tuple[Tensor, Tensor]:
        b0 = self.alpha0 * Phi.T(y.unsqueeze(dim=0))
        num_contrasts = y.size(dim=1)
        img_size = alpha.size(dim=-1)
        b1 = self._samp_probes(
            (self.num_probes, self.num_grads, num_contrasts, img_size)
        )
        b = torch.cat([b0, b1], dim=0)
        alpha = alpha.unsqueeze(dim=1)
        A = lambda x: self.alpha0 * (Phi.T(Phi(x))) + alpha * x
        x = conjugate_gradient(A, b, -1, self.max_cg_iters, self.cg_tol)
        mu = x[0]
        sigma_diag = (b1 * x[1:]).mean(dim=0).clamp(min=0)
        return mu, sigma_diag

    def _mstep(self, mu: Tensor, sigma_diag: Tensor):
        alpha = 1 / (mu ** 2 + sigma_diag).mean(dim=1)
        return alpha

    def _samp_probes(self, size: Tuple[int, ...]):
        return 2 * self.bernoulli.sample(size) - 1

    def _grad_to_img(self, grad: Tensor, size_x: int, size_y: int) -> Tensor:
        grad = grad.view((self.num_grads, -1, size_x, size_y))
        if self.num_grads == 1:
            # grad = grad.squeeze(dim=0)
            # grad_dim = -1 if self.grad_dim == 'y' else -2
            # size = list(grad.size())
            # N = size[grad_dim]
            # size[grad_dim] = 1
            # x1 = torch.zeros(size, device=grad.device)
            # x2 = grad.narrow(grad_dim, 1, N-1).cumsum(dim=grad_dim)
            # x = torch.cat([x1, x2], dim=grad_dim)
            # return x

            grad_y = grad.squeeze(dim=0)

            ky = torch.arange(size_y, device=self.device).view(1, 1, -1)
            kfactor_y = (1 - torch.exp(-2 * np.pi * 1j * ky / size_y))

            grad_y_fft = fftn(grad_y, dim=(-2, -1), norm='ortho')

            img_fft = torch.conj(kfactor_y) * grad_y_fft

            corr = torch.zeros((1, size_x, size_y), device=self.device)
            corr[0, :, 0] = 1
            norm = (torch.abs(kfactor_y) ** 2) + corr

            img_fft = img_fft / norm * (self.kspaces == 0) + self.kspaces

            img = ifftn(img_fft, dim=(-2, -1), norm='ortho').real

            return img

        elif self.num_grads == 2:
            grad_x, grad_y = grad

            kx = torch.arange(size_x, device=self.device).view(1, -1, 1)
            kfactor_x = (1 - torch.exp(-2 * np.pi * 1j * kx / size_x))

            ky = torch.arange(size_y, device=self.device).view(1, 1, -1)
            kfactor_y = (1 - torch.exp(-2 * np.pi * 1j * ky / size_y))

            grad_x_fft = fftn(grad_x, dim=(-2, -1), norm='ortho')
            grad_y_fft = fftn(grad_y, dim=(-2, -1), norm='ortho')

            img_fft = torch.conj(kfactor_x) * grad_x_fft + torch.conj(kfactor_y) * grad_y_fft

            corr = torch.zeros((1, size_x, size_y), device=self.device)
            corr[0, 0, 0] = 1
            norm = (torch.abs(kfactor_x) ** 2 + torch.abs(kfactor_y) ** 2) + corr

            img_fft = img_fft / norm * (self.kspaces == 0) + self.kspaces

            img = ifftn(img_fft, dim=(-2, -1), norm='ortho').real

            return img

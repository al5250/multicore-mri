from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

from numpy import ndarray
from numpy.fft import fftshift
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Bernoulli
from torch.fft import fftn, ifftn
import matplotlib.pyplot as plt

from scipy.sparse.linalg import LinearOperator
from scipy.io import savemat

import pdb
import time

from multicore.dataset import MRIDataset
from multicore.logger import Logger
from multicore.algorithm.algorithm import ReconstructionAlgorithm
from multicore.projections import (
    Projection,
    Undersampled2DFastFourierTransform,
    GradientTransform
)
from multicore.utils import conjugate_gradient
from multicore.metric import RootMeanSquareError
from multicore.algorithm import BayesianCompressedSensing


import pdb


class BayesianCompressedSensingMultiCoil(BayesianCompressedSensing):

    def __init__(self, save_path: Optional[str] = None, **kwargs):
        self.save_path = save_path
        super().__init__(**kwargs)

    @torch.no_grad()
    def reconstruct(self, dataset: MRIDataset, logger: Logger) -> List[ndarray]:
        out = super().reconstruct(dataset, logger)
        if self.save_path is not None:
            savemat(self.save_path, {'recon': self.recon_imgs.transpose((1, 2, 0))})
        return out

    def _compute_imgs(
        self,
        grad: Tensor,
        kspaces: Tensor
    ) -> Tensor:

        imgs = super()._compute_imgs(grad, kspaces)

        self.recon_imgs = imgs.cpu().numpy()

        self.coils = torch.tensor(self.dataset.coils, device=self.device)
        img = torch.sum(torch.conj(self.coils) * imgs, dim=0) / torch.sum(torch.abs(self.coils) ** 2, dim=0)
        img = img.unsqueeze(dim=0).real

        return img


class SenseOperator(LinearOperator):

    def __init__(self, coils: ndarray, **kwargs):
        n_coils, size_x, size_y = coils.shape
        in_dim = size_x * size_y
        out_dim = n_coils * size_x * size_y

        def forward(x):
            x = coils * x.reshape(size_x, size_y).expand_dims(axis=0)
            x = np.fft.fft2(x, norm='ortho').reshape(out_dim)
            return x
        super().__init__(shape=(out_dim, in_dim))


class BayesianCompressedSensingMultiCoilForward(BayesianCompressedSensing):

    @torch.no_grad()
    def reconstruct(self, dataset: MRIDataset, logger: Logger) -> List[ndarray]:
        self.coils = torch.tensor(dataset.coils, device=self.device).flatten(start_dim=-2).unsqueeze(dim=0).unsqueeze(dim=0)
        start_time = time.time()

        kspaces = torch.tensor(dataset.kspaces, device=self.device)
        #
        # if self.complex_imgs:
        #     zero_fill = ifftn(kspaces, dim=(-2, -1), norm='ortho').cpu().numpy()
        #     self.bias = (np.real(zero_fill).min() + 1j * np.imag(zero_fill).min())
        #     max_val = np.max([np.real(zero_fill).max(), np.imag(zero_fill).max()])
        #     self.scale = max_val - np.min([self.bias.real, self.bias.imag])
        # else:
        #     zero_fill = ifftn(kspaces, dim=(-2, -1), norm='ortho').cpu().numpy().real
        #     self.bias = np.real(zero_fill).min()
        #     max_val = np.real(zero_fill).max()
        #     self.scale = max_val - self.bias
        #
        # for i in range(kspaces.size(dim=0)):
        #     kspaces[i, 0, 0] -= self.bias * kspaces.size(dim=1)

        # kspaces /= self.scale

        self.kspaces = kspaces
        self.dataset = dataset
        kmasks = torch.tensor(dataset.kmasks, device=self.device)
        self.kmasks = kmasks
        # pdb.set_trace()
        num_contrasts, size_x, size_y = kspaces.size()

        # Unsqueeze grad dim
        kspaces = kspaces.unsqueeze(dim=0)

        flat_kmasks = kmasks.flatten(start_dim=-2)
        kindices = torch.cat([torch.nonzero(m).view(1, -1) for m in flat_kmasks], dim=0)
        kindices = kindices.unsqueeze(dim=0)

        if self.complex_imgs:
            kspaces_sym = torch.conj(kspaces)
            kspaces_sym = torch.flip(kspaces_sym, dims=(-2, -1))
            kspaces_sym = torch.roll(kspaces_sym, shifts=(1, 1), dims=(-2, -1))
            kspaces_real = 0.5 * (kspaces + kspaces_sym)
            kspaces_imag = -0.5j * (kspaces - kspaces_sym)

            if self.tie_real_imag:
                # Split real/imaginary channels as extra contrasts (same sparsity)
                kspaces = torch.cat([kspaces_real, kspaces_imag], dim=1)
                kindices = torch.cat([kindices, kindices], dim=1)
                num_contrasts *= 2
            else:
                # Split real/imaginary channels as extra gradients (different sparsity)
                kspaces = torch.cat([kspaces_real, kspaces_imag], dim=0)
                kindices = torch.cat([kindices, kindices], dim=0)

        data_lst = []
        if 'x' in self.grad_dim:
            k = torch.arange(size_x, device=self.device).view(1, 1, -1, 1)
            data_x = (1 - torch.exp(-2 * np.pi * 1j * k / size_x)) * kspaces
            data_lst.append(data_x)
        if 'y' in self.grad_dim:
            k = torch.arange(size_y, device=self.device).view(1, 1, 1, -1)
            data_y = (1 - torch.exp(-2 * np.pi * 1j * k / size_y)) * kspaces
            data_lst.append(data_y)
        data = torch.cat(data_lst, dim=0)
        data = data.flatten(start_dim=-2)

        if self.grad_dim == 'xy':
            kindices = torch.cat([kindices, kindices], dim=0)

        data = torch.gather(data, dim=-1, index=kindices)
        Phi = Undersampled2DFastFourierTransform(index=kindices, size=(size_x, size_y))
        alpha_init = self.alpha_init * torch.ones(self.num_grads, size_x * size_y, device=self.device)

        alpha, mu, sigma_diag, imgs = self._fastem(
            data, Phi, alpha_init, kspaces, dataset, logger
        )

        if self.log_variances:
            variances = self._compute_variances(
                Phi, alpha, kmasks, self.num_probes, num_cg_iters=256
            )
            variances = variances.cpu().numpy()
            variances = np.array([v / v.max() for v in variances])

            logger.log_imgs(
                f"{str(self)}/Variances", variances
            )

        end_time = time.time()
        print(f'Total Bayesian CS Time: {end_time - start_time}')

        return imgs

    def _estep(
        self,
        alpha: Tensor,
        y: Tensor,
        Phi: Projection,
        num_cg_iters: int
    ) -> Tuple[Tensor, Tensor, bool]:
        Phi.real_input = False
        b0 = self.alpha0 * (torch.conj(self.coils) * Phi.T(y.unsqueeze(dim=0))).real
        num_contrasts = y.size(dim=1)
        img_size = alpha.size(dim=-1)
        b1 = self._samp_probes(
            (self.num_probes, self.num_grads, num_contrasts, img_size)
        )
        b = torch.cat([b0, b1], dim=0)
        alpha = alpha.unsqueeze(dim=1)
        A = lambda x: self.alpha0 * (torch.conj(self.coils) * (Phi.T(Phi(self.coils * x)))).real + alpha * x

        def stop_criterion(x):
            mu_new = x[0]
            sigma_diag_new = (b1 * x[1:]).mean(dim=0).clamp(min=0)
            alpha_new = self._mstep(mu_new, sigma_diag_new)
            alpha_diff = torch.abs(alpha_new - alpha).mean().item()
            alpha_ratio = alpha_diff / (torch.abs(alpha).mean()).item()
            return alpha_ratio < self.max_alpha_ratio

        x, converged = conjugate_gradient(A, b, -1, num_cg_iters, self.cg_tol, self.x, stop_criterion=stop_criterion)
        mu = x[0]
        sigma_diag = (b1 * x[1:]).mean(dim=0).clamp(min=0)
        return mu, sigma_diag, converged

    def _compute_imgs(
        self,
        grad: Tensor,
        kspaces: Tensor
    ) -> Tensor:


        _, size_x, size_y = self.kspaces.size()
        num_grads, num_contrasts, _ = grad.size()

        grad = grad.view((self.num_grads, num_contrasts, size_x, size_y)).squeeze(dim=0)
        img = torch.cumsum(grad, dim=2).mean(dim=0).unsqueeze(dim=0)

        data_imgs = np.array(self.dataset.imgs)
        diff = data_imgs - np.roll(data_imgs, shift=1, axis=2)
        print(np.abs(diff - grad.numpy()).mean() / np.abs(diff).mean())
        pdb.set_trace()

        # img = img * self.scale + self.bias

        return img

        # coils = self.coils.squeeze(dim=0).squeeze(dim=0).view(-1, size_x, size_y)
        #
        # num_grads, num_contrasts, _ = grad.size()
        # grad = grad.view((self.num_grads, num_contrasts, size_x, size_y))
        # grad_old = grad
        #
        # if self.complex_imgs and self.tie_real_imag:
        #     num_contrasts //= 2
        #     grad = grad[:, :num_contrasts] + 1j * grad[:, num_contrasts:]
        #     kspaces = kspaces[:num_contrasts]
        # elif self.complex_imgs and not self.tie_real_imag:
        #     num_grads = self.num_grads // 2
        #     grad = grad[:num_grads] + 1j * grad[num_grads:]
        #
        # img_fft = torch.tensor(0., device=self.device)
        # norm = torch.tensor(0., device=self.device)
        #
        # if 'x' in self.grad_dim:
        #     kx = torch.arange(size_x, device=self.device).view(1, -1, 1)
        #     kfactor_x = (1 - torch.exp(-2 * np.pi * 1j * kx / size_x))
        #
        #     grad_x = grad[0]
        #     grad_x_fft = fftn(coils * grad_x, dim=(-2, -1), norm='ortho')
        #
        #     img_fft = img_fft + torch.conj(kfactor_x) * grad_x_fft
        #     norm = norm + torch.abs(kfactor_x) ** 2
        #
        # if 'y' in self.grad_dim:
        #     ky = torch.arange(size_y, device=self.device).view(1, 1, -1)
        #     kfactor_y = (1 - torch.exp(-2 * np.pi * 1j * ky / size_y))
        #
        #     grad_y = grad[-1]
        #     grad_y_fft = fftn(coils * grad_y, dim=(-2, -1), norm='ortho')
        #
        #     img_fft = img_fft + torch.conj(kfactor_y) * grad_y_fft
        #     norm = norm + torch.abs(kfactor_y) ** 2
        #
        # corr = torch.zeros((1, size_x, size_y), device=self.device)
        # if self.grad_dim == 'x':
        #     corr[0, 0, :] = 1
        # elif self.grad_dim == 'y':
        #     corr[0, :, 0] = 1
        # elif self.grad_dim == 'xy':
        #     corr[0, 0, 0] = 1
        # norm = norm + corr
        #
        # img_fft = img_fft / norm * (self.kspaces == 0) + self.kspaces
        # img = ifftn(img_fft, dim=(-2, -1), norm='ortho')
        #
        # img = img * self.scale + self.bias
        #
        #
        # img = torch.sum(torch.conj(coils) * img, dim=0) / torch.sum(torch.abs(coils) ** 2, dim=0)
        # img = img.unsqueeze(dim=0).real
        #
        # return img

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
    GradientTransform
)
from multicore.utils import conj_grad
from multicore.metric import RootMeanSquareError


class MulticoilBCS(ReconstructionAlgorithm):

    def __init__(
        self,
        sparse_proj: Projection,
        grad_dim: str = 'y',
        num_em_iters: int = 40,
        max_alpha_diff: Optional[float] = None,
        max_alpha_ratio: float = 1.,
        num_probes: int = 30,
        num_init_cg_iters: int = 32,
        cg_tol: float = 1e-10,
        alpha0: float = 1e10,
        alpha_init: float = 1.,
        log_imgs_interval: Optional[int] = 5,
        log_variances: bool = False,
        log_rmses: bool = False,
        complex_imgs: bool = False,
        tie_real_imag: bool = True,
        normalize: bool = True,
        save_img: bool = False,
        device: Optional[str] = None
    ) -> None:
        if grad_dim not in ['x', 'y', 'xy']:
            raise ValueError('Argument `grad_dim` must be one of ["x", "y", "xy"].')
        self.grad_dim = grad_dim
        self.num_grads = len(grad_dim)

        torch.set_default_dtype(torch.double)
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
        self.complex_imgs = complex_imgs

        self.tie_real_imag = tie_real_imag
        if not self.tie_real_imag:
            self.num_grads *= 2
        self.normalize = normalize
        self.save_img = save_img

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.bernoulli = Bernoulli(probs=torch.tensor(0.5, device=device))

    @torch.no_grad()
    def reconstruct(self, dataset: MRIDataset, logger: Logger) -> List[ndarray]:

        start_time = time.time()

        kspaces = torch.tensor(dataset.kspaces, device=self.device)

        if self.normalize:
            if self.complex_imgs:
                zero_fill = ifftn(kspaces, dim=(-2, -1), norm='ortho').cpu().numpy()
                self.bias = (np.real(zero_fill).min() + 1j * np.imag(zero_fill).min())
                max_val = np.max([np.real(zero_fill).max(), np.imag(zero_fill).max()])
                self.scale = max_val - np.min([self.bias.real, self.bias.imag])
            else:
                zero_fill = ifftn(kspaces, dim=(-2, -1), norm='ortho').cpu().numpy().real
                self.bias = np.real(zero_fill).min()
                max_val = np.real(zero_fill).max()
                self.scale = max_val - self.bias

            for i in range(kspaces.size(dim=0)):
                kspaces[i, 0, 0] -= self.bias * np.sqrt(kspaces.size(dim=1) * kspaces.size(dim=2))

            kspaces /= self.scale

        self.kspaces = kspaces
        self.dataset = dataset
        kmasks = torch.tensor(dataset.kmasks, device=self.device)
        self.kmasks = kmasks
        Phi = UndersampledFourier2D(mask=kmasks, real_input=False)

        N, C, H, W = kspaces.size()

        self.coils = torch.tensor(dataset.coils, device=self.device).unsqueeze(dim=0).unsqueeze(dim=0)

        # Convert data to gradients
        kspaces = kspaces.unsqueeze(dim=0)
        data_lst = []
        if 'x' in self.grad_dim:
            k = torch.arange(H, device=self.device).view(1, 1, 1, -1, 1)
            data_x = kspaces
            # data_x = (1 - torch.exp(-2 * np.pi * 1j * k / H)) * kspaces
            data_lst.append(data_x)
        if 'y' in self.grad_dim:
            k = torch.arange(W, device=self.device).view(1, 1, 1, 1, -1)
            data_y = kspaces
            # data_y = (1 - torch.exp(-2 * np.pi * 1j * k / W)) * kspaces
            data_lst.append(data_y)
        data = torch.cat(data_lst, dim=0)

        alpha_init = self.alpha_init * torch.ones(self.num_grads, 1, 1, H, W, device=self.device)

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

        if self.save_img:
            _pred_imgs = imgs.transpose(1, 2, 0)
            _kmasks = np.array(dataset.kmasks).transpose(1, 2, 0)
            savemat('bcs.mat', {'pred_img': _pred_imgs, 'kmasks': _kmasks})

        return imgs

    def _fastem(
        self,
        y: Tensor,
        Phi: Projection,
        alpha_init: Tensor,
        kspaces: Tensor,
        dataset: MRIDataset,
        logger: Logger
    ) -> Tuple[Tensor, Tensor, Tensor, List[ndarray]]:
        _, N, C, H, W = y.size()
        num_cg_iters = self.num_init_cg_iters
        factor = 1

        alpha = alpha_init
        mu = torch.zeros(
            size=(self.num_grads, N, 1, H, W),
            device=alpha.device,
            dtype=alpha.dtype
        )
        sigma_diag = torch.zeros_like(mu)

        for i in range(self.num_em_iters):
            t = time.time()

            self.x = None
            self.b1 = None
            alpha_diff = float('inf')

            mu_new, sigma_diag_new, converged = self._estep(alpha, y, Phi, num_cg_iters)
            if not converged:
                break
            alpha_new = self._mstep(mu_new, sigma_diag_new)
            pdb.set_trace()
            alpha_diff = (torch.abs(alpha_new - alpha).mean()).item()
            alpha_ratio = alpha_diff / (torch.abs(alpha).mean()).item()

            alpha = alpha_new
            mu = mu_new
            sigma_diag = sigma_diag_new

            # mu_new, sigma_diag_new = self._estep(alpha, y, Phi, num_cg_iters)
            # alpha_new = self._mstep(mu_new, sigma_diag_new)
            #
            # alpha_diff = torch.abs(alpha_new - alpha).mean().item()
            # if alpha_diff > self.max_alpha_diff:
            #     print('Got Here')
            #     num_cg_iters *= 2
            #     continue
            # else:
            #     alpha = alpha_new
            #     mu = mu_new
            #     sigma_diag = sigma_diag_new

            logger.log_vals(f"{str(self)}/alpha_diff", {'alpha_diff': alpha_diff}, i)
            logger.log_vals(f"{str(self)}/alpha_ratio", {'alpha_ratio': alpha_ratio}, i)
            logger.log_vals(
                f"{str(self)}/num_cg_iters", {'num_cg_iters': num_cg_iters}, i
            )

            imgs = self._compute_imgs(mu, kspaces)
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

        return alpha, mu, sigma_diag, imgs

    def _estep(
        self,
        alpha: Tensor,
        y: Tensor,
        Phi: Projection,
        num_cg_iters: int
    ) -> Tuple[Tensor, Tensor, bool]:

        coils = self.coils.unsqueeze(dim=0)

        _, N, C, H, W = y.size()
        # b0 = torch.sum(torch.conj(coils) * Phi.T(y.unsqueeze(dim=0)), dim=-3, keepdim=True).real
        b0 = self.sparse_proj(torch.sum(torch.conj(coils) * Phi.T(y.unsqueeze(dim=0)), dim=-3, keepdim=True).real)
        b1 = self._samp_probes(
            (self.num_probes, self.num_grads, N, 1, H, W)
        )
        # b1 = torch.zeros((
        #     self.num_probes, self.num_grads, N, 1, H, W
        # ))
        b = torch.cat([b0, b1], dim=0)
        # b = b0
        alpha = alpha.unsqueeze(dim=0)

        # A = lambda x: (torch.sum(torch.conj(coils) * Phi.T(Phi(coils * x)), dim=-3, keepdim=True)).real + 1 / self.alpha0 * alpha * x
        A = lambda x: self.sparse_proj(torch.sum(torch.conj(coils) * Phi.T(Phi(coils * self.sparse_proj.T(x))), dim=-3, keepdim=True).real) \
                + 1 / self.alpha0 * alpha * x

        # A = self.alpha0 * torch.sum(torch.conj(coils) * coils, dim=3, keepdim=True).real + alpha
        # A_inv = 1 / A

        def stop_criterion(x):
            # mu_new = x[0]
            # sigma_diag_new = (b1 * x[1:]).mean(dim=0).clamp(min=0)
            # alpha_new = self._mstep(mu_new, sigma_diag_new)
            # alpha_diff = torch.abs(alpha_new - alpha).mean().item()
            # alpha_ratio = alpha_diff / (torch.abs(alpha).mean()).item()
            # resid = torch.abs(b - A(x)).mean()
            # print('resid', resid)
            # return resid < 1e-1
            return False
            # return alpha_ratio < self.max_alpha_ratio
        x, converged = conj_grad(A, b, (-2, -1), num_cg_iters, self.cg_tol, stop_criterion=stop_criterion)
        # x = A_inv * b
        mu = x[0]
        # sigma_diag = mu
        sigma_diag =  1 / self.alpha0 * (b1 * x[1:]).mean(dim=0).clamp(min=0)
        return mu, sigma_diag, converged

    def _mstep(self, mu: Tensor, sigma_diag: Tensor):
        alpha = 1 / (mu ** 2 + sigma_diag).mean(dim=1, keepdim=True)
        return alpha

    def _samp_probes(self, size: Tuple[int, ...]):
        return 2 * self.bernoulli.sample(size) - 1

    def _compute_imgs(
        self,
        grad: Tensor,
        kspaces: Tensor
    ) -> Tensor:
        # grad = grad.squeeze(dim=2).squeeze(dim=0)
        # return grad
        img = self.sparse_proj.T(grad).squeeze(dim=2).squeeze(dim=0)
        return img
        # img = torch.cumsum(grad, dim=-1)
        # img = img.squeeze(dim=2).squeeze(dim=0)
        # return img
        # _, size_x, size_y = self.kspaces.size()
        coils = self.coils.squeeze(dim=0)
        _, N, _, H, W = grad.size()
        grad = self.coils * grad
        kspaces = kspaces.squeeze(dim=0)

        # if self.complex_imgs and self.tie_real_imag:
        #     num_contrasts //= 2
        #     grad = grad[:, :num_contrasts] + 1j * grad[:, num_contrasts:]
        #     kspaces = kspaces[:num_contrasts]
        # elif self.complex_imgs and not self.tie_real_imag:
        #     num_grads = self.num_grads // 2
        #     grad = grad[:num_grads] + 1j * grad[num_grads:]

        img_fft = torch.tensor(0., device=self.device)
        norm = torch.tensor(0., device=self.device)

        if 'x' in self.grad_dim:
            kx = torch.arange(H, device=self.device).view(1, 1, -1, 1)
            kfactor_x = (1 - torch.exp(-2 * np.pi * 1j * kx / H))

            grad_x = grad[0]
            grad_x_fft = fftn(coils * grad_x, dim=(-2, -1), norm='ortho')

            img_fft = img_fft + torch.conj(kfactor_x) * grad_x_fft
            norm = norm + torch.abs(kfactor_x) ** 2

        if 'y' in self.grad_dim:
            ky = torch.arange(W, device=self.device).view(1, 1, 1, -1)
            kfactor_y = (1 - torch.exp(-2 * np.pi * 1j * ky / W))

            grad_y = grad[-1]
            grad_y_fft = fftn(coils * grad_y, dim=(-2, -1), norm='ortho')

            img_fft = img_fft + torch.conj(kfactor_y) * grad_y_fft
            norm = norm + torch.abs(kfactor_y) ** 2

        corr = torch.zeros((1, 1, H, W), device=self.device)
        if self.grad_dim == 'x':
            corr[0, 0, 0, :] = 1
        elif self.grad_dim == 'y':
            corr[0, 0, :, 0] = 1
        elif self.grad_dim == 'xy':
            corr[0, 0, 0, 0] = 1
        norm = norm + corr

        img_fft = img_fft / norm * (kspaces == 0) + kspaces
        img = ifftn(img_fft, dim=(-2, -1), norm='ortho')

        img = torch.sum(torch.conj(coils) * img, dim=1) / torch.sum((torch.conj(coils) * coils), dim=1)
        # img.real = img.real.clamp(min=0, max=1)
        # img.imag = img.imag.clamp(min=0, max=1)

        if self.normalize:
            img = img * self.scale + self.bias

        if not self.complex_imgs:
            img = img.real

        if self.complex_imgs:
            num_contrasts //= 2
            img_real = img[:num_contrasts]
            img_imag = img[num_contrasts:]
            img = img_real + 1j * img_imag

        return img

    def _compute_variances(
        self,
        Phi: Projection,
        alpha: Tensor,
        kmasks: Tensor,
        num_probes: int,
        num_cg_iters: int = 32,
        cg_tol: float = 1e-10
    ) -> Tensor:

        num_contrasts, size_x, size_y = kmasks.size()
        masks = ~kmasks.unsqueeze(dim=0)

        z = self._samp_probes((num_probes, num_contrasts, size_x, size_y))
        b = fftn(z, dim=(-2, -1), norm='ortho')
        b = masks * b

        b_lst = []
        norm = torch.tensor(0., device=self.device)

        if 'x' in self.grad_dim:
            kx = torch.arange(size_x, device=self.device).view(1, 1, -1, 1)
            kfactor_x = (1 - torch.exp(-2 * np.pi * 1j * kx / size_x))
            b_x = kfactor_x * b
            b_lst.append(b_x)
            norm = norm + torch.abs(kfactor_x) ** 2

        if 'y' in self.grad_dim:
            ky = torch.arange(size_y, device=self.device).view(1, 1, 1, -1)
            kfactor_y = (1 - torch.exp(-2 * np.pi * 1j * ky / size_y))
            b_y = kfactor_y * b
            b_lst.append(b_y)
            norm = norm + torch.abs(kfactor_y) ** 2

        corr = torch.zeros((1, 1, size_x, size_y), device=self.device)
        if self.grad_dim == 'x':
            corr[0, 0, 0, :] = 1
        elif self.grad_dim == 'y':
            corr[0, 0, :, 0] = 1
        elif self.grad_dim == 'xy':
            corr[0, 0, 0, 0] = 1
        norm = norm + corr
        b = torch.stack(b_lst, dim=1) / norm.unsqueeze(dim=1)

        b = ifftn(b, dim=(-2, -1), norm='ortho').real
        b = b.flatten(start_dim=-2)

        alpha = alpha.unsqueeze(dim=1).unsqueeze(dim=0)
        A = lambda x: self.alpha0 * (Phi.T(Phi(x))) + alpha * x
        out = conj_grad(A, b, -1, num_cg_iters, cg_tol)

        out = out.unflatten(dim=-1, sizes=(size_x, size_y))
        out = fftn(out, dim=(-2, -1), norm='ortho')

        if 'x' in self.grad_dim:
            out[:, 0] = torch.conj(kfactor_x) * out[:, 0] / norm
        if 'y' in self.grad_dim:
            out[:, -1] = torch.conj(kfactor_y) * out[:, -1] / norm

        if self.grad_dim == 'xy':
            out = out[:, 0] + out[:, -1]
        else:
            out = out.squeeze(dim=1)

        out = masks * out
        out = ifftn(out, dim=(-2, -1), norm='ortho').real

        var = (z * out).mean(dim=0).clamp(min=0)
        return var

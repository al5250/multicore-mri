from abc import abstractmethod
from typing import Dict, List, Optional

from numpy import ndarray
import torch
from torch import Tensor
import pdb

from multicore.dataset import MRIDataset
from multicore.algorithm.algorithm import ReconstructionAlgorithm
from multicore.utils import finite_diff_2d, inv_finite_diff_2d, ufft, iufft


class L1ConjugateGradient(ReconstructionAlgorithm):

    def __init__(
        self,
        sparse_dom: str = 'diff',
        sparse_pen: float = 1e-3,
        tol_grad: float = 1e-4,
        max_iter: int = 100,
        alpha: float = 0.01,
        beta: float = 0.6,
        mu: float = 1e-6,
        device: str = 'cpu'
    ) -> None:
        if sparse_dom not in ['diff', 'wavelet']:
            raise ValueError(
                'Argument `sparse_dom` must be either `diff` or `wavelet.`'
            )
        self.sparse_dom = sparse_dom
        self.sparse_pen = sparse_pen
        self.tol_grad = tol_grad
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.device = device

    def setup(self, dataset: MRIDataset) -> None:
        self.masks = torch.tensor(dataset.kmasks)
        self.y = torch.tensor(dataset.kspaces, dtype=torch.complex64).to(self.device)
        self._imgs = torch.zeros(self.y.size())
        self.n_img = self._imgs.size(0)
        self.loss = self._eval_obj(self._imgs)
        self.grad = self._eval_grad_obj(self._imgs)
        self.grad_norm = torch.norm(self.grad, dim=(1, 2))
        self.delta_imgs = -self.grad
        self._finished = torch.zeros(self.n_img, dtype=bool)

    @torch.no_grad()
    def iterate(self, itr: int) -> Dict[str, float]:
        grad_delta = (self.grad * self.delta_imgs).sum(dim=(1, 2))
        t = 10 * torch.ones(self.n_img) / self.beta
        found_step = torch.zeros(self.n_img, dtype=bool)
        while not found_step.all().item():
            t[~found_step] = t[~found_step] * self.beta
            new_imgs = self._imgs + t.view(-1, 1, 1) * self.delta_imgs
            new_loss = self._eval_obj(new_imgs)
            found_step = (new_loss <= self.loss + self.alpha * t * grad_delta)
        new_grad = self._eval_grad_obj(new_imgs)
        new_grad_norm = torch.norm(new_grad, dim=(1, 2))
        gamma =  ((new_grad_norm / self.grad_norm) ** 2).view(-1, 1, 1)

        self._imgs[~self._finished] = new_imgs[~self._finished]
        self.grad[~self._finished] = new_grad[~self._finished]
        self.grad_norm[~self._finished] = new_grad_norm[~self._finished]
        self.delta_imgs = gamma * self.delta_imgs - new_grad
        self.loss = new_loss
        self._finished |= (self.grad_norm < self.tol_grad)

        print()
        print(f'Iter: {itr}')
        print(f'Grad Delta: {grad_delta}')
        print(f'Loss: {self.loss}')
        print(f'Grad Norm: {self.grad_norm}')
        print(f'Max: {self._imgs.max()}')
        print(f'Min: {self._imgs.min()}')
        print(f'Step Size: {t}')
        print(f'Finished: {self._finished}')
        return {'Loss': self.loss}

    def finished(self, itr: int) -> bool:
        return (itr >= self.max_iter) or self._finished.all().item()

    @property
    def imgs(self) -> List[ndarray]:
        return [x.numpy() for x in self._imgs]

    def _transform(self, m: Tensor) -> Tensor:
        if self.sparse_dom == 'diff':
            dx, dy = finite_diff_2d(m)
            return torch.cat([dx, dy], dim=-1)
        elif self.sparse_dom == 'wavelet':
            raise NotImplementedError()
        else:
            raise ValueError()

    def _inv_transform(self, x: Tensor) -> Tensor:
        if self.sparse_dom == 'diff':
            dim = x.size(-1)
            half_dim = dim // 2
            return inv_finite_diff_2d(x[:, :, :half_dim], x[:, :, half_dim:])
        elif self.sparse_dom == 'wavelet':
            raise NotImplementedError()
        else:
            raise ValueError()

    def _eval_obj(self, m: Tensor) -> Tensor:
        recon_loss = torch.norm(
            (ufft(m, mask=self.masks, signal_ndim=2, normalized=True) - self.y).abs(), dim=(1, 2)
        ) ** 2
        regul_loss = torch.abs(self._transform(m)).sum(dim=(1, 2))
        return (recon_loss + self.sparse_pen * regul_loss)

    def _eval_grad_obj(self, m: Tensor) -> Tensor:
        recon_grad = 2 * iufft(
            ufft(m, mask=self.masks, signal_ndim=2, normalized=True) - self.y,
            signal_ndim=2,
            normalized=True
        )
        Tm = self._transform(m)
        w = torch.sqrt(self.mu + (Tm ** 2))
        regul_grad = self._inv_transform(1 / w * Tm)
        return (recon_grad + self.sparse_pen * regul_grad)

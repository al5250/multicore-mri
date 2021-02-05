from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from numpy import ndarray
from numpy.fft import fft2, ifftshift
from scipy.io import loadmat
import torch
from torch import Tensor
import pdb

from multicore.dataset.dataset import MRIDataset
from multicore.ksampler import KSampler, PowerRuleSampler


class SheppLoganMultiCoilDataset(MRIDataset):

    def __init__(self, file_path: str, coil_path: str, **kwargs):
        self._ksampler = PowerRuleSampler(**kwargs)

        sim8 = loadmat(coil_path)
        self.coils = (sim8['b1'] / np.abs(sim8['b1']).max()).transpose(2, 0, 1)
        # self.coils = np.ones((8, 128, 128)) + 0j

        data = loadmat(file_path)
        # self._imgs = data['img'].transpose(2, 0, 1)[0]
        # self._imgs = sim8['anatomy_orig'] / np.abs(sim8['anatomy_orig']).max()
        # self._imgs = np.expand_dims(self._imgs, axis=0)

        self._imgs = data['img'].transpose(2, 0, 1)

        coil_imgs = np.expand_dims(self._imgs, axis=1) * np.expand_dims(self.coils, axis=0)

        self._kspaces_full = fft2(coil_imgs, norm='ortho', axes=(-2, -1))
        #
        mask = np.zeros((128, 128), dtype=bool).T
        # mask[56:72, :] = True
        mask1 = mask.copy()
        mask2 = mask.copy()
        mask3 = mask.copy()

        mask1[0::3, :] = True
        mask2[1::3, :] = True
        mask3[2::3, :] = True
        mask1 = ifftshift(mask1)
        mask2 = ifftshift(mask2)
        mask3 = ifftshift(mask3)

        pdb.set_trace()

        kmasks = np.stack([mask1, mask2, mask3], axis=0)
        self._kmasks = np.stack([kmasks] * 8, axis=1)
        # mask[:, 0:16] = True
        # mask[:, -16:] = True
        # self._kmasks = np.stack([mask] * 8, axis=0)
        # self._kmasks = mask

        noise_std = 1e-3
        noise = noise_std * np.random.normal(size=self._kspaces_full.shape) + noise_std * 1j * np.random.normal(size=self._kspaces_full.shape)
        self._kspaces = self._kspaces_full.copy() + noise
        self._kspaces[..., ~self._kmasks] = 0

        # _kspaces = []
        # for i in range(len(self._kspaces_full[0])):
        #     noise = 1e-3 * np.random.normal(size=self._ksampler(self._kspaces_full[0][i:i+1])[0].shape)
        #     _kspaces.append(self._ksampler(self._kspaces_full[0][i:i+1])[0] + noise)
        # self._kspaces = np.array(_kspaces)
        # self._kspaces = np.expand_dims(self._kspaces, axis=0)

        # self._kspaces = np.array([self._ksampler(self._kspaces_full[0][i:i+1])[0] for i in range(len(self._kspaces_full[0]))])
        # print(self._kspaces.shape)

        # self._kspaces = self._kspaces[0]

    @property
    def imgs(self) -> List[ndarray]:
        return self._imgs

    @property
    def kspaces(self) -> List[ndarray]:
        return self._kspaces

    @property
    def kmasks(self) -> List[ndarray]:
        return self._kmasks
        # return self._ksampler.masks[0]

    @property
    def kspaces_full(self) -> List[ndarray]:
        return self._kspaces_full

    @property
    def img_size(self) -> Tuple[int]:
        return self._imgs[0].shape

    @property
    def names(self) -> List[str]:
        return ['Image']

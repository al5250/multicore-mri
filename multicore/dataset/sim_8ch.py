from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from numpy import ndarray
from numpy.fft import fft2
from scipy.io import loadmat
import torch
from torch import Tensor

from multicore.dataset.dataset import MRIDataset
from multicore.ksampler import KSampler, PowerRuleSampler

import pdb


class Sim8ChannelDataset(MRIDataset):

    def __init__(self, file_path: str, ksampler: KSampler, noise_std: float = 1e-3):
        self._ksampler = ksampler

        sim8 = loadmat(file_path)
        self._imgs = [sim8['anatomy_orig'] / np.abs(sim8['anatomy_orig']).max()]
        self.coils = (sim8['b1'] / np.abs(sim8['b1']).max()).transpose(2, 0, 1)

        coil_imgs = self.coils * np.array(self._imgs)
        self._kspaces_full = np.array([fft2(img, norm='ortho') for img in coil_imgs])

        shape = self._kspaces_full.shape
        self._kspaces_full = self._kspaces_full + np.random.normal(0, noise_std, size=shape) + np.random.normal(0, noise_std, size=shape) * 1j
        pdb.set_trace()
        self._kspaces = ksampler(self._kspaces_full)

    @property
    def imgs(self) -> List[ndarray]:
        return self._imgs

    @property
    def kspaces(self) -> List[ndarray]:
        return self._kspaces

    @property
    def kmasks(self) -> List[ndarray]:
        return self._ksampler.masks

    @property
    def kspaces_full(self) -> List[ndarray]:
        return self._kspaces_full

    @property
    def img_size(self) -> Tuple[int]:
        return self._imgs[0].shape

    @property
    def names(self) -> List[str]:
        return ['Image']


class Sim8ChannelDatasetPowerRule(Sim8ChannelDataset):

    def __init__(self, file_path: str, noise_std: float = 1e-3, **kwargs):
        ksampler = PowerRuleSampler(**kwargs)
        super().__init__(file_path, ksampler, noise_std)

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


class MultiCoilInVivoDataset(MRIDataset):

    def __init__(self, file_path: str, ksampler: KSampler):
        self._ksampler = ksampler

        multi_coil = loadmat(file_path)
        self._imgs = multi_coil['img_noisy_under'].transpose(2, 0, 1)
        self._kspaces_full = np.array([fft2(img, norm='ortho') for img in self._imgs])
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
        return [f'Coil{i}' for i in range(32)]


class MultiCoilInVivoDatasetPowerRule(MultiCoilInVivoDataset):

    def __init__(self, file_path: str, **kwargs):
        ksampler = PowerRuleSampler(**kwargs)
        super().__init__(file_path, ksampler)

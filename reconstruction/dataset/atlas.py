from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from numpy import ndarray
from numpy.fft import fft2
from scipy.io import loadmat
import torch
from torch import Tensor

from reconstruction.dataset.dataset import MRIDataset
from reconstruction.ksampler import KSampler, PowerRuleSampler


class AtlasDataset(MRIDataset):

    def __init__(self, file_path: str, ksampler: KSampler):
        self._ksampler = ksampler

        atlas = loadmat(file_path)
        self._imgs = [atlas['img'][:, :, i] for i in range(3)]
        self._kspaces_full = [fft2(img, norm='ortho') for img in self.imgs]
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
        return ['Proton Density', 'T2 Weighted', 'T1 Weighted']


class AtlasDatasetPowerRule(AtlasDataset):

    def __init__(self, file_path: str, **kwargs):
        ksampler = PowerRuleSampler(**kwargs)
        super().__init__(file_path, ksampler)

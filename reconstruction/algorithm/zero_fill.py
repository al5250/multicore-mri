from abc import abstractmethod
from typing import Dict, List, Optional

from numpy import ndarray
import torch
from torch.fft import ifftn

from reconstruction.dataset import MRIDataset
from reconstruction.logger import Logger
from reconstruction.algorithm.algorithm import ReconstructionAlgorithm

from reconstruction.metric import RootMeanSquareError


class ZeroFilling(ReconstructionAlgorithm):

    def __init__(self, complex_imgs: bool = False) -> None:
        self.complex_imgs = complex_imgs
        pass

    @torch.no_grad()
    def reconstruct(self, dataset: MRIDataset, logger: Logger) -> List[ndarray]:
        kspaces = torch.tensor(dataset.kspaces)
        imgs = ifftn(kspaces, dim=(-2, -1), norm='ortho')

        combined_metric = RootMeanSquareError(percentage=True, combine=True)
        combined_rmse = combined_metric(imgs, dataset.imgs).item()
        logger.log_vals(
            f"{str(self)}/{str(combined_metric)}", {'Combined': combined_rmse}
        )

        if self.complex_imgs:
            imgs.real = imgs.real.clamp(0, 1)
            imgs.imag = imgs.imag.clamp(0, 1)
        else:
            imgs = imgs.real.clamp(0, 1)

        return [x.numpy()for x in imgs]

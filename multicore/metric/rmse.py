from typing import List

from numpy import ndarray
import numpy as np

from multicore.metric.metric import Metric


class RootMeanSquareError(Metric):

    def __init__(self, percentage: bool = True, combine: bool = False):
        self.percentage = percentage
        self.combine = combine

    def __call__(self, pred: List[ndarray], targ: List[ndarray]) -> float:
        pred_ = np.array(pred)
        targ_ = np.array(targ)
        if self.combine:
            rmse = np.linalg.norm(pred_ - targ_) / np.linalg.norm(targ_)
        else:
            rmse = np.linalg.norm(pred_ - targ_, axis=(-1, -2)) / np.linalg.norm(targ_, axis=(-1, -2))
        if self.percentage:
            rmse = rmse * 100
        return rmse

from typing import Dict, Optional, List

from torch.utils.tensorboard import SummaryWriter
from numpy import ndarray
import numpy as np

from reconstruction.logger.logger import Logger


class TensorboardLogger(Logger):

    def __init__(self, log_dir="experiment"):
        self.writer = SummaryWriter(log_dir)

    def log_vals(
        self,
        tag: str,
        vals: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        self.writer.add_scalars(tag, vals, step)

    def log_imgs(
        self,
        tag: str,
        imgs: List[ndarray],
        step: Optional[int] = None
    ) -> None:
        self.writer.add_images(tag, np.expand_dims(np.array(imgs), 1), step, dataformats='NCHW')

    def close(self):
        self.writer.close()

from typing import Dict, Optional, List

from torch.utils.tensorboard import SummaryWriter
from numpy import ndarray
import numpy as np

from multicore.logger.logger import Logger


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
        if np.iscomplexobj(imgs):
            imgs_real = self._scale(np.real(imgs))
            imgs_imag = self._scale(np.imag(imgs))
            self.writer.add_images(
                tag + '/real', np.expand_dims(imgs_real, 1), step, dataformats='NCHW'
            )
            self.writer.add_images(
                tag + '/imag', np.expand_dims(imgs_imag, 1), step, dataformats='NCHW'
            )
        else:
            imgs = self._scale(np.array(imgs))
            self.writer.add_images(
                tag, np.expand_dims(imgs, 1), step, dataformats='NCHW'
            )

    def _scale(self, imgs: ndarray):
        max_val = imgs.max()
        min_val = imgs.min()
        return (imgs - min_val) / (max_val - min_val)


    def close(self):
        self.writer.close()

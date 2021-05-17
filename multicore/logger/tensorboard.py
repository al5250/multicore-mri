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
        step: Optional[int] = None,
        scale_individually: bool = False
    ) -> None:
        if np.iscomplexobj(imgs):
            imgs_real = self._scale(np.real(imgs), scale_individually)
            imgs_imag = self._scale(np.imag(imgs), scale_individually)
            self.writer.add_images(
                tag + '/real', np.expand_dims(imgs_real, 1), step, dataformats='NCHW'
            )
            self.writer.add_images(
                tag + '/imag', np.expand_dims(imgs_imag, 1), step, dataformats='NCHW'
            )
        else:
            imgs = self._scale(np.array(imgs), scale_individually)
            self.writer.add_images(
                tag, np.expand_dims(imgs, 1), step, dataformats='NCHW'
            )

    def _scale(self, imgs: ndarray, scale_individually: bool):
        max_val = imgs.max()
        min_val = imgs.min()
        # if scale_individually:
        #     max_val = imgs.max(axis=(-2, -1), keepdims=True)
        #     min_val = imgs.min(axis=(-2, -1), keepdims=True)
        # else:
        #     max_val = imgs.max()
        #     min_val = imgs.min()
        return (imgs - min_val) / (max_val - min_val)


    def close(self):
        self.writer.close()

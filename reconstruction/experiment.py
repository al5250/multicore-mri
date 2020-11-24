from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import numpy as np

from reconstruction.dataset import MRIDataset
from reconstruction.algorithm import ReconstructionAlgorithm
from reconstruction.metric import Metric


@hydra.main(config_path='../configs/experiment', config_name='experiment')
def experiment(config: DictConfig) -> None:
    dataset = instantiate(config.dataset)
    algorithm = instantiate(config.algorithm)
    metric = instantiate(config.metric)
    logger = instantiate(config.logger)

    alg_imgs = algorithm.reconstruct(dataset, logger)
    vals = metric(alg_imgs, dataset.imgs)
    logger.log_vals(f'Final/{str(metric)}', dict(zip(dataset.names, vals)))
    logger.log_imgs('Original', dataset.imgs)
    logger.log_imgs(str(algorithm), alg_imgs)
    logger.close()


if __name__ == "__main__":
    experiment()

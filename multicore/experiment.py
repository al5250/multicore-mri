from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import numpy as np

from multicore.dataset import MRIDataset
from multicore.algorithm import ReconstructionAlgorithm
from multicore.metric import Metric


@hydra.main(config_path='../configs/experiment', config_name='experiment')
def experiment(config: DictConfig) -> None:
    dataset = instantiate(config.dataset)
    algorithm = instantiate(config.algorithm)
    metric = instantiate(config.metric)
    logger = instantiate(config.logger)

    logger.log_imgs('Original', dataset.imgs)

    alg_imgs = algorithm.reconstruct(dataset, logger)
    vals = metric(alg_imgs, dataset.imgs)
    logger.log_vals(f'Final/{str(metric)}', dict(zip(dataset.names, vals)))
    logger.log_imgs(str(algorithm), alg_imgs)
    logger.close()


if __name__ == "__main__":
    experiment()

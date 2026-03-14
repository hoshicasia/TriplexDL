import random

import numpy as np
import torch


def set_worker_seed(worker_id):
    """
    Set random seed for a dataloader worker.

    Args:
        worker_id (int): dataloader worker id.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def set_random_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): random seed.
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

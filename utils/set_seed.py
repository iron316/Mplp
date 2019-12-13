import random

import numpy as np
import torch


def worker_init_fn(worker_id):
    random.seed(worker_id + 2434)
    np.random.seed(worker_id + 2434)


def set_random_seed(seed=2434):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

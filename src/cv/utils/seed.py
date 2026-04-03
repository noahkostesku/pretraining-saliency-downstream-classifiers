"""Random seed helpers for reproducible experiments."""

import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Set all supported random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

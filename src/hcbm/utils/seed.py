"""Reproducibility utilities: fixed random seed configuration and logging."""

import logging
import random
import numpy as np

logger = logging.getLogger(__name__)


def fix_seed(seed: int = 42) -> int:
    """Fix random seeds across standard library, numpy, and torch (if installed).

    Args:
        seed: Integer random seed (default: 42).

    Returns:
        The fixed seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    logger.info("Fixed random seed set to %d", seed)
    return seed

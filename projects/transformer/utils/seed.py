"""Reproducibility utilities for seeding RNGs across Python, NumPy, and PyTorch."""

from __future__ import annotations

import logging
import os
import random
import warnings
from dataclasses import dataclass

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_CUBLAS_DETERMINISTIC_CONFIG = ":4096:8"


@dataclass(frozen=True)
class SeedConfig:
    """Immutable configuration for reproducibility seeding.

    Args:
        seed: Non-negative integer used to seed all RNGs.
        deterministic: When True, enables deterministic CUDA operations and
            disables cuDNN benchmarking. May reduce performance.
    """

    seed: int
    deterministic: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed!r}")

    def apply(self) -> None:
        """Apply this seed configuration globally."""
        set_seed(self.seed, deterministic=self.deterministic)


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Set seeds for Python, NumPy, and (if installed) PyTorch.

    Args:
        seed: Non-negative integer used to seed all RNGs.
        deterministic: When True, aims for repeatable results at the cost of
            speed by enabling CUDA deterministic mode and disabling cuDNN
            benchmarking.

    Raises:
        ValueError: If seed is negative or not an integer.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    _seed_python(seed, deterministic=deterministic)
    _seed_numpy(seed)
    _seed_torch(seed, deterministic=deterministic)

    logger.debug("All RNGs seeded with %d (deterministic=%s)", seed, deterministic)



def _seed_python(seed: int, *, deterministic: bool) -> None:
    """Seed the Python stdlib RNG and set environment variables."""
    random.seed(seed)

    # NOTE: PYTHONHASHSEED only affects hash randomisation when set *before*
    # the interpreter starts.  Setting it here is a best-effort convenience
    # (e.g. for child processes) but won't change hashes in the current process.
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_DETERMINISTIC_CONFIG


def _seed_numpy(seed: int) -> None:
    """Seed both the legacy and modern NumPy RNGs."""
    # Legacy global RNG — still needed by libraries that call np.random.*
    np.random.seed(seed)


def _seed_torch(seed: int, *, deterministic: bool) -> None:
    """Seed PyTorch RNGs and optionally enable deterministic mode."""
    if torch is None:
        return

    torch.manual_seed(seed)  # seeds CPU + current CUDA device

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch (<1.11) doesn't support warn_only
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.use_deterministic_algorithms(True)
        except AttributeError:
            # Very old PyTorch without this API at all
            logger.debug("torch.use_deterministic_algorithms not available")
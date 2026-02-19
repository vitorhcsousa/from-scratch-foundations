"""Tests for tfs.utils.seed module."""

from __future__ import annotations

import os
import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.foundations.projects.transformer.utils.seed import SeedConfig, set_seed


class TestValidation:
    """Seed value validation for both set_seed and SeedConfig."""

    @pytest.mark.parametrize("bad_seed", [-1, -100, -999])
    def test_set_seed_rejects_negative(self, bad_seed: int) -> None:
        with pytest.raises(ValueError, match="non-negative integer"):
            set_seed(bad_seed)

    @pytest.mark.parametrize("bad_seed", [42.0, "42", None, [42]])
    def test_set_seed_rejects_non_int(self, bad_seed: object) -> None:
        with pytest.raises(ValueError, match="non-negative integer"):
            set_seed(bad_seed)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_seed", [-1, 3.14, "oops"])
    def test_seed_config_rejects_invalid(self, bad_seed: object) -> None:
        with pytest.raises(ValueError, match="non-negative integer"):
            SeedConfig(seed=bad_seed)  # type: ignore[arg-type]

    def test_seed_config_accepts_zero(self) -> None:
        config = SeedConfig(seed=0)
        assert config.seed == 0


class TestPythonSeeding:
    """Verify Python stdlib and environment variable seeding."""

    def test_random_reproducibility(self) -> None:
        set_seed(42)
        first = [random.random() for _ in range(10)]

        set_seed(42)
        second = [random.random() for _ in range(10)]

        assert first == second

    def test_different_seeds_differ(self) -> None:
        set_seed(42)
        first = [random.random() for _ in range(10)]

        set_seed(99)
        second = [random.random() for _ in range(10)]

        assert first != second

    def test_pythonhashseed_env_var(self) -> None:
        set_seed(123)
        assert os.environ["PYTHONHASHSEED"] == "123"

    def test_cublas_env_set_when_deterministic(self) -> None:
        set_seed(42, deterministic=True)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    def test_cublas_env_not_set_when_not_deterministic(self) -> None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        set_seed(42, deterministic=False)
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


class TestNumpySeeding:
    """Verify NumPy RNG seeding."""

    def test_numpy_reproducibility(self) -> None:
        set_seed(42)
        first = np.random.rand(10).tolist()

        set_seed(42)
        second = np.random.rand(10).tolist()

        assert first == second

    def test_numpy_different_seeds_differ(self) -> None:
        set_seed(42)
        first = np.random.rand(10).tolist()

        set_seed(99)
        second = np.random.rand(10).tolist()

        assert first != second


# ---------------------------------------------------------------------------
# PyTorch seeding (mocked — torch may not be installed)
# ---------------------------------------------------------------------------


class TestTorchSeeding:
    """Verify PyTorch seeding behavior via mocks."""

    def _make_mock_torch(self, *, cuda_available: bool = True) -> MagicMock:
        mock = MagicMock()
        mock.cuda.is_available.return_value = cuda_available
        return mock

    def test_torch_manual_seed_called(self) -> None:
        mock_torch = self._make_mock_torch()
        with patch("src.foundations.projects.transformer.utils.seed.torch", mock_torch):
            set_seed(42)

        mock_torch.manual_seed.assert_called_once_with(42)

    def test_torch_cuda_seeded_when_available(self) -> None:
        mock_torch = self._make_mock_torch(cuda_available=True)
        with patch("src.foundations.projects.transformer.utils.seed.torch", mock_torch):
            set_seed(42)

        mock_torch.cuda.manual_seed_all.assert_called_once_with(42)

    def test_torch_cuda_skipped_when_unavailable(self) -> None:
        mock_torch = self._make_mock_torch(cuda_available=False)
        with patch("src.foundations.projects.transformer.utils.seed.torch", mock_torch):
            set_seed(42)

        mock_torch.cuda.manual_seed_all.assert_not_called()

    def test_deterministic_mode_enabled(self) -> None:
        mock_torch = self._make_mock_torch()
        with patch("src.foundations.projects.transformer.utils.seed.torch", mock_torch):
            set_seed(42, deterministic=True)

        assert mock_torch.backends.cudnn.deterministic is True
        assert mock_torch.backends.cudnn.benchmark is False
        mock_torch.use_deterministic_algorithms.assert_called_once_with(True, warn_only=True)

    def test_deterministic_mode_skipped(self) -> None:
        mock_torch = self._make_mock_torch()
        with patch("src.foundations.projects.transformer.utils.seed.torch", mock_torch):
            set_seed(42, deterministic=False)

        mock_torch.use_deterministic_algorithms.assert_not_called()

    def test_no_torch_installed(self) -> None:
        with patch("src.foundations.projects.transformer.utils.seed.torch", None):
            set_seed(42)  # should not raise

    def test_old_torch_without_warn_only(self) -> None:
        """Fallback when warn_only kwarg is not supported."""
        mock_torch = self._make_mock_torch()
        mock_torch.use_deterministic_algorithms.side_effect = [
            TypeError("unexpected keyword argument 'warn_only'"),
            None,  # second call without warn_only succeeds
        ]
        with patch("src.foundations.projects.transformer.utils.seed.torch", mock_torch):
            set_seed(42, deterministic=True)  # should not raise

    def test_very_old_torch_without_deterministic_api(self) -> None:
        """Fallback when use_deterministic_algorithms doesn't exist."""
        mock_torch = self._make_mock_torch()
        mock_torch.use_deterministic_algorithms.side_effect = AttributeError
        with patch("src.foundations.projects.transformer.utils.seed.torch", mock_torch):
            set_seed(42, deterministic=True)  # should not raise


class TestSeedConfig:
    """Verify SeedConfig dataclass behavior."""

    def test_frozen(self) -> None:
        config = SeedConfig(seed=42)
        with pytest.raises(AttributeError):
            config.seed = 99  # type: ignore[misc]

    def test_defaults(self) -> None:
        config = SeedConfig(seed=42)
        assert config.deterministic is True

    def test_apply_seeds_correctly(self) -> None:
        SeedConfig(seed=42).apply()
        first = [random.random() for _ in range(10)]

        SeedConfig(seed=42).apply()
        second = [random.random() for _ in range(10)]

        assert first == second

    def test_apply_passes_deterministic_flag(self) -> None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        SeedConfig(seed=42, deterministic=False).apply()
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ

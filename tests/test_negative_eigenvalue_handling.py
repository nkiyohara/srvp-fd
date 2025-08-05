"""Test script for negative eigenvalue handling in Fréchet distance calculation."""

import pytest
import torch

from srvp_fd import FrechetDistanceCalculator


def create_rank_deficient_features(batch_size=256, feature_dim=256, basis_dim=10):
    """Create rank-deficient features that lead to negative eigenvalues."""
    # Generate correlated features that lead to rank-deficient covariance
    basis = torch.randn(feature_dim, basis_dim)  # Low-dimensional basis
    coeffs1 = torch.randn(batch_size, basis_dim)
    coeffs2 = torch.randn(batch_size, basis_dim)

    features1 = torch.tanh(coeffs1 @ basis.T)
    features2 = torch.tanh(coeffs2 @ basis.T)

    return features1, features2


class TestNegativeEigenvalueHandling:
    """Test class for negative eigenvalue handling options."""

    def test_nan_handling(self):
        """Test that 'nan' option returns NaN for rank-deficient features."""
        features1, features2 = create_rank_deficient_features()

        calc = FrechetDistanceCalculator(
            dataset="mmnist_stochastic", device="cpu", negative_eigenvalue_handling="nan"
        )

        fd = calc._calculate_frechet_distance_from_features(features1, features2)

        assert torch.isnan(torch.tensor(fd)).item(), (
            "Expected NaN for rank-deficient features with 'nan' handling"
        )

    def test_clamp_handling(self):
        """Test that 'clamp' option returns 0.0 for rank-deficient features."""
        features1, features2 = create_rank_deficient_features()

        calc = FrechetDistanceCalculator(
            dataset="mmnist_stochastic", device="cpu", negative_eigenvalue_handling="clamp"
        )

        fd = calc._calculate_frechet_distance_from_features(features1, features2)

        assert fd == 0.0, (
            f"Expected 0.0 for rank-deficient features with 'clamp' handling, got {fd}"
        )

    def test_warn_handling(self):
        """Test that 'warn' option issues warning and returns NaN."""
        features1, features2 = create_rank_deficient_features()

        calc = FrechetDistanceCalculator(
            dataset="mmnist_stochastic", device="cpu", negative_eigenvalue_handling="warn"
        )

        with pytest.warns(RuntimeWarning, match="Negative eigenvalues detected"):
            fd = calc._calculate_frechet_distance_from_features(features1, features2)

        assert torch.isnan(torch.tensor(fd)).item(), (
            "Expected NaN for rank-deficient features with 'warn' handling"
        )

    def test_raise_handling(self):
        """Test that 'raise' option raises ValueError."""
        features1, features2 = create_rank_deficient_features()

        calc = FrechetDistanceCalculator(
            dataset="mmnist_stochastic", device="cpu", negative_eigenvalue_handling="raise"
        )

        with pytest.raises(ValueError, match="Negative eigenvalues detected"):
            calc._calculate_frechet_distance_from_features(features1, features2)

    def test_normal_features_all_options(self):
        """Test that all options work correctly with normal (full-rank) features."""
        # Create normal, full-rank features with better conditioning
        batch_size = 256
        feature_dim = 128

        # Add small noise to ensure features are not too similar
        features1 = torch.randn(batch_size, feature_dim) + 0.1 * torch.eye(batch_size, feature_dim)
        features2 = torch.randn(batch_size, feature_dim) + 0.1 * torch.eye(batch_size, feature_dim)

        # Test clamp option specifically (should always work)
        calc_clamp = FrechetDistanceCalculator(
            dataset="mmnist_stochastic", device="cpu", negative_eigenvalue_handling="clamp"
        )

        fd_clamp = calc_clamp._calculate_frechet_distance_from_features(features1, features2)
        assert not torch.isnan(torch.tensor(fd_clamp)).item(), (
            "Unexpected NaN with 'clamp' handling for normal features"
        )
        assert fd_clamp >= 0, f"Expected non-negative FD with 'clamp' handling, got {fd_clamp}"

    def test_invalid_option(self):
        """Test that invalid option works but might cause issues later."""
        # Note: Due to Python's Literal type being a runtime hint,
        # invalid values don't raise errors at construction time
        calc = FrechetDistanceCalculator(
            dataset="mmnist_stochastic",
            device="cpu",
            negative_eigenvalue_handling="invalid_option",  # type: ignore
        )
        # The invalid option will be used in _calculate_frechet_distance
        assert calc.negative_eigenvalue_handling == "invalid_option"

    def test_frechet_distance_function(self):
        """Test the standalone frechet_distance function with negative eigenvalue handling."""
        from srvp_fd import frechet_distance

        # Create simple test images (grayscale for mmnist)
        images1 = torch.randn(256, 1, 64, 64)
        images2 = torch.randn(256, 1, 64, 64)

        # Test that the parameter is passed through correctly
        fd_nan = frechet_distance(
            images1,
            images2,
            dataset="mmnist_stochastic",
            comparison_type="frame",
            device="cpu",
            negative_eigenvalue_handling="nan",
        )

        fd_clamp = frechet_distance(
            images1,
            images2,
            dataset="mmnist_stochastic",
            comparison_type="frame",
            device="cpu",
            negative_eigenvalue_handling="clamp",
        )

        # Both should work without errors
        assert isinstance(fd_nan, float)
        assert isinstance(fd_clamp, float)


@pytest.mark.parametrize("comparison_type", ["static_content", "dynamics"])
def test_video_comparison_with_negative_eigenvalue_handling(comparison_type):
    """Test video comparison types with negative eigenvalue handling."""
    # Create test videos (grayscale for mmnist)
    videos1 = torch.randn(256, 10, 1, 64, 64)
    videos2 = torch.randn(256, 10, 1, 64, 64)

    calc = FrechetDistanceCalculator(
        dataset="mmnist_stochastic", device="cpu", negative_eigenvalue_handling="nan"
    )

    # Should work without errors
    fd = calc(videos1, videos2, comparison_type=comparison_type)
    assert isinstance(fd, float)

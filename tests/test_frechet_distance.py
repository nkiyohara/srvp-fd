"""Tests for the frechet_distance module."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from srvp_mmnist_fd.frechet_distance import _calculate_frechet_distance


def test_calculate_frechet_distance():
    """Test the _calculate_frechet_distance function."""
    # Create two identical distributions
    mu1 = np.array([0.0, 0.0, 0.0])
    sigma1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mu2 = np.array([0.0, 0.0, 0.0])
    sigma2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # The Fréchet distance between identical distributions should be 0
    fd = _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    assert fd == pytest.approx(0.0, abs=1e-6)

    # Create two different distributions
    mu1 = np.array([0.0, 0.0, 0.0])
    sigma1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mu2 = np.array([1.0, 1.0, 1.0])
    sigma2 = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])

    # The Fréchet distance between these distributions should be positive
    fd = _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    assert fd > 0.0

    # Test with non-finite values in covmean
    mu1 = np.array([0.0, 0.0, 0.0])
    sigma1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mu2 = np.array([0.0, 0.0, 0.0])
    sigma2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    # Should not raise an error due to the offset added
    fd = _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    assert fd >= 0.0


@pytest.mark.parametrize(
    ("shape1", "shape2", "expected_error"),
    [
        ((512, 1, 64, 64), (512, 1, 64, 64), None),  # Valid shapes
        ((512, 1, 64, 64), (512, 3, 64, 64), ValueError),  # Different channel dimensions
        ((512, 1, 64, 64), (512, 1, 32, 32), ValueError),  # Different spatial dimensions
        ((512, 1), (512, 1, 64, 64), ValueError),  # Invalid dimensions
    ],
)
def test_frechet_distance_input_validation(shape1, shape2, expected_error, mocker):
    """Test input validation in the frechet_distance function."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model

    # Mock the encoder to directly return features that will work with covariance calculation
    # We need to ensure the features have enough samples and dimensions to avoid singular matrices
    def mock_encoder_call(images):
        batch_size = images.shape[0]
        # Use a small feature dimension that will work well with covariance calculations
        feature_dim = 5

        # Create features with controlled values to ensure non-singular covariance matrices
        features = torch.zeros((batch_size, feature_dim))
        for i in range(batch_size):
            for j in range(feature_dim):
                # Create values that vary across both batch and feature dimensions
                features[i, j] = 1.0 + 0.1 * i + 0.2 * j

        return features

    # Set up the mock to use our function
    mock_model.side_effect = mock_encoder_call
    mock_model.__call__ = mock_model

    # Mock the _get_encoder function
    mocker.patch("srvp_mmnist_fd.frechet_distance._get_encoder", return_value=mock_model)

    # Also mock numpy's cov function to ensure it returns a well-conditioned matrix
    # This is a defensive measure to prevent NaN values
    original_cov = np.cov

    def mock_cov(m, *args, **kwargs):
        cov_matrix = original_cov(m, *args, **kwargs)
        # Add a small value to the diagonal to ensure positive definiteness
        np.fill_diagonal(cov_matrix, cov_matrix.diagonal() + 1e-6)
        return cov_matrix

    cov_patch = mocker.patch("numpy.cov", side_effect=mock_cov)

    # Import the frechet_distance function
    from srvp_mmnist_fd.frechet_distance import frechet_distance

    # Create test tensors
    images1 = torch.rand(*shape1)
    images2 = torch.rand(*shape2)

    if expected_error:
        with pytest.raises(expected_error):
            frechet_distance(images1, images2)
    else:
        # Should not raise an error
        fd = frechet_distance(images1, images2)
        assert isinstance(fd, float)

    # Remove the patch after the test
    cov_patch.stop()

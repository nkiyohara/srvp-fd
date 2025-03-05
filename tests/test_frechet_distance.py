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
        ((10, 1, 64, 64), (10, 1, 64, 64), None),  # Valid shapes
        ((10, 1, 64, 64), (10, 3, 64, 64), ValueError),  # Different channel dimensions
        ((10, 1, 64, 64), (10, 1, 32, 32), ValueError),  # Different spatial dimensions
        ((10, 1), (10, 1, 64, 64), ValueError),  # Invalid dimensions
    ],
)
def test_frechet_distance_input_validation(shape1, shape2, expected_error, mocker):
    """Test input validation in the frechet_distance function."""
    # Import the frechet_distance function here to ensure the mock is applied before import
    from srvp_mmnist_fd.frechet_distance import frechet_distance

    # Create a mock model
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.encode.return_value = torch.zeros(1, 10, 128)

    # Create a mock config
    mock_config = {"nhx": 128}

    # Mock the _get_model_and_config function
    mocker.patch(
        "srvp_mmnist_fd.frechet_distance._get_model_and_config",
        return_value=(mock_model, mock_config),
    )

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

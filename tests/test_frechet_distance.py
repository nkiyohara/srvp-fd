"""Tests for the frechet_distance module."""

import warnings

try:
    import numpy as np
    import pytest
    import torch
except ImportError:
    # These imports are required for the tests to run
    # If they're not available, the tests will fail
    pass

from srvp_fd.frechet_distance import (
    DATASET_PATHS,
    FrechetDistanceCalculator,
    _calculate_frechet_distance,
    frechet_distance,
)


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
        ((127, 1, 64, 64), (512, 1, 64, 64), ValueError),  # Sample size too small in first set
        ((512, 1, 64, 64), (127, 1, 64, 64), ValueError),  # Sample size too small in second set
        ((127, 1, 64, 64), (127, 1, 64, 64), ValueError),  # Sample size too small in both sets
        ((128, 1, 64, 64), (512, 1, 64, 64), ValueError),  # Sample size exactly 128 in first set
        ((512, 1, 64, 64), (128, 1, 64, 64), ValueError),  # Sample size exactly 128 in second set
        ((128, 1, 64, 64), (128, 1, 64, 64), ValueError),  # Sample size exactly 128 in both sets
    ],
)
def test_frechet_distance_input_validation(shape1, shape2, expected_error):
    """Test input validation in the frechet_distance function."""
    # Create mock tensors
    images1 = torch.rand(*shape1)
    images2 = torch.rand(*shape2)

    # Create a calculator with a default dataset
    calculator = FrechetDistanceCalculator(dataset="mmnist_stochastic")

    if expected_error:
        with pytest.raises(expected_error):
            calculator(images1, images2)
    else:
        # Should not raise an error
        fd = calculator(images1, images2)
        assert isinstance(fd, float)


@pytest.mark.parametrize(
    "dataset",
    ["mmnist_stochastic"],  # Use only one dataset to speed up tests
)
def test_frechet_distance_with_dataset(dataset):
    """Test frechet_distance function with a real dataset."""
    # Create tensors with appropriate channels for the dataset
    channels = 3 if dataset == "bair" else 1
    images1 = torch.rand(129, channels, 64, 64)
    images2 = torch.rand(129, channels, 64, 64)

    # Calculate Fréchet distance using the real implementation
    fd = frechet_distance(images1, images2, dataset=dataset)

    # Check that the result is a float
    assert isinstance(fd, float)
    assert fd >= 0.0


def test_frechet_distance_calculator():
    """Test the FrechetDistanceCalculator class."""
    # Create tensors
    images1 = torch.rand(129, 1, 64, 64)
    images2 = torch.rand(129, 1, 64, 64)
    images3 = torch.rand(129, 1, 64, 64)

    # Create a calculator
    calculator = FrechetDistanceCalculator(dataset="mmnist_stochastic")

    # Calculate Fréchet distance
    fd1 = calculator(images1, images2)
    fd2 = calculator(images1, images3)

    # Check that the results are floats
    assert isinstance(fd1, float)
    assert isinstance(fd2, float)
    assert fd1 >= 0.0
    assert fd2 >= 0.0

    # Test extract_features method
    features1 = calculator.extract_features(images1)
    features2 = calculator.extract_features(images2)

    # Check that the features have the expected shape
    assert features1.shape[0] == 129  # Batch size
    assert features1.shape[1] > 0  # Feature dimension

    # Test calculate_frechet_distance_from_features method
    fd3 = calculator.calculate_frechet_distance_from_features(features1, features2)
    assert isinstance(fd3, float)
    assert fd3 >= 0.0

    # Check that the result is the same as calling the calculator directly
    assert fd3 == pytest.approx(fd1, abs=1e-6)


@pytest.mark.parametrize(
    "dataset",
    list(DATASET_PATHS.keys()),
)
def test_frechet_distance_calculator_with_different_datasets(dataset):
    """Test FrechetDistanceCalculator with different datasets."""
    # Skip tests for datasets other than mmnist_stochastic to speed up tests
    if dataset != "mmnist_stochastic":
        pytest.skip(f"Skipping test for dataset {dataset} to speed up tests")

    # Create tensors with appropriate channels for the dataset
    channels = 3 if dataset == "bair" else 1
    images1 = torch.rand(129, channels, 64, 64)
    images2 = torch.rand(129, channels, 64, 64)

    # Create a calculator
    calculator = FrechetDistanceCalculator(dataset=dataset)

    # Calculate Fréchet distance
    fd = calculator(images1, images2)

    # Check that the result is a float
    assert isinstance(fd, float)
    assert fd >= 0.0


def test_skip_connection_warning():
    """Test that a warning is issued when the model has skip connections."""
    # Create tensors with appropriate channels for each dataset
    images1_rgb = torch.rand(129, 3, 64, 64)  # RGB for BAIR
    images2_rgb = torch.rand(129, 3, 64, 64)

    images1_gray = torch.rand(129, 1, 64, 64)  # Grayscale for MMNIST
    images2_gray = torch.rand(129, 1, 64, 64)

    # Trigger the warning by calling frechet_distance with a dataset that uses skip connections
    with pytest.warns(UserWarning, match="skip connections"):
        fd = frechet_distance(images1_rgb, images2_rgb, dataset="bair")

        # Check that the result is a float
        assert isinstance(fd, float)

    # No warning should be issued for datasets without skip connections
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        fd = frechet_distance(images1_gray, images2_gray, dataset="mmnist_stochastic")
        assert isinstance(fd, float)
        assert len(record) == 0, "Warning was issued for a dataset without skip connections"


def test_default_encoder_warning_when_no_dataset_or_model_path():
    """Test that a warning is issued when neither dataset nor model_path is provided."""
    # Create tensors
    images1 = torch.rand(129, 1, 64, 64)
    images2 = torch.rand(129, 1, 64, 64)

    # This should issue a warning but not raise an error
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        fd = frechet_distance(images1, images2, dataset=None, model_path=None)
        assert isinstance(fd, float)
        assert len(record) > 0, (
            "No warning was issued when neither dataset nor model_path was provided"
        )
        assert any("default encoder" in str(w.message) for w in record), (
            "Warning about default encoder was not issued"
        )

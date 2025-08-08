"""Tests for the frechet_distance module."""

import warnings

import numpy as np

try:
    import pytest
    import torch
except ImportError:
    # These imports are required for the tests to run
    # If they're not available, the tests will fail
    pass

from srvp_fd.frechet_distance import (
    DATASET_PATHS,
    FrechetDistanceCalculator,
    _calculate_frechet_distance_numpy,
    frechet_distance,
)


def test_calculate_frechet_distance():
    """Test the _calculate_frechet_distance_numpy function."""
    # Create two identical distributions
    mu1 = np.array([0.0, 0.0, 0.0])
    sigma1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mu2 = np.array([0.0, 0.0, 0.0])
    sigma2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # The Fréchet distance between identical distributions should be 0
    fd = _calculate_frechet_distance_numpy(mu1, sigma1, mu2, sigma2)
    assert fd == pytest.approx(0.0, abs=1e-6)

    # Create two different distributions
    mu1 = np.array([0.0, 0.0, 0.0])
    sigma1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mu2 = np.array([1.0, 1.0, 1.0])
    sigma2 = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])

    # The Fréchet distance between these distributions should be positive
    fd = _calculate_frechet_distance_numpy(mu1, sigma1, mu2, sigma2)
    assert fd > 0.0

    # Test with near-zero values in sigma matrices
    mu1 = np.array([0.0, 0.0, 0.0])
    sigma1 = np.array([[1e-10, 0.0, 0.0], [0.0, 1e-10, 0.0], [0.0, 0.0, 1e-10]])
    mu2 = np.array([0.0, 0.0, 0.0])
    sigma2 = np.array([[1e-10, 0.0, 0.0], [0.0, 1e-10, 0.0], [0.0, 0.0, 1e-10]])

    # Should not raise an error due to SciPy's robust implementation
    fd = _calculate_frechet_distance_numpy(mu1, sigma1, mu2, sigma2)
    assert fd >= 0.0


@pytest.mark.parametrize(
    ("shape1", "shape2", "expected_error"),
    [
        ((512, 1, 64, 64), (512, 1, 64, 64), None),  # Valid shapes
        ((129, 1, 64, 64), (129, 1, 64, 64), None),  # Valid shapes, just above minimum
        ((512, 1, 64, 64), (512, 3, 64, 64), ValueError),  # Different channel dimensions
        ((512, 1, 64, 64), (512, 1, 32, 32), ValueError),  # Different spatial dimensions
        ((512, 1), (512, 1, 64, 64), ValueError),  # Invalid dimensions
    ],
)
def test_frechet_distance_frame_input_validation(shape1, shape2, expected_error):
    """Test input validation for frame comparison type in the frechet_distance function."""
    # Create mock tensors
    images1 = torch.rand(*shape1)
    images2 = torch.rand(*shape2)

    # Create a calculator with a default dataset
    calculator = FrechetDistanceCalculator(dataset="mmnist_stochastic")

    if expected_error:
        with pytest.raises(expected_error):
            calculator(images1, images2, comparison_type="frame")
    else:
        # Should not raise an error
        fd = calculator(images1, images2, comparison_type="frame")
        assert isinstance(fd, float)


@pytest.mark.parametrize(
    ("shape1", "shape2", "expected_error"),
    [
        ((129, 8, 1, 64, 64), (129, 8, 1, 64, 64), None),  # Valid shapes
        ((129, 8, 1, 64, 64), (129, 8, 3, 64, 64), ValueError),  # Different channel dimensions
        ((129, 8, 1, 64, 64), (129, 8, 1, 32, 32), ValueError),  # Different spatial dimensions
        ((129, 8, 1), (129, 8, 1, 64, 64), ValueError),  # Invalid dimensions
    ],
)
def test_frechet_distance_video_input_validation(shape1, shape2, expected_error):
    """Test input validation for video comparison types in the frechet_distance function."""
    # Create mock tensors
    videos1 = torch.rand(*shape1)
    videos2 = torch.rand(*shape2)

    # Create a calculator with a default dataset
    calculator = FrechetDistanceCalculator(dataset="mmnist_stochastic")

    # Test for static_content comparison
    if expected_error:
        with pytest.raises(expected_error):
            calculator(videos1, videos2, comparison_type="static_content")
    else:
        # Should not raise an error
        fd = calculator(videos1, videos2, comparison_type="static_content")
        assert isinstance(fd, float)

    # Test for dynamics comparison
    if expected_error:
        with pytest.raises(expected_error):
            calculator(videos1, videos2, comparison_type="dynamics")
    else:
        # Should not raise an error
        fd = calculator(videos1, videos2, comparison_type="dynamics")
        assert isinstance(fd, float)


@pytest.mark.parametrize(
    "dataset",
    list(DATASET_PATHS.keys()),
)
@pytest.mark.parametrize(
    "comparison_type",
    ["frame", "static_content", "dynamics"],
)
def test_frechet_distance_with_comparison_types(dataset, comparison_type):
    """Test frechet_distance function with different comparison types."""
    # Create tensors with appropriate channels for the dataset
    channels = 3 if dataset in ["bair", "human"] else 1

    if comparison_type == "frame":
        # For frame comparison, we need 4D tensors
        input1 = torch.rand(129, channels, 64, 64)
        input2 = torch.rand(129, channels, 64, 64)
    else:
        # For static_content and dynamics comparisons, we need 5D tensors (videos)
        input1 = torch.rand(129, 16, channels, 64, 64)
        input2 = torch.rand(129, 16, channels, 64, 64)

    # Calculate Fréchet distance using the specified comparison type
    fd = frechet_distance(input1, input2, dataset=dataset, comparison_type=comparison_type)

    # Check that the result is a float
    assert isinstance(fd, float)
    assert fd >= 0.0


def test_invalid_comparison_type():
    """Test that an invalid comparison type raises a ValueError."""
    # Create tensors
    images = torch.rand(129, 1, 64, 64)

    # Create a calculator
    calculator = FrechetDistanceCalculator(dataset="mmnist_stochastic")

    # Test with an invalid comparison type
    with pytest.raises(ValueError, match="Unrecognized comparison_type"):
        calculator(images, images, comparison_type="invalid_type")


def test_frechet_distance_calculator():
    """Test the FrechetDistanceCalculator class."""
    # Create tensors
    images = torch.rand(129, 1, 64, 64)
    videos = torch.rand(129, 16, 1, 64, 64)

    # Create a calculator
    calculator = FrechetDistanceCalculator(dataset="mmnist_stochastic")

    # Test frame comparison
    fd_frame = calculator(images, images, comparison_type="frame")
    assert isinstance(fd_frame, float)
    assert fd_frame >= 0.0

    # Test static_content comparison
    fd_static = calculator(videos, videos, comparison_type="static_content")
    assert isinstance(fd_static, float)
    assert fd_static >= 0.0

    # Test dynamics comparison
    fd_dynamics = calculator(videos, videos, comparison_type="dynamics")
    assert isinstance(fd_dynamics, float)
    assert fd_dynamics >= 0.0

    # Test extract_features method
    features = calculator.extract_features(images)
    assert isinstance(features, torch.Tensor)
    assert features.shape[0] == 129  # Batch size
    assert features.shape[1] > 0  # Feature dimension

    # Test extract_w method
    w_features = calculator.extract_w(videos)
    assert isinstance(w_features, torch.Tensor)
    assert w_features.shape[0] == 129  # Batch size
    assert w_features.shape[1] > 0  # Feature dimension

    # Test extract_q_y_0_params method
    q_y_0_params = calculator.extract_q_y_0_params(videos)
    assert isinstance(q_y_0_params, torch.Tensor)
    assert q_y_0_params.shape[0] == 129  # Batch size
    assert q_y_0_params.shape[1] > 0  # Feature dimension
    # Should be twice the size of ny (mean and variance for each dimension)
    assert q_y_0_params.shape[1] % 2 == 0


@pytest.mark.parametrize(
    "dataset",
    list(DATASET_PATHS.keys()),
)
def test_frechet_distance_calculator_with_different_datasets(dataset):
    """Test FrechetDistanceCalculator with different datasets."""
    # Create tensors with appropriate channels for the dataset
    channels = 3 if dataset in ["bair", "human"] else 1
    images = torch.rand(129, channels, 64, 64)
    videos = torch.rand(129, 16, channels, 64, 64)

    # Create a calculator
    calculator = FrechetDistanceCalculator(dataset=dataset)

    # Test frame comparison
    fd_frame = calculator(images, images, comparison_type="frame")
    assert isinstance(fd_frame, float)
    assert fd_frame >= 0.0

    # Test static_content comparison
    fd_static = calculator(videos, videos, comparison_type="static_content")
    assert isinstance(fd_static, float)
    assert fd_static >= 0.0

    # Test dynamics comparison
    fd_dynamics = calculator(videos, videos, comparison_type="dynamics")
    assert isinstance(fd_dynamics, float)
    assert fd_dynamics >= 0.0


def test_skip_connection_warning():
    """Test that a warning is issued when the model has skip connections."""
    # Create tensors with appropriate channels for each dataset
    images_rgb = torch.rand(129, 3, 64, 64)  # RGB for BAIR
    videos_rgb = torch.rand(129, 16, 3, 64, 64)  # RGB videos for BAIR

    images_gray = torch.rand(129, 1, 64, 64)  # Grayscale for MMNIST
    videos_gray = torch.rand(129, 16, 1, 64, 64)  # Grayscale videos for MMNIST

    # Test for frame comparison
    with pytest.warns(UserWarning, match="skip connections"):
        fd = frechet_distance(images_rgb, images_rgb, dataset="bair", comparison_type="frame")
    assert isinstance(fd, float)

    # Test for static_content comparison
    with pytest.warns(UserWarning, match="skip connections"):
        fd = frechet_distance(
            videos_rgb, videos_rgb, dataset="bair", comparison_type="static_content"
        )
    assert isinstance(fd, float)

    # Test for dynamics comparison
    with pytest.warns(UserWarning, match="skip connections"):
        fd = frechet_distance(videos_rgb, videos_rgb, dataset="bair", comparison_type="dynamics")
    assert isinstance(fd, float)

    # No warning should be issued for datasets without skip connections
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")

        # Test frame comparison
        fd = frechet_distance(
            images_gray, images_gray, dataset="mmnist_stochastic", comparison_type="frame"
        )
        assert isinstance(fd, float)

        # Test static_content comparison
        fd = frechet_distance(
            videos_gray, videos_gray, dataset="mmnist_stochastic", comparison_type="static_content"
        )
        assert isinstance(fd, float)

        # Test dynamics comparison
        fd = frechet_distance(
            videos_gray, videos_gray, dataset="mmnist_stochastic", comparison_type="dynamics"
        )
        assert isinstance(fd, float)

        assert len(record) == 0, "Warning was issued for a dataset without skip connections"


def test_matrix_sqrt():
    """Test the matrix square root function."""
    from srvp_fd.frechet_distance import _matrix_sqrt

    # Test with identity matrix
    identity = torch.eye(3)
    sqrt_identity = _matrix_sqrt(identity)
    # Square root of identity is identity
    assert torch.allclose(sqrt_identity, identity, atol=1e-6)

    # Test with a more complex matrix
    matrix = torch.tensor([[4.0, 1.0], [1.0, 9.0]])
    sqrt_matrix = _matrix_sqrt(matrix)
    # Verify A = sqrt_A @ sqrt_A
    reconstructed_matrix = sqrt_matrix @ sqrt_matrix
    assert torch.allclose(reconstructed_matrix, matrix, atol=1e-6)

    # Test with diagonal matrix
    diag = torch.diag(torch.tensor([4.0, 9.0, 16.0]))
    sqrt_diag = _matrix_sqrt(diag)
    expected_sqrt_diag = torch.diag(torch.tensor([2.0, 3.0, 4.0]))
    assert torch.allclose(sqrt_diag, expected_sqrt_diag, atol=1e-6)

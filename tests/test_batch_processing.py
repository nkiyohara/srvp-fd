"""Tests for batch processing functionality in Fréchet distance calculation."""

import pytest
import torch

from srvp_fd.frechet_distance import FrechetDistanceCalculator, frechet_distance


class TestBatchProcessing:
    """Test cases for batch processing functionality."""

    @pytest.fixture
    def test_data(self):
        """Create test data for batch processing tests."""
        batch_size_total = 200  # Must be > 128 for existing validation
        channels, height, width = 1, 64, 64  # MMNIST uses 1 channel (grayscale)

        # Generate reproducible test data
        torch.manual_seed(42)
        images1 = torch.randn(batch_size_total, channels, height, width)
        images2 = torch.randn(batch_size_total, channels, height, width)

        return images1, images2

    @pytest.fixture
    def test_video_data(self):
        """Create test video data for batch processing tests."""
        batch_size_total = 150  # Must be > 128 for existing validation
        seq_length = 10
        channels, height, width = 1, 64, 64

        # Generate reproducible test video data
        torch.manual_seed(42)
        videos1 = torch.randn(batch_size_total, seq_length, channels, height, width)
        videos2 = torch.randn(batch_size_total, seq_length, channels, height, width)

        return videos1, videos2

    def test_frame_batch_processing_consistency(self, test_data):
        """Test that batch processing gives consistent results for frame comparison."""
        images1, images2 = test_data

        # Calculate FD without batching
        calculator = FrechetDistanceCalculator(dataset="mmnist_stochastic")
        fd_no_batch = calculator(images1, images2, comparison_type="frame")

        # Calculate FD with different batch sizes
        fd_batch_10 = calculator(images1, images2, comparison_type="frame", batch_size=10)
        fd_batch_50 = calculator(images1, images2, comparison_type="frame", batch_size=50)

        # Results should be very close (allowing for small numerical differences)
        assert abs(fd_no_batch - fd_batch_10) < 1e-4
        assert abs(fd_no_batch - fd_batch_50) < 1e-4
        assert abs(fd_batch_10 - fd_batch_50) < 1e-4

    def test_static_content_batch_processing_consistency(self, test_video_data):
        """Test that batch processing gives consistent results for static content comparison."""
        videos1, videos2 = test_video_data

        # Calculate FD without batching
        calculator = FrechetDistanceCalculator(dataset="mmnist_stochastic")
        fd_no_batch = calculator(videos1, videos2, comparison_type="static_content")

        # Calculate FD with different batch sizes
        fd_batch_5 = calculator(videos1, videos2, comparison_type="static_content", batch_size=5)
        fd_batch_10 = calculator(videos1, videos2, comparison_type="static_content", batch_size=10)

        # Results should be very close
        assert abs(fd_no_batch - fd_batch_5) < 1e-4
        assert abs(fd_no_batch - fd_batch_10) < 1e-4
        assert abs(fd_batch_5 - fd_batch_10) < 1e-4

    def test_dynamics_batch_processing_consistency(self, test_video_data):
        """Test that batch processing gives consistent results for dynamics comparison."""
        videos1, videos2 = test_video_data

        # Calculate FD without batching
        calculator = FrechetDistanceCalculator(dataset="mmnist_stochastic")
        fd_no_batch = calculator(videos1, videos2, comparison_type="dynamics")

        # Calculate FD with different batch sizes
        fd_batch_5 = calculator(videos1, videos2, comparison_type="dynamics", batch_size=5)
        fd_batch_10 = calculator(videos1, videos2, comparison_type="dynamics", batch_size=10)

        # Results should be very close
        assert abs(fd_no_batch - fd_batch_5) < 1e-4
        assert abs(fd_no_batch - fd_batch_10) < 1e-4
        assert abs(fd_batch_5 - fd_batch_10) < 1e-4

    def test_batch_size_edge_cases(self, test_data):
        """Test edge cases for batch size parameter."""
        images1, images2 = test_data

        calculator = FrechetDistanceCalculator(dataset="mmnist_stochastic")

        # Batch size equal to total size should work
        fd_full_batch = calculator(
            images1, images2, comparison_type="frame", batch_size=images1.shape[0]
        )

        # Batch size larger than total size should work
        fd_large_batch = calculator(
            images1, images2, comparison_type="frame", batch_size=images1.shape[0] * 2
        )

        # Results should be identical
        assert abs(fd_full_batch - fd_large_batch) < 1e-6

    def test_single_sample_batch(self, test_data):
        """Test batch processing with batch_size=1."""
        images1, images2 = test_data
        # Use subset that still meets validation requirements (>128)
        images1_small = images1[:130]  # Still > 128
        images2_small = images2[:130]

        calculator = FrechetDistanceCalculator(dataset="mmnist_stochastic")

        # Calculate with batch_size=1
        fd_batch_1 = calculator(images1_small, images2_small, comparison_type="frame", batch_size=1)

        # Calculate without batching
        fd_no_batch = calculator(images1_small, images2_small, comparison_type="frame")

        # Results should be very close
        assert abs(fd_batch_1 - fd_no_batch) < 1e-4

    def test_frechet_distance_function_with_batch_size(self, test_data):
        """Test the top-level frechet_distance function with batch_size parameter."""
        images1, images2 = test_data

        # Test with different batch sizes
        fd_no_batch = frechet_distance(
            images1, images2, dataset="mmnist_stochastic", comparison_type="frame"
        )
        fd_batch_20 = frechet_distance(
            images1, images2, dataset="mmnist_stochastic", comparison_type="frame", batch_size=20
        )

        # Results should be consistent
        assert abs(fd_no_batch - fd_batch_20) < 1e-4

    @pytest.mark.parametrize("batch_size", [None, 5, 10, 25, 50])
    def test_batch_processing_deterministic(self, test_data, batch_size):
        """Test that batch processing is deterministic for various batch sizes."""
        images1, images2 = test_data

        calculator = FrechetDistanceCalculator(dataset="mmnist_stochastic")

        # Calculate FD multiple times with same batch_size
        results = []
        for _ in range(3):
            fd = calculator(images1, images2, comparison_type="frame", batch_size=batch_size)
            results.append(fd)

        # All results should be identical
        assert all(abs(r - results[0]) < 1e-8 for r in results)

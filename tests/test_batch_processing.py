"""Tests for batch processing functionality in Fréchet distance calculation."""

import pytest
import torch

from srvp_fd.frechet_distance import FrechetDistanceCalculator, frechet_distance


class TestBatchProcessing:
    """Test cases for batch processing functionality."""

    @pytest.fixture
    def test_data(self):
        """Create test data for batch processing tests."""
        batch_size_total = 200
        channels, height, width = 1, 64, 64  # MMNIST uses 1 channel (grayscale)

        # Generate reproducible test data
        torch.manual_seed(42)
        images1 = torch.randn(batch_size_total, channels, height, width)
        images2 = torch.randn(batch_size_total, channels, height, width)

        return images1, images2

    @pytest.fixture
    def calculator(self):
        """Create a FrechetDistanceCalculator instance."""
        return FrechetDistanceCalculator(dataset="mmnist_stochastic")

    def test_no_batch_processing(self, calculator, test_data):
        """Test processing all data at once (no batching)."""
        images1, images2 = test_data

        fd_no_batch = calculator(images1, images2, batch_size=None)

        assert isinstance(fd_no_batch, float)
        assert fd_no_batch > 0
        assert not torch.isnan(torch.tensor(fd_no_batch))

    @pytest.mark.parametrize("batch_size", [32, 64, 100])
    def test_batch_processing_consistency(self, calculator, test_data, batch_size):
        """Test that different batch sizes produce identical results."""
        images1, images2 = test_data

        # Calculate without batching (reference)
        fd_reference = calculator(images1, images2, batch_size=None)

        # Calculate with batching
        fd_batched = calculator(images1, images2, batch_size=batch_size)

        assert isinstance(fd_batched, float)
        assert fd_batched > 0
        assert not torch.isnan(torch.tensor(fd_batched))

        # Results should be identical (or very close due to floating point precision)
        assert abs(fd_reference - fd_batched) < 1e-6

    def test_batch_size_larger_than_data(self, calculator, test_data):
        """Test that batch_size larger than data size works correctly."""
        images1, images2 = test_data
        data_size = images1.shape[0]

        # Use batch size larger than data
        large_batch_size = data_size + 50

        fd_large_batch = calculator(images1, images2, batch_size=large_batch_size)
        fd_no_batch = calculator(images1, images2, batch_size=None)

        # Should produce identical results
        assert abs(fd_large_batch - fd_no_batch) < 1e-10

    def test_convenience_function_with_batching(self, test_data):
        """Test the convenience function with batch processing."""
        images1, images2 = test_data

        # Test convenience function with batching
        fd_convenience = frechet_distance(
            images1, images2, dataset="mmnist_stochastic", batch_size=50
        )

        assert isinstance(fd_convenience, float)
        assert fd_convenience > 0
        assert not torch.isnan(torch.tensor(fd_convenience))

    def test_batch_processing_with_small_data(self, calculator):
        """Test batch processing with small datasets."""
        # Create small dataset
        torch.manual_seed(123)
        small_images1 = torch.randn(10, 1, 64, 64)
        small_images2 = torch.randn(10, 1, 64, 64)

        # Use batch size larger than data
        fd_small = calculator(small_images1, small_images2, batch_size=20)

        assert isinstance(fd_small, float)
        assert fd_small >= 0
        assert not torch.isnan(torch.tensor(fd_small))

    def test_batch_processing_memory_efficiency(self, calculator):
        """Test that batch processing doesn't load all data simultaneously."""
        # This is more of a conceptual test - in practice, this would require
        # monitoring actual memory usage, but we can at least verify the functionality
        torch.manual_seed(456)
        images1 = torch.randn(150, 1, 64, 64)
        images2 = torch.randn(150, 1, 64, 64)

        # Use very small batch size
        fd_small_batch = calculator(images1, images2, batch_size=10)
        fd_reference = calculator(images1, images2, batch_size=None)

        # Results should be consistent
        assert abs(fd_small_batch - fd_reference) < 1e-6

    @pytest.mark.parametrize("comparison_type", ["frame"])
    def test_batch_processing_different_comparison_types(
        self, calculator, test_data, comparison_type
    ):
        """Test batch processing with different comparison types."""
        images1, images2 = test_data

        fd_batched = calculator(images1, images2, comparison_type=comparison_type, batch_size=50)
        fd_reference = calculator(
            images1, images2, comparison_type=comparison_type, batch_size=None
        )

        assert abs(fd_batched - fd_reference) < 1e-6

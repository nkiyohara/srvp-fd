"""Example script demonstrating how to use the srvp-fd package."""

import torch

from srvp_fd import DATASET_PATHS, frechet_distance

print("Available datasets:")
for dataset in DATASET_PATHS:
    print(f"  - {dataset}")

# Example 1: Moving MNIST (grayscale images)
print("\nExample 1: Moving MNIST (grayscale images)")
# Generate random tensors to simulate Moving MNIST images
# Shape: [batch_size, channels, height, width]
batch_size = 512
channels = 1  # Grayscale
height = 64
width = 64

# Create two sets of random images
random_images1 = torch.rand(batch_size, channels, height, width)
random_images2 = torch.rand(batch_size, channels, height, width)

# Create two sets of similar images (with small differences)
similar_images1 = torch.rand(batch_size, channels, height, width)
similar_images2 = similar_images1 + 0.01 * torch.randn(batch_size, channels, height, width)

# Calculate Fréchet distance between random images using the stochastic MMNIST model
fd_random = frechet_distance(random_images1, random_images2, dataset="mmnist_stochastic")
print(f"Fréchet distance between random images (stochastic model): {fd_random}")

# Calculate Fréchet distance between similar images using the stochastic MMNIST model
fd_similar = frechet_distance(similar_images1, similar_images2, dataset="mmnist_stochastic")
print(f"Fréchet distance between similar images (stochastic model): {fd_similar}")

# The Fréchet distance between similar images should be smaller
print(f"Is fd_similar < fd_random? {fd_similar < fd_random}")

# Example 2: BAIR dataset (RGB images)
print("\nExample 2: BAIR dataset (RGB images)")
# Generate random tensors to simulate BAIR images
# Shape: [batch_size, channels, height, width]
batch_size = 256
channels = 3  # RGB
height = 64
width = 64

# Create two sets of random images
bair_images1 = torch.rand(batch_size, channels, height, width)
bair_images2 = torch.rand(batch_size, channels, height, width)

# Calculate Fréchet distance using the BAIR model
fd_bair = frechet_distance(bair_images1, bair_images2, dataset="bair")
print(f"Fréchet distance between BAIR images: {fd_bair}")

# Example 3: Using a local model file (if available)
print("\nExample 3: Using a local model file (commented out)")
# If you have a local model file, you can use it like this:
# model_path = "path/to/your/model.pt"
# fd_local = frechet_distance(random_images1, random_images2, model_path=model_path)
# print(f"Fréchet distance using local model: {fd_local}")

# Example 4: Comparing different datasets
print("\nExample 4: Comparing different datasets")
# Note: This is just for demonstration. In practice, you should use the appropriate
# dataset for your images, as each model is trained on specific data distributions.
grayscale_images1 = torch.rand(256, 1, 64, 64)
grayscale_images2 = torch.rand(256, 1, 64, 64)

print("Fréchet distances using different models:")
for dataset in ["mmnist_stochastic", "mmnist_deterministic"]:
    fd = frechet_distance(grayscale_images1, grayscale_images2, dataset=dataset)
    print(f"  - {dataset}: {fd}")

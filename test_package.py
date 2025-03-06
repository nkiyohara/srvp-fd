"""Test script to verify the functionality of the srvp-fd package."""

import torch

from srvp_fd import frechet_distance

# Generate random tensors to simulate Moving MNIST images
# Shape: [batch_size, channels, height, width]
batch_size = 100
channels = 1
height = 64
width = 64

# Create two sets of random images
random_images1 = torch.rand(batch_size, channels, height, width)
random_images2 = torch.rand(batch_size, channels, height, width)

# Create two sets of similar images (with small differences)
similar_images1 = torch.rand(batch_size, channels, height, width)
similar_images2 = similar_images1 + 0.01 * torch.randn(batch_size, channels, height, width)

# Calculate Fréchet distance between random images
fd_random = frechet_distance(random_images1, random_images2)
print(fd_random)

# Calculate Fréchet distance between similar images
fd_similar = frechet_distance(similar_images1, similar_images2)
print(fd_similar)

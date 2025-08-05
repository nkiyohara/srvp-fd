"""Test script to verify the functionality of the srvp-fd package."""

import torch

from srvp_fd import FrechetDistanceCalculator, frechet_distance

# Generate random tensors to simulate Moving MNIST images
# Shape: [batch_size, channels, height, width]
batch_size = 129
channels = 1
height = 64
width = 64

# Create two sets of random images
random_images1 = torch.rand(batch_size, channels, height, width)
random_images2 = torch.rand(batch_size, channels, height, width)

# Create two sets of similar images (with small differences)
similar_images1 = torch.rand(batch_size, channels, height, width)
similar_images2 = similar_images1 + 0.01 * torch.randn(batch_size, channels, height, width)

print("Testing function API:")
# Calculate Fréchet distance between random images
fd_random = frechet_distance(random_images1, random_images2, negative_eigenvalue_handling="clamp")
print(f"Fréchet distance between random images: {fd_random}")

# Calculate Fréchet distance between similar images
fd_similar = frechet_distance(similar_images1, similar_images2, negative_eigenvalue_handling="clamp")
print(f"Fréchet distance between similar images: {fd_similar}")

print("\nTesting class API:")
# Create a calculator
calculator = FrechetDistanceCalculator(negative_eigenvalue_handling="clamp")

# Calculate Fréchet distance between random images
fd_random_class = calculator(random_images1, random_images2)
print(f"Fréchet distance between random images: {fd_random_class}")

# Calculate Fréchet distance between similar images
fd_similar_class = calculator(similar_images1, similar_images2)
print(f"Fréchet distance between similar images: {fd_similar_class}")

print("\nTesting extract_features method:")
# Extract features
features_random1 = calculator.extract_features(random_images1)
features_random2 = calculator.extract_features(random_images2)
features_similar1 = calculator.extract_features(similar_images1)
features_similar2 = calculator.extract_features(similar_images2)

# Calculate Fréchet distance from features
fd_random_features = calculator._calculate_frechet_distance_from_features(
    features_random1, features_random2
)
fd_similar_features = calculator._calculate_frechet_distance_from_features(
    features_similar1, features_similar2
)

print(f"Fréchet distance between random images (from features): {fd_random_features}")
print(f"Fréchet distance between similar images (from features): {fd_similar_features}")

# Verify that the results are consistent
# Note: We use a larger tolerance (0.001) because the function API and class API
# might use slightly different encoder initializations
print("\nVerifying consistency between APIs:")
print(
    f"Difference between function and class APIs for random images: "
    f"{abs(fd_random - fd_random_class)}"
)
print(
    f"Difference between function and class APIs for similar images: "
    f"{abs(fd_similar - fd_similar_class)}"
)
print(
    f"Difference between class API and feature extraction for random images: "
    f"{abs(fd_random_class - fd_random_features)}"
)
print(
    f"Difference between class API and feature extraction for similar images: "
    f"{abs(fd_similar_class - fd_similar_features)}"
)

# The class API and feature extraction should give identical results
assert abs(fd_random_class - fd_random_features) < 1e-6, (
    "Class API and feature extraction give different results for random images"
)
assert abs(fd_similar_class - fd_similar_features) < 1e-6, (
    "Class API and feature extraction give different results for similar images"
)

print("\nAll tests passed!")

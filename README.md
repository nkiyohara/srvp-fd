# SRVP-FD

[![Python Package](https://github.com/nkiyohara/srvp-fd/actions/workflows/python-package.yml/badge.svg)](https://github.com/nkiyohara/srvp-fd/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/srvp-fd.svg)](https://badge.fury.io/py/srvp-fd)
[![Python Versions](https://img.shields.io/pypi/pyversions/srvp-fd.svg)](https://pypi.org/project/srvp-fd/)
[![License](https://img.shields.io/github/license/nkiyohara/srvp-fd.svg)](https://github.com/nkiyohara/srvp-fd/blob/main/LICENSE)

A package for calculating Fréchet distance between video frames using the encoder from the Stochastic Latent Residual Video Prediction (SRVP) model.

## Installation

```bash
pip install srvp-fd  # Using pip
uv pip install srvp-fd  # Using uv
```

## Basic Usage

```python
import torch
from srvp_fd import frechet_distance

# Load image tensors
# Shape: [batch_size, channels, height, width]
images1 = torch.randn(512, 1, 64, 64)  # Replace with your images
images2 = torch.randn(512, 1, 64, 64)  # Replace with your images

# Calculate Fréchet distance
fd = frechet_distance(images1, images2)
print(f"Fréchet distance: {fd}")

# Specify different dataset
# Options: "mmnist_stochastic", "mmnist_deterministic", "bair", "kth", "human"
fd_bair = frechet_distance(images1, images2, dataset="bair")
print(f"Fréchet distance (BAIR): {fd_bair}")
```

## Advanced Usage

### Video Comparison Types

The package now supports three types of comparisons for video sequences:

```python
import torch
from srvp_fd import frechet_distance

# Load video tensors
# Shape: [batch_size, seq_length, channels, height, width]
videos1 = torch.randn(512, 16, 1, 64, 64)  # Replace with your videos
videos2 = torch.randn(512, 16, 1, 64, 64)  # Replace with your videos

# 1. Frame-wise comparison (spatial patterns)
fd_frame = frechet_distance(videos1[:, 0], videos2[:, 0], comparison_type="frame")

# 2. Static content comparison (scene/object appearance)
fd_static = frechet_distance(videos1, videos2, comparison_type="static_content")

# 3. Dynamics comparison (motion patterns)
fd_dynamics = frechet_distance(videos1, videos2, comparison_type="dynamics")

print(f"Frame Fréchet distance: {fd_frame}")
print(f"Static content Fréchet distance: {fd_static}")
print(f"Dynamics Fréchet distance: {fd_dynamics}")
```

### Class-based API for Efficiency

```python
import torch
from srvp_fd import FrechetDistanceCalculator

# Create calculator (loads model once)
calculator = FrechetDistanceCalculator(dataset="mmnist_stochastic")

# Calculate multiple Fréchet distances efficiently
fd1 = calculator(images1, images2)
fd2 = calculator(images1, images3)

# Extract and reuse features
features1 = calculator.extract_features(images1)
features2 = calculator.extract_features(images2)
fd = calculator._calculate_frechet_distance_from_features(features1, features2)

# Extract video-specific features
w1 = calculator.extract_w(videos1)  # Static content
w2 = calculator.extract_w(videos2)
q_y_0_params1 = calculator.extract_q_y_0_params(videos1)  # Dynamics
q_y_0_params2 = calculator.extract_q_y_0_params(videos2)
```

## Features

- **Multiple comparison types**: Analyze frame-wise features, static content, or motion dynamics
- **Pre-trained models**: Automatically downloads models from HuggingFace Hub
- **Supported datasets**: Moving MNIST (stochastic/deterministic), BAIR, KTH, Human3.6M
- **Efficient API**: Class-based implementation for multiple calculations
- **Device support**: Works on both CPU and GPU
- **Numerically stable**: Implementation includes safeguards for computation stability

## About Fréchet Distance

The Fréchet distance measures similarity between two probability distributions. For video analysis, it compares feature distributions to evaluate generated content quality.

The distance is calculated as:
```
d²((m₁, C₁), (m₂, C₂)) = ||m₁ - m₂||² + Tr(C₁ + C₂ - 2√(C₁C₂))
```
Where m₁,m₂ are means, C₁,C₂ are covariances, and √(C₁C₂) is the matrix square root of the product.

## Citation

If you use this package, please cite the original SRVP paper:

```
@inproceedings{franceschi2020stochastic,
  title={Stochastic Latent Residual Video Prediction},
  author={Franceschi, Jean-Yves and Delasalles, Edouard and Chen, Mickael and Lamprier, Sylvain and Gallinari, Patrick},
  booktitle={International Conference on Machine Learning},
  pages={3233--3246},
  year={2020},
  organization={PMLR}
}
```

## License

Apache License 2.0, the same as the original SRVP implementation.

## Acknowledgements

This package builds upon the work of the SRVP authors: Jean-Yves Franceschi, Edouard Delasalles, Mickael Chen, Sylvain Lamprier, and Patrick Gallinari.

- [SRVP GitHub Repository](https://github.com/edouardelasalles/srvp)
- [SRVP Project Website](https://sites.google.com/view/srvp/)
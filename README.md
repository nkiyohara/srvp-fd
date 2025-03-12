# SRVP-FD

[![Python Package](https://github.com/nkiyohara/srvp-fd/actions/workflows/python-package.yml/badge.svg)](https://github.com/nkiyohara/srvp-fd/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/srvp-fd.svg)](https://badge.fury.io/py/srvp-fd)
[![Python Versions](https://img.shields.io/pypi/pyversions/srvp-fd.svg)](https://pypi.org/project/srvp-fd/)
[![License](https://img.shields.io/github/license/nkiyohara/srvp-fd.svg)](https://github.com/nkiyohara/srvp-fd/blob/main/LICENSE)

A package for calculating Fréchet distance between video frames using the encoder from the Stochastic Latent Residual Video Prediction (SRVP) model.

## Overview

This package provides a simple interface to calculate the Fréchet distance between two sets of video frames. It uses the encoder from the SRVP model to extract features from the images, and then calculates the Fréchet distance between the feature distributions.

The Fréchet distance is a measure of similarity between two probability distributions. In the context of image generation, it is often used to evaluate the quality of generated images by comparing their feature distribution with the feature distribution of real images.

## Installation

```bash
# Using pip
pip install srvp-fd

# Using uv
uv pip install srvp-fd
```

## Usage

```python
import torch
from srvp_fd import frechet_distance

# Load your image tensors
# images1 and images2 should be torch tensors with shape [batch_size, channels, height, width]
# For example, for Moving MNIST: [batch_size, 1, 64, 64]
images1 = torch.randn(512, 1, 64, 64)  # Replace with your actual images
images2 = torch.randn(512, 1, 64, 64)  # Replace with your actual images

# Calculate Fréchet distance using the stochastic Moving MNIST model (default)
fd = frechet_distance(images1, images2)
print(f"Fréchet distance: {fd}")

# You can specify a different dataset
# Options: "mmnist_stochastic", "mmnist_deterministic", "bair", "kth", "human"
fd_bair = frechet_distance(images1, images2, dataset="bair")
print(f"Fréchet distance (BAIR): {fd_bair}")

# You can also specify a different model path or device
# fd = frechet_distance(images1, images2, model_path='path/to/your/model.pt', device='cpu')
```

### Class-based API

For more efficient usage when calculating multiple Fréchet distances, you can use the class-based API:

```python
import torch
from srvp_fd import FrechetDistanceCalculator

# Load your image tensors
images1 = torch.randn(512, 1, 64, 64)  # Replace with your actual images
images2 = torch.randn(512, 1, 64, 64)  # Replace with your actual images
images3 = torch.randn(512, 1, 64, 64)  # Replace with your actual images

# Create a calculator (loads the model once)
calculator = FrechetDistanceCalculator(dataset="mmnist_stochastic")

# Calculate multiple Fréchet distances efficiently
fd1 = calculator(images1, images2)
fd2 = calculator(images1, images3)

print(f"Fréchet distance 1: {fd1}")
print(f"Fréchet distance 2: {fd2}")

# For even more efficiency, you can extract features once and reuse them
features1 = calculator.extract_features(images1)
features2 = calculator.extract_features(images2)
features3 = calculator.extract_features(images3)

# Calculate Fréchet distances from pre-extracted features
fd1 = calculator.calculate_frechet_distance_from_features(features1, features2)
fd2 = calculator.calculate_frechet_distance_from_features(features1, features3)
```

## Features

- Supports multiple datasets: Moving MNIST (stochastic and deterministic), BAIR, KTH, and Human3.6M
- Automatically downloads the pre-trained SRVP model from HuggingFace Hub
- Supports both CPU and GPU computation
- Simple and easy-to-use interface
- Works with any PyTorch tensor of the correct shape
- Warns when using models with skip connections that might affect feature extraction

## Fréchet Distance Calculation

The Fréchet distance (also known as Fréchet Inception Distance or FID when used with Inception features) is a measure of similarity between two probability distributions. In the context of image generation, it is often used to evaluate the quality of generated images by comparing their feature distribution with the feature distribution of real images.

### Mathematical Formula

The Fréchet distance between two multivariate Gaussian distributions is calculated as:

$$d^2((m_1, C_1), (m_2, C_2)) = ||m_1 - m_2||_2^2 + \text{Tr}(C_1 + C_2 - 2\sqrt{C_1 C_2})$$

Where:
- $m_1, m_2$ are the mean feature vectors of the two distributions
- $C_1, C_2$ are the covariance matrices of the feature vectors
- $\text{Tr}$ is the trace operator
- $\sqrt{C_1 C_2}$ is the matrix square root of the product of the covariance matrices

### Implementation Details

Our implementation follows these steps:

1. **Feature Extraction**: The images are passed through the encoder part of the SRVP model to extract meaningful features.
2. **Distribution Estimation**: The mean and covariance of the features are calculated to estimate the distribution.
3. **Distance Calculation**: The Fréchet distance is calculated using the formula above.

The implementation includes safeguards against numerical instability, such as adding a small offset to the covariance matrices when they are not positive definite.

### Supported Datasets

The package supports the following datasets:

- **mmnist_stochastic**: Stochastic Moving MNIST (default)
- **mmnist_deterministic**: Deterministic Moving MNIST
- **bair**: BAIR robot pushing dataset
- **kth**: KTH human actions dataset
- **human**: Human3.6M dataset

Each dataset has its own pre-trained SRVP model, which is automatically downloaded from the HuggingFace Hub when needed.

### Skip Connections Warning

Some SRVP models use skip connections between the encoder and decoder. When calculating the Fréchet distance, we only use the encoder part of the model. If the model was trained with skip connections, the encoder might not capture all the necessary information, as some of it was meant to be passed directly to the decoder through the skip connections.

The package will issue a warning when using a model with skip connections, recommending to use a model without skip connections for more accurate results.

## Pre-trained Model Details

This package utilizes the pre-trained encoder from the SRVP (Stochastic Latent Residual Video Prediction) model, which was **developed and trained by the original SRVP authors** (Franceschi et al., 2020). The encoder is a convolutional neural network trained on various video datasets to extract meaningful features from images.

**Important Note**: The pre-trained weights used in this package are the original weights created by the SRVP authors (Franceschi, Delasalles, Chen, Lamprier, and Gallinari). This package serves as a convenient interface to their work, making it easier to use their sophisticated models for Fréchet distance calculations.

### Model Architecture

The SRVP model uses a sophisticated architecture designed by the original authors:
- The encoder is based on a DCGAN-style convolutional network
- It extracts a 128-dimensional feature vector from each frame
- These features capture the essential characteristics of the input video frames
- Different model variants are trained specifically for each supported dataset

### Model Weights and Availability

The pre-trained model weights are hosted on HuggingFace Hub as a mirror of the original weights and are automatically downloaded when you first use the package. These weights represent years of research and development by the original SRVP team and are made available here with proper attribution.

The weights are mirrored on HuggingFace Hub to ensure reliable and fast access, but all intellectual credit belongs to the original SRVP authors.

## Citation

If you use this package in your research, please cite the original SRVP paper:

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

This package is licensed under the Apache License 2.0, the same as the original SRVP implementation.

## Acknowledgements

This package is fundamentally built upon the groundbreaking work of the SRVP authors: **Jean-Yves Franceschi, Edouard Delasalles, Mickael Chen, Sylvain Lamprier, and Patrick Gallinari**. Their research and implementation form the core of this package.

**Key contributions from the original authors**:
- Development of the SRVP architecture and training methodology
- Training of high-quality models on multiple video datasets
- Release of pre-trained weights that enable state-of-the-art feature extraction
- Open-sourcing their implementation under an Apache 2.0 license

This package simply provides a convenient interface to leverage their encoder for calculating Fréchet distances between video distributions. The significant research contribution and intellectual property of the models belong entirely to the original SRVP authors.

For more information about their work:
- [SRVP GitHub Repository](https://github.com/edouardelasalles/srvp)
- [SRVP Project Website](https://sites.google.com/view/srvp/)
- [SRVP Paper](http://proceedings.mlr.press/v119/franceschi20a.html)

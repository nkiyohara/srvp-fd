# SRVP-MMNIST-FD

[![Python Package](https://github.com/nkiyohara/srvp-mmnist-fd/actions/workflows/python-package.yml/badge.svg)](https://github.com/nkiyohara/srvp-mmnist-fd/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/srvp-mmnist-fd.svg)](https://badge.fury.io/py/srvp-mmnist-fd)
[![Python Versions](https://img.shields.io/pypi/pyversions/srvp-mmnist-fd.svg)](https://pypi.org/project/srvp-mmnist-fd/)
[![License](https://img.shields.io/github/license/nkiyohara/srvp-mmnist-fd.svg)](https://github.com/nkiyohara/srvp-mmnist-fd/blob/main/LICENSE)

A package for calculating Fréchet distance between Moving MNIST images using the encoder from the Stochastic Latent Residual Video Prediction (SRVP) model.

## Overview

This package provides a simple interface to calculate the Fréchet distance between two sets of Moving MNIST images. It uses the encoder from the SRVP model to extract features from the images, and then calculates the Fréchet distance between the feature distributions.

The Fréchet distance is a measure of similarity between two probability distributions. In the context of image generation, it is often used to evaluate the quality of generated images by comparing their feature distribution with the feature distribution of real images.

## Installation

```bash
# Using pip
pip install srvp-mmnist-fd

# Using uv
uv pip install srvp-mmnist-fd
```

## Usage

```python
import torch
from srvp_mmnist_fd import frechet_distance

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

This package utilizes the pre-trained encoder from the SRVP (Stochastic Latent Residual Video Prediction) model, which was developed and trained by the original SRVP authors (Franceschi et al., 2020). The encoder is a convolutional neural network specifically trained on the Stochastic Moving MNIST dataset to extract meaningful features from images.

The pre-trained weights are automatically downloaded from the HuggingFace Hub repository when you first use the package. These weights are the original weights trained by the SRVP authors and are used with their permission.

### Model Architecture

The SRVP model uses a sophisticated architecture:
- The encoder is based on a DCGAN-style convolutional network
- It extracts a 128-dimensional feature vector from each frame
- These features capture the essential characteristics of the Moving MNIST digits

### Model Weights

The pre-trained model weights are hosted on HuggingFace Hub and are automatically downloaded when you first use the package. The weights represent the intellectual property of the original SRVP authors and are used in this package with proper attribution.

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

This package is based on the [SRVP](https://github.com/edouardelasalles/srvp) implementation by Jean-Yves Franceschi, Edouard Delasalles, Mickael Chen, Sylvain Lamprier, and Patrick Gallinari. The pre-trained model weights and architecture are the work of these authors, and this package simply provides a convenient interface to use their encoder for calculating Fréchet distances.

- [SRVP GitHub Repository](https://github.com/edouardelasalles/srvp)
- [SRVP Project Website](https://sites.google.com/view/srvp/)
- [SRVP Paper](http://proceedings.mlr.press/v119/franceschi20a.html)

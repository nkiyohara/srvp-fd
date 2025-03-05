"""Fréchet distance calculation module for Moving MNIST images.

This module provides functions to calculate the Fréchet distance between two sets of
Moving MNIST images using the encoder from the SRVP model to extract features.
"""

import json
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy import linalg

# Import the SRVP model components
from .srvp_model import StochasticLatentResidualVideoPredictor


def _calculate_frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """Calculate Fréchet Distance between two multivariate Gaussians.

    Args:
        mu1: Mean of the first Gaussian distribution
        sigma1: Covariance matrix of the first Gaussian distribution
        mu2: Mean of the second Gaussian distribution
        sigma2: Covariance matrix of the second Gaussian distribution

    Returns:
        Fréchet distance between the two distributions
    """
    # Calculate squared difference between means
    diff = mu1 - mu2

    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate Fréchet distance
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def _get_model_and_config(
    model_path: Optional[str] = None,
) -> Tuple[StochasticLatentResidualVideoPredictor, dict]:
    """Load the SRVP model and its configuration.

    Args:
        model_path: Path to the model file. If None, the model will be downloaded from HuggingFace.

    Returns:
        Tuple of (model, config)
    """
    # Default HuggingFace repository and filenames
    repo_id = "nkiyohara/srvp-mmnist-fd"
    model_filename = "model.pt"
    config_filename = "config.json"
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "srvp-mmnist-fd")

    if model_path is None:
        # Try to download from HuggingFace
        try:
            # Download the model
            model_path = hf_hub_download(
                repo_id=repo_id, filename=model_filename, cache_dir=cache_dir
            )
            print(f"Successfully downloaded model from {model_filename}")

            # Download the config
            config_path = hf_hub_download(
                repo_id=repo_id, filename=config_filename, cache_dir=cache_dir
            )
            print(f"Successfully downloaded config from {config_filename}")

            # Load config
            with open(config_path) as f:
                config = json.load(f)
        except Exception as e:
            print(f"Failed to download model: {e}")
            raise FileNotFoundError(
                "Could not download the model from HuggingFace. "
                "Please provide a local model_path or check the repository structure."
            ) from e
    else:
        # If model_path is provided, look for config in the same directory
        model_dir = os.path.dirname(model_path)
        config_path = os.path.join(model_dir, config_filename)

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found at {config_path}. "
                f"Please ensure {config_filename} is in the same directory as the model."
            )

        # Load config
        with open(config_path) as f:
            config = json.load(f)

    # Load model
    model = StochasticLatentResidualVideoPredictor(
        nx=config["nx"],
        nc=config["nc"],
        nf=config["nf"],
        nhx=config["nhx"],
        ny=config["ny"],
        nz=config["nz"],
        skipco=config["skipco"],
        nt_inf=config["nt_inf"],
        nh_inf=config["nh_inf"],
        nlayers_inf=config["nlayers_inf"],
        nh_res=config["nh_res"],
        nlayers_res=config["nlayers_res"],
        archi=config["archi"],
    )

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    return model, config


def frechet_distance(
    images1: torch.Tensor,
    images2: torch.Tensor,
    model_path: Optional[str] = None,
    device: Union[str, torch.device] = None,
) -> float:
    """Calculate the Fréchet distance between two sets of images using the SRVP encoder.

    Args:
        images1: First set of images, tensor of shape [batch_size, channels, height, width]
        images2: Second set of images, tensor of shape [batch_size, channels, height, width]
        model_path: Path to the SRVP model file. If None, the model will be downloaded from
            HuggingFace.
        device: Device to run the model on.

    Returns:
        Fréchet distance between the two sets of images
    """
    # Set default device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Validate input dimensions
    if images1.dim() != 4 or images2.dim() != 4:
        raise ValueError(
            "Images must be 4-dimensional tensors [batch_size, channels, height, width]"
        )

    if images1.shape[1:] != images2.shape[1:]:
        raise ValueError(
            f"Image dimensions must match. Got {images1.shape[1:]} and {images2.shape[1:]}"
        )

    # Load model and config
    model, _ = _get_model_and_config(model_path)
    model = model.to(device)
    model.eval()

    # Extract features using the encoder
    with torch.no_grad():
        # Move images to the device
        images1 = images1.to(device)
        images2 = images2.to(device)

        # Extract features
        features1 = []
        features2 = []

        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, images1.shape[0], batch_size):
            batch1 = images1[i : i + batch_size]
            # Add a time dimension (required by the encoder)
            batch1 = batch1.unsqueeze(1)  # [batch, 1, channels, height, width]
            # Extract features
            feat1 = model.encode(batch1)
            # Remove time dimension and append to list
            features1.append(feat1.squeeze(0))

        for i in range(0, images2.shape[0], batch_size):
            batch2 = images2[i : i + batch_size]
            # Add a time dimension (required by the encoder)
            batch2 = batch2.unsqueeze(1)  # [batch, 1, channels, height, width]
            # Extract features
            feat2 = model.encode(batch2)
            # Remove time dimension and append to list
            features2.append(feat2.squeeze(0))

        # Concatenate features
        features1 = torch.cat(features1, dim=0)
        features2 = torch.cat(features2, dim=0)

        # Convert to numpy arrays
        features1 = features1.cpu().numpy()
        features2 = features2.cpu().numpy()

    # Calculate mean and covariance
    mu1 = np.mean(features1, axis=0)
    sigma1 = np.cov(features1, rowvar=False)

    mu2 = np.mean(features2, axis=0)
    sigma2 = np.cov(features2, rowvar=False)

    # Calculate Fréchet distance
    return _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

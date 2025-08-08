"""Fréchet distance calculation module for various video datasets.

This module provides functions to calculate the Fréchet distance between two sets of
video frames using the encoder from the SRVP model to extract features.
"""

import json
import os
import warnings
from typing import Literal, Tuple, Union

import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import hf_hub_download

# Import the SRVP model components
from .srvp_model import StochasticLatentResidualVideoPredictor

# Define dataset options as Literal type
DatasetType = Literal["mmnist_stochastic", "mmnist_deterministic", "bair", "kth", "human"]

# Map dataset names to their paths in the repository
DATASET_PATHS = {
    "mmnist_stochastic": "mmnist/stochastic",
    "mmnist_deterministic": "mmnist/deterministic",
    "bair": "bair",
    "kth": "kth",
    "human": "human",
}


def _matrix_sqrt(matrix: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Compute the square root of a positive‑definite matrix.

    Args:
        matrix: A positive‑definite matrix.
        eps: Small epsilon value for numerical stability.

    Returns:
        The matrix square root.
    """
    # Add a small value to the diagonal for numerical stability
    matrix = matrix + torch.eye(matrix.size(0), device=matrix.device) * eps

    # Compute the eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

    # Compute the square root using the eigendecomposition
    sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=eps))
    return eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T


def _calculate_frechet_distance(
    mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor
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
    covmean = _matrix_sqrt(sigma1 @ sigma2)

    # If covmean has any non-finite values, retry with a diagonal offset
    if not torch.isfinite(covmean).all():
        offset = torch.eye(sigma1.size(0), device=sigma1.device) * 1e-6
        covmean = _matrix_sqrt((sigma1 + offset) @ (sigma2 + offset))

    # Calculate Fréchet distance
    tr_covmean = torch.trace(covmean)
    fd = float((diff @ diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean)

    return max(fd, 0.0)  # Ensure non-negative result due to numerical precision


def _get_model(dataset: DatasetType) -> Tuple[StochasticLatentResidualVideoPredictor, dict]:
    """Load the SRVP model and its configuration.

    Args:
        model_path: Path to the model file. If None, the model will be downloaded from HuggingFace.
        dataset: The dataset to use. Required if model_path is None.
            Options: "mmnist_stochastic", "mmnist_deterministic", "bair", "kth", "human"

    Returns:
        A tuple containing the model and its configuration.

    Raises:
        ValueError: If dataset is None when model_path is None.
        FileNotFoundError: If the model or config file cannot be found.
    """
    # Get the dataset path
    dataset_path = DATASET_PATHS[dataset]

    # Download the model and config from HuggingFace Hub
    try:
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "srvp-fd")
        os.makedirs(cache_dir, exist_ok=True)
        # Download the config first
        config_path = hf_hub_download(
            repo_id="nkiyohara/srvp-pretrained-model-mirror",
            filename=f"{dataset_path}/config.json",
            cache_dir=cache_dir,
            force_download=False,
        )
        print(f"Successfully downloaded config from {config_path}")
        model_path = hf_hub_download(
            repo_id="nkiyohara/srvp-pretrained-model-mirror",
            filename=f"{dataset_path}/model.pt",
            cache_dir=cache_dir,
            force_download=False,
        )
        print(f"Successfully downloaded model from {model_path}")

        # Load config
        with open(config_path) as f:
            config = json.load(f)

        # Check if skipco is True and issue a warning
        if config.get("skipco", False):
            warnings.warn(
                f"The model for dataset '{dataset}' uses skip connections (skipco=True). "
                "This may affect the quality of the Fréchet distance calculation, "
                "as skip connections can bypass the encoder's feature extraction. "
                "Consider using a model without skip connections for more accurate results.",
                UserWarning,
                stacklevel=2,
            )

        # Create a dummy model to hold the encoder
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

        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    except Exception as e:
        print(f"Failed to download or load model: {e}")
        raise FileNotFoundError(
            f"Could not download or load the model for dataset '{dataset}' from HuggingFace. "
            "Please check your internet connection or provide a local model_path."
        ) from e


def _validate_input_shapes(images1: torch.Tensor, images2: torch.Tensor) -> None:
    """Validate the shapes of the input tensors.

    Args:
        images1: First set of images.
        images2: Second set of images.

    Raises:
        ValueError: If the input shapes are invalid.
    """
    # Check dimensions
    if images1.dim() != 4 or images2.dim() != 4:
        raise ValueError(
            f"Input tensors must be 4D (batch, channels, height, width). "
            f"Got shapes {images1.shape} and {images2.shape}."
        )

    # Check channel dimensions match
    if images1.shape[1] != images2.shape[1]:
        raise ValueError(
            f"Channel dimensions must match. Got {images1.shape[1]} and {images2.shape[1]}."
        )

    # Check spatial dimensions match
    if images1.shape[2:] != images2.shape[2:]:
        raise ValueError(
            f"Spatial dimensions must match. Got {images1.shape[2:]} and {images2.shape[2:]}."
        )

    # Check that sample size is greater than 128 (feature dimension)
    if images1.shape[0] <= 128 or images2.shape[0] <= 128:
        raise ValueError(
            f"Sample size must be greater than 128 (feature dimension). "
            f"Got {images1.shape[0]} and {images2.shape[0]}."
        )


def _validate_video_input_shapes(videos1: torch.Tensor, videos2: torch.Tensor, model=None) -> None:
    """Validate the shapes of the input video tensors.

    Args:
        videos1: First set of videos.
        videos2: Second set of videos.
        model: Optional model to get nt_inf parameter, otherwise default to 10.

    Raises:
        ValueError: If the input shapes are invalid.
    """
    # Check dimensions
    if videos1.dim() != 5 or videos2.dim() != 5:
        raise ValueError(
            f"Input tensors must be 5D (batch, seq_length, channels, height, width). "
            f"Got shapes {videos1.shape} and {videos2.shape}."
        )

    # Check channel dimensions match
    if videos1.shape[2] != videos2.shape[2]:
        raise ValueError(
            f"Channel dimensions must match. Got {videos1.shape[2]} and {videos2.shape[2]}."
        )

    # Check spatial dimensions match
    if videos1.shape[3:] != videos2.shape[3:]:
        raise ValueError(
            f"Spatial dimensions must match. Got {videos1.shape[3:]} and {videos2.shape[3:]}."
        )

    # Check that sample size is greater than 128 (feature dimension)
    if videos1.shape[0] <= 128 or videos2.shape[0] <= 128:
        raise ValueError(
            f"Sample size must be greater than 128 (feature dimension). "
            f"Got {videos1.shape[0]} and {videos2.shape[0]}."
        )

    # Check that sequence length is at least 10 frames
    nt_inf = (
        getattr(model, "nt_inf", 10) if model is not None else 10
    )  # Default to 10 if not specified
    if videos1.shape[1] < nt_inf or videos2.shape[1] < nt_inf:
        raise ValueError(
            f"Sequence length should be at least {nt_inf} frames for model inference. "
            f"Got {videos1.shape[1]} and {videos2.shape[1]}."
        )


class FrechetDistanceCalculator:
    """A class for calculating Fréchet distance between sets of images or videos.

    This class loads the SRVP model once during initialization and
    can be reused for multiple Fréchet distance calculations, avoiding
    repeated model loading.

    Attributes:
        model: The SRVP model used for feature extraction.
        device: The device used for computation.
    """

    def __init__(
        self,
        dataset: DatasetType = "mmnist_stochastic",
        device: Union[str, torch.device] = None,
    ):
        """Initialize the Fréchet distance calculator.

        Args:
            dataset: The dataset to use for feature extraction. Required if model_path is None.
                Options: "mmnist_stochastic", "mmnist_deterministic", "bair", "kth", "human"
            device: Device to use for computation. If None, will use CUDA if available,
                otherwise CPU.
        """
        # Get the device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Get the model
        self.model = _get_model(dataset)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    def __call__(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
        comparison_type: Literal["frame", "static_content", "dynamics"] = "frame",
        batch_size: int = None,
    ) -> float:
        """Calculate the Fréchet distance between two sets of images or videos.

        Args:
            images1: First set of images/videos.
                For "frame" comparison: Shape [batch_size, channels, height, width]
                For "static_content"/"dynamics" comparisons:
                    Shape [batch_size, seq_length, channels, height, width]
            images2: Second set of images/videos with same shape requirements as images1.
            comparison_type: The type of Fréchet distance to calculate:
                - "frame": Compare frame-wise visual features from encoder (spatial patterns)
                - "static_content": Compare static content information (w) that
                    captures scene/object appearance
                - "dynamics": Compare dynamics information (q_y_0) that captures motion patterns
            batch_size: Optional batch size for processing large datasets. If None, processes
                all data at once. Use this to reduce GPU memory usage for large datasets.

        Returns:
            The Fréchet distance between the two sets.

        Raises:
            ValueError: If the input shapes are invalid or comparison_type is unrecognized.
        """
        if comparison_type == "frame":
            # Validate input shapes for frame comparison_type
            _validate_input_shapes(images1, images2)

            # Extract features (with optional batching)
            if batch_size is None or batch_size >= images1.shape[0]:
                with torch.no_grad():
                    features1 = self.model.encoder(images1.to(self.device))
                    features2 = self.model.encoder(images2.to(self.device))
            else:
                # Process in batches
                def extract_frame_features(batch):
                    return self.model.encoder(batch.to(self.device))

                features1 = self._extract_features_in_batches(
                    images1, batch_size, extract_frame_features
                )
                features2 = self._extract_features_in_batches(
                    images2, batch_size, extract_frame_features
                )

            # Calculate Fréchet distance
            return self._calculate_frechet_distance_from_features(features1, features2)

        if comparison_type in ["static_content", "dynamics"]:
            # Validate video input shapes
            _validate_video_input_shapes(images1, images2, self.model)

            # Extract w or q_y_0_params (with optional batching)
            if comparison_type == "static_content":
                if batch_size is None or batch_size >= images1.shape[0]:
                    features1 = self._extract_w(images1)
                    features2 = self._extract_w(images2)
                else:
                    features1 = self._extract_features_in_batches(
                        images1, batch_size, self._extract_w
                    )
                    features2 = self._extract_features_in_batches(
                        images2, batch_size, self._extract_w
                    )
                return self._calculate_frechet_distance_from_features(features1, features2)

            # comparison_type == "dynamics"
            if batch_size is None or batch_size >= images1.shape[0]:
                q_y_0_params1 = self._extract_q_y_0_params(images1)
                q_y_0_params2 = self._extract_q_y_0_params(images2)
            else:
                q_y_0_params1 = self._extract_features_in_batches(
                    images1, batch_size, self._extract_q_y_0_params
                )
                q_y_0_params2 = self._extract_features_in_batches(
                    images2, batch_size, self._extract_q_y_0_params
                )
            return self._calculate_frechet_distance_from_gaussian_params(
                q_y_0_params1, q_y_0_params2
            )

        raise ValueError(
            f"Unrecognized comparison_type '{comparison_type}'. Must be one of: "
            "'frame', 'static_content', 'dynamics'"
        )

    def _extract_w(self, videos: torch.Tensor) -> torch.Tensor:
        """Extract static content information (w) from videos.

        Args:
            videos: Input videos of shape [batch_size, seq_length, channels, height, width]

        Returns:
            Tensor of w features with shape [batch_size, feature_dim]
        """
        # Permute to [seq_len, batch_size, channels, height, width]
        videos_permuted = videos.permute(1, 0, 2, 3, 4)

        with torch.no_grad():
            # Encode frames
            hx, _ = self.model.encode(videos_permuted.to(self.device))
            # Extract static content w
            return self.model.infer_w(hx)

    def _extract_features_in_batches(
        self, images: torch.Tensor, batch_size: int, extraction_fn
    ) -> torch.Tensor:
        """Extract features in batches to handle memory constraints.

        Args:
            images: Input tensor to process
            batch_size: Size of each processing batch
            extraction_fn: Function to extract features from a batch

        Returns:
            Concatenated features from all batches
        """
        features_list = []
        num_samples = images.shape[0]

        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch = images[i:end_idx]
            with torch.no_grad():
                batch_features = extraction_fn(batch)
            features_list.append(batch_features)

        return torch.cat(features_list, dim=0)

    def _extract_q_y_0_params(self, videos: torch.Tensor) -> torch.Tensor:
        """Extract dynamics information (q_y_0_params) from videos.

        Args:
            videos: Input videos of shape [batch_size, seq_length, channels, height, width]

        Returns:
            Tensor of q_y_0_params with shape [batch_size, 2*ny]
        """
        # Permute to [seq_len, batch_size, channels, height, width]
        videos_permuted = videos.permute(1, 0, 2, 3, 4)

        with torch.no_grad():
            # Encode frames
            hx, _ = self.model.encode(videos_permuted.to(self.device))
            # Extract dynamics parameters
            _, q_y_0_params = self.model.infer_y(hx[: self.model.nt_inf])
            return q_y_0_params

    def _calculate_frechet_distance_from_features(
        self, features1: torch.Tensor, features2: torch.Tensor
    ) -> float:
        """Calculate Fréchet distance from features.

        Args:
            features1: First set of features. Shape: [batch_size, feature_dim]
            features2: Second set of features. Shape: [batch_size, feature_dim]

        Returns:
            The Fréchet distance between the two sets of features.
        """
        # Calculate mean and covariance
        mu1 = torch.mean(features1, dim=0)
        # Covariance calculation
        centered1 = features1 - mu1.unsqueeze(0)
        sigma1 = (centered1.T @ centered1) / (features1.size(0) - 1)

        mu2 = torch.mean(features2, dim=0)
        centered2 = features2 - mu2.unsqueeze(0)
        sigma2 = (centered2.T @ centered2) / (features2.size(0) - 1)

        # Calculate Fréchet distance
        return _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    def _calculate_frechet_distance_from_gaussian_params(
        self, params1: torch.Tensor, params2: torch.Tensor
    ) -> float:
        """Calculate Fréchet distance from Gaussian mixture parameters.

        For q_y_0_params, each sample in the batch represents a Gaussian distribution,
        making the batch a Gaussian mixture model. We use moment matching to compute
        the mean and covariance of this mixture.

        Args:
            params1: First set of Gaussian parameters. Shape: [batch_size, 2*ny]
            params2: Second set of Gaussian parameters. Shape: [batch_size, 2*ny]

        Returns:
            The Fréchet distance between the two Gaussian mixtures.
        """
        # Split into means and raw scales
        ny = params1.shape[1] // 2
        mu1_samples = params1[:, :ny]  # Shape: [batch_size, ny]
        raw_scale1_samples = params1[:, ny:]  # Shape: [batch_size, ny]

        mu2_samples = params2[:, :ny]
        raw_scale2_samples = params2[:, ny:]

        # Process raw_scale with softplus to get scale (standard deviation)
        # This matches the SRVP utils.py implementation
        # Use the same eps value as in the original implementation
        eps = 1e-8
        scale1_samples = F.softplus(raw_scale1_samples) + eps  # standard deviation
        scale2_samples = F.softplus(raw_scale2_samples) + eps  # standard deviation

        # Convert to variance for covariance calculation
        var1_samples = scale1_samples**2
        var2_samples = scale2_samples**2

        # Moment matching for the first mixture
        # Mean of the mixture is the average of the component means
        mu1 = torch.mean(mu1_samples, dim=0)  # Shape: [ny]

        # Covariance of the mixture combines component covariances and means
        # Cov = E[Cov] + Cov[E]
        # E[Cov] is average of component covariances
        # Cov[E] is covariance of component means

        # Average of component variances (diagonal covariance matrices)
        avg_var1 = torch.mean(var1_samples, dim=0)
        e_cov1 = torch.diag(avg_var1)

        # Covariance of component means
        centered_mu1 = mu1_samples - mu1.unsqueeze(0)
        cov_e1 = (centered_mu1.T @ centered_mu1) / (mu1_samples.size(0) - 1)
        sigma1 = e_cov1 + cov_e1

        # Repeat for the second mixture
        mu2 = torch.mean(mu2_samples, dim=0)
        avg_var2 = torch.mean(var2_samples, dim=0)
        e_cov2 = torch.diag(avg_var2)

        centered_mu2 = mu2_samples - mu2.unsqueeze(0)
        cov_e2 = (centered_mu2.T @ centered_mu2) / (mu2_samples.size(0) - 1)
        sigma2 = e_cov2 + cov_e2

        # Calculate Fréchet distance between the two Gaussian mixtures
        return _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from a set of images.

        This method can be used to extract features from images for later use,
        which can be useful when you want to compare multiple sets of images
        against a reference set.

        Args:
            images: Set of images. Shape: [batch_size, channels, height, width]

        Returns:
            Tensor of features with shape [batch_size, feature_dim]
        """
        # Validate input shape
        if not isinstance(images, torch.Tensor):
            raise ValueError("Images must be a torch.Tensor")
        if len(images.shape) != 4:
            raise ValueError(f"Images must have 4 dimensions, got {len(images.shape)}")

        # Extract features
        with torch.no_grad():
            return self.model.encoder(images.to(self.device))

    def extract_w(self, videos: torch.Tensor) -> torch.Tensor:
        """Extract static content information (w) from videos.

        Args:
            videos: Input videos. Shape: [batch_size, seq_length, channels, height, width]

        Returns:
            Tensor of w features with shape [batch_size, feature_dim]
        """
        _validate_video_input_shapes(videos, videos, self.model)  # Validate shape with itself
        return self._extract_w(videos)

    def extract_q_y_0_params(self, videos: torch.Tensor) -> torch.Tensor:
        """Extract dynamics information (q_y_0_params) from videos.

        Args:
            videos: Input videos. Shape: [batch_size, seq_length, channels, height, width]

        Returns:
            Tensor of q_y_0_params with shape [batch_size, 2*ny]
        """
        _validate_video_input_shapes(videos, videos, self.model)  # Validate shape with itself
        return self._extract_q_y_0_params(videos)


def frechet_distance(
    images1: torch.Tensor,
    images2: torch.Tensor,
    dataset: DatasetType = "mmnist_stochastic",
    comparison_type: Literal["frame", "static_content", "dynamics"] = "frame",
    device: Union[str, torch.device] = None,
    batch_size: int = None,
) -> float:
    """Calculate the Fréchet distance between two sets of images or videos.

    Args:
        images1: First set of images/videos.
            For "frame" comparison: Shape [batch_size, channels, height, width]
            For "static_content"/"dynamics" comparisons:
                Shape [batch_size, seq_length, channels, height, width]
        images2: Second set of images/videos with same shape requirements as images1.
        dataset: The dataset to use for feature extraction.
            Options: "mmnist_stochastic", "mmnist_deterministic", "bair", "kth", "human"
        comparison_type: The type of Fréchet distance to calculate:
            - "frame": Compare frame-wise visual features from encoder (spatial patterns)
            - "static_content": Compare static content information (w) that
                captures scene/object appearance
            - "dynamics": Compare dynamics information (q_y_0) that captures motion patterns
        device: Device to use for computation. If None, will use CUDA if available, otherwise CPU.
        batch_size: Optional batch size for processing large datasets. If None, processes
            all data at once. Use this to reduce GPU memory usage for large datasets.

    Returns:
        The Fréchet distance between the two sets.

    Raises:
        ValueError: If the input shapes are invalid or comparison_type is unrecognized.
    """
    calculator = FrechetDistanceCalculator(dataset=dataset, device=device)
    return calculator(images1, images2, comparison_type=comparison_type, batch_size=batch_size)

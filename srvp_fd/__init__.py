"""Fréchet distance calculator for video datasets using SRVP encoder.

This package provides a simple interface to calculate the Fréchet distance
between two sets of video frames, using the encoder from the SRVP model
to extract features.
"""

from .frechet_distance import (
    DATASET_PATHS,
    DatasetType,
    FrechetDistanceCalculator,
    frechet_distance,
)

__all__ = ["frechet_distance", "FrechetDistanceCalculator", "DatasetType", "DATASET_PATHS"]

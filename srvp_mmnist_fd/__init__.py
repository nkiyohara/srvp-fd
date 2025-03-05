"""Fréchet distance calculator for Moving MNIST images using SRVP encoder.

This package provides a simple interface to calculate the Fréchet distance
between two sets of Moving MNIST images, using the encoder from the SRVP model
to extract features.
"""

from .frechet_distance import frechet_distance

__all__ = ["frechet_distance"]

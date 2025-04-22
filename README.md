# SRVP‑FD

[![Python Package](https://github.com/nkiyohara/srvp-fd/actions/workflows/python-package.yml/badge.svg)](https://github.com/nkiyohara/srvp-fd/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/srvp-fd.svg)](https://badge.fury.io/py/srvp-fd)
[![Python Versions](https://img.shields.io/pypi/pyversions/srvp-fd.svg)](https://pypi.org/project/srvp-fd/)
[![License](https://img.shields.io/github/license/nkiyohara/srvp-fd.svg)](https://github.com/nkiyohara/srvp-fd/blob/main/LICENSE)

`SRVP‑FD` computes **Fréchet distance** between images or videos using the encoder of the  
**Stochastic Latent Residual Video Prediction (SRVP)** model.

---

## Installation

```bash
pip install srvp-fd          # via pip
uv pip install srvp-fd       # via uv (faster resolver)
```

---

## Usage

```python
import torch
from srvp_fd import frechet_distance, FrechetDistanceCalculator

# Images – shape: [batch, channels, height, width]
images1 = torch.randn(512, 1, 64, 64)
images2 = torch.randn(512, 1, 64, 64)

# Basic image comparison
fd = frechet_distance(images1, images2)
print(fd)

# Using a different pretrained encoder
fd_bair = frechet_distance(images1, images2, dataset="bair")

# Videos – shape: [batch, seq_len, channels, height, width]
videos1 = torch.randn(512, 16, 1, 64, 64)
videos2 = torch.randn(512, 16, 1, 64, 64)

# Different comparison types
fd_frame    = frechet_distance(videos1[:, 0], videos2[:, 0], comparison_type="frame")
fd_static   = frechet_distance(videos1, videos2, comparison_type="static_content")
fd_dynamics = frechet_distance(videos1, videos2, comparison_type="dynamics")

# Class-based API (weights loaded only once at initialization - more efficient for multiple calculations)
calc = FrechetDistanceCalculator(dataset="mmnist_stochastic")

fd1 = calc(images1, images2)                           # frame by default
fd2 = calc(videos1, videos2, comparison_type="static_content")
fd3 = calc(videos1, videos2, comparison_type="dynamics")
```

### Comparison Types

| `comparison_type` | Latent signal (SRVP notation) | Captures |
|-------------------|--------------------------------|----------|
| `"frame"` | Per‑frame embedding $`\tilde{\mathbf x}_t = h_\phi(\mathbf{x}_t)`$ | Appearance of a **single** frame (no temporal context) |
| `"static_content"` | Global content vector $`\mathbf{w} = c_\psi(\tilde{\mathbf{x}}_{i_1}, …, \tilde{\mathbf{x}}_{i_k})`$ pooled from the first *k* conditioning frames | Scene or object identity that remains constant throughout the clip |
| `"dynamics"` | Parameters $`(\boldsymbol{\mu}_\theta,\boldsymbol{\sigma}_\theta)`$ of the initial latent‑state distribution that seeds the residual dynamics $`f_\theta`$ | Motion patterns and stochastic variation over time |

*For precise mathematical definitions consult [SRVP paper](https://arxiv.org/abs/2002.09219) (Franceschi&nbsp;*et al.*, 2020) and the&nbsp;[official implementation](https://github.com/edouardelasalles/srvp).*

> **Gaussian‑mixture approximation**: For the `"dynamics"` comparison we collapse the batch of per‑sample Gaussians into a single Gaussian using the closed‑form mean and covariance of an equal‑weighted mixture for the Fréchet distance computation.

## Features

* Multiple comparison modes (frame / static content / dynamics)  
* Pre‑trained weights downloaded automatically from **HuggingFace Hub**  
* Supported datasets: Moving MNIST, BAIR, KTH, Human3.6M  
* Single‑load, class‑based API for high‑throughput evaluation  
* CPU **and** GPU compatible  
* Covariance maths guarded for numerical stability  

---

## Fréchet distance formula

$$
d^2\bigl((m_1,\,C_1),(m_2,\,C_2)\bigr)
= \lVert m_1 - m_2\rVert^2 + \mathrm{Tr}\bigl(C_1 + C_2 - 2\sqrt{C_1C_2}\bigr)
$$

where $`m`$ are the feature means and $`C`$ the covariances.

---

## Citation

If you use **SRVP‑FD** in your research or publications, please cite the original SRVP paper, which provides the foundation for this tool's encoder and latent representations:

*This package builds on the encoder from the original SRVP model to define a set of Fréchet-based distances for evaluating image and video generation models.*

```bibtex
@inproceedings{franceschi2020stochastic,
  title   = {Stochastic Latent Residual Video Prediction},
  author  = {Franceschi, Jean-Yves and Delasalles, Edouard and Chen, Mickael
             and Lamprier, Sylvain and Gallinari, Patrick},
  booktitle = {International Conference on Machine Learning},
  pages   = {3233--3246},
  year    = {2020},
  organization = {PMLR}
}
```

---

## License

Apache License 2.0 – same as the original SRVP implementation.

---

## Acknowledgements

Built on the excellent work of the SRVP authors:  
Jean‑Yves Franceschi · Edouard Delasalles · Mickaël Chen · Sylvain Lamprier · Patrick Gallinari  

* <https://github.com/edouardelasalles/srvp>  
* <https://sites.google.com/view/srvp/>

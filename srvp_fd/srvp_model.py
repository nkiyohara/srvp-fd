"""SRVP - Stochastic Latent Residual Video Prediction.

=================================================

This file is **adapted** from the official SRVP implementation
(https://github.com/edouardelasalles/srvp), which is released under the
Apache License 2.0.

Original authors of the SRVP paper and code
-------------------------------------------
* Jean-Yves Franceschi
* Edouard Delasalles
* Mickaël Chen
* Sylvain Lamprier
* Patrick Gallinari

Copyright (c) 2019-2020 — the original SRVP authors
Modifications © 2025 — Naoki Kiyohara (and contributors).

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at <http://www.apache.org/licenses/LICENSE-2.0>.  Unless required by
applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the specific language
governing permissions and limitations under the License.

Summary of local modifications
------------------------------
* Consolidated encoder and decoder definitions into a single module for feature
  extraction / Fréchet-distance evaluation.
* Applied Ruff/Flake8 formatting and stylistic fixes.
* Added extensive docstrings and typing hints.

This module now provides the SRVP back-bones used by
`srvp_fd.frechet_distance` to extract per-frame, static-content and dynamics
features.
"""

import torch
import torch.distributions as distrib
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


def make_conv_block(conv, activation, bn=True):
    """Supplements a convolutional block with activation functions and batch normalization."""
    if activation == "relu":
        act = nn.ReLU(inplace=True)
    elif activation == "leaky_relu":
        act = nn.LeakyReLU(0.2, inplace=True)
    elif activation == "tanh":
        act = nn.Tanh()
    elif activation == "sigmoid":
        act = nn.Sigmoid()
    else:
        raise ValueError(f"Activation {activation} not supported")

    if bn:
        bn = nn.BatchNorm2d(conv.out_channels)
        return nn.Sequential(conv, bn, act)
    return nn.Sequential(conv, act)


def make_lin_block(n_inp, n_out, activation):
    """Creates a linear block formed by an activation function and a linear operation.

    Parameters
    ----------
    n_inp : int
        Input dimension.
    n_out : int
        Output dimension.
    activation : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', 'tanh', or 'none'. Adds the corresponding
        activation function, or no activation if 'none' is chosen, before the linear
        operation.

    Returns:
    -------
    torch.nn.Sequential
        Sequence of the potentially chosen activation function and the input linear block.
    """
    modules = []
    if activation != "none":
        modules.append(activation_factory(activation))
    modules.append(nn.Linear(n_inp, n_out))
    return nn.Sequential(*modules)


def activation_factory(name):
    """Returns the activation layer corresponding to the input activation name.

    Parameters
    ----------
    name : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', or 'tanh'. Adds the corresponding activation
        function after the convolution.

    Returns:
    -------
    torch.nn.Module
        Element-wise activation layer.
    """
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Activation function '{name}' not yet implemented")


def encoder_factory(name, _, nc, nh, nf):
    """Creates an encoder with the given parameters according the input architecture name."""
    if name == "dcgan":
        return DCGAN64Encoder(nc, nh, nf)
    if name == "vgg":
        return VGG64Encoder(nc, nh, nf)
    raise ValueError(f"Architecture {name} not supported")


def decoder_factory(name, _nx, nc, ny, nf, skip=False):
    """Creates a decoder with the given parameters according the input architecture name.

    Parameters
    ----------
    name : str
        'dcgan' or 'vgg'. Name of the architecture to use.
    _nx : int
        Width and height of the video frames (unused).
    nc : int
        Number of channels in the output shape.
    ny : int
        Number of dimensions of the input flat vector.
    nf : int
        Number of filters per channel of the first convolution of the mirror encoder
        architecture.

    Returns:
    -------
    module.conv.BaseDecoder
        Either a module.conv.DCGAN64Decoder or a module.conv.VGG64Decoder depending on
        the chosen architecture.
    """
    if name == "dcgan":
        return DCGAN64Decoder(nc, ny, nf, skip)
    if name == "vgg":
        return VGG64Decoder(nc, ny, nf, skip)
    raise ValueError(f"No decoder named '{name}'")


def make_normal_from_raw_params(raw_params, scale_stddev=1, dim=-1, eps=1e-8):
    """Creates a normal distribution from the given parameters.

    Parameters
    ----------
    raw_params : torch.*.Tensor
        Tensor containing the Gaussian mean and a raw scale parameter on a given dimension.
    scale_stddev : float
        Multiplier of the final scale parameter of the Gaussian.
    dim : int
        Dimensions of raw_params so that the first half corresponds to the mean, and the
        second half to the scale.
    eps : float
        Minimum possible value of the final scale parameter.

    Returns:
    -------
    torch.distributions.Normal
        Normal distribution with the input mean and eps + softplus(raw scale) * scale_stddev
        as standard deviation.
    """
    loc, raw_scale = torch.chunk(raw_params, 2, dim)
    assert loc.shape[dim] == raw_scale.shape[dim]
    scale = F.softplus(raw_scale) + eps
    return distrib.Normal(loc, scale * scale_stddev)


def rsample_normal(raw_params, scale_stddev=1):
    """Samples from a normal distribution with given parameters.

    Parameters
    ----------
    raw_params : torch.*.Tensor
        Tensor containing a Gaussian mean and a raw scale parameter on its last dimension.
    scale_stddev : float
        Multiplier of the final scale parameter of the Gaussian.

    Returns:
    -------
    torch.*.Tensor
        Sample from the normal distribution with the input mean and eps + softplus(raw scale)
        * scale_stddev as standard deviation.
    """
    return make_normal_from_raw_params(raw_params, scale_stddev=scale_stddev).rsample()


class BaseEncoder(nn.Module):
    """Module implementing the encoders forward method.

    Attributes:
    ----------
    nh : int
        Number of dimensions of the output flat vector.
    """

    def __init__(self, nh):
        """Initialize BaseEncoder.

        Parameters
        ----------
        nh : int
            Number of dimensions of the output flat vector.
        """
        super().__init__()
        self.nh = nh

    def forward(self, x, return_skip=False):
        """Forward pass of the encoder.

        Parameters
        ----------
        x : torch.*.Tensor
            Encoder input.
        return_skip : bool
            Whether to extract and return, besides the network output, skip connections.

        Returns:
        -------
        torch.*.Tensor
            Encoder output as a tensor of shape (batch, size).
        list
            Only if ``return_skip`` is ``True``. List of skip connections represented as
            ``torch.*.Tensor`` corresponding to each convolutional block in reverse order
            (from the deepest to the shallowest convolutional block).
        """
        skips = []
        h = x
        for layer in self.conv:
            h = layer(h)
            skips.append(h)
        h = self.last_conv(h).view(-1, self.nh)
        if return_skip:
            return h, skips[::-1]
        return h


class DCGAN64Encoder(BaseEncoder):
    """Module implementing the DCGAN encoder."""

    def __init__(self, nc, nh, nf):
        """Initialize DCGAN64Encoder.

        Parameters
        ----------
        nc : int
            Number of channels in the input data.
        nh : int
            Number of dimensions of the output flat vector.
        nf : int
            Number of filters per channel of the first convolution.
        """
        super().__init__(nh)
        self.conv = nn.ModuleList(
            [
                make_conv_block(
                    nn.Conv2d(nc, nf, 4, 2, 1, bias=False), activation="leaky_relu", bn=False
                ),
                make_conv_block(
                    nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False), activation="leaky_relu"
                ),
                make_conv_block(
                    nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False), activation="leaky_relu"
                ),
                make_conv_block(
                    nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False), activation="leaky_relu"
                ),
            ]
        )
        self.last_conv = make_conv_block(
            nn.Conv2d(nf * 8, nh, 4, 1, 0, bias=False), activation="tanh"
        )


class VGG64Encoder(BaseEncoder):
    """Module implementing the VGG encoder."""

    def __init__(self, nc, nh, nf):
        """Initialize VGG64Encoder.

        Parameters
        ----------
        nc : int
            Number of channels in the input data.
        nh : int
            Number of dimensions of the output flat vector.
        nf : int
            Number of filters per channel of the first convolution.
        """
        super().__init__(nh)
        self.conv = nn.ModuleList(
            [
                nn.Sequential(
                    make_conv_block(
                        nn.Conv2d(nc, nf, 3, 1, 1, bias=False), activation="leaky_relu"
                    ),
                    make_conv_block(
                        nn.Conv2d(nf, nf, 3, 1, 1, bias=False), activation="leaky_relu"
                    ),
                ),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    make_conv_block(
                        nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False), activation="leaky_relu"
                    ),
                    make_conv_block(
                        nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=False), activation="leaky_relu"
                    ),
                ),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    make_conv_block(
                        nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False), activation="leaky_relu"
                    ),
                    make_conv_block(
                        nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=False), activation="leaky_relu"
                    ),
                    make_conv_block(
                        nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=False), activation="leaky_relu"
                    ),
                ),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    make_conv_block(
                        nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False), activation="leaky_relu"
                    ),
                    make_conv_block(
                        nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False), activation="leaky_relu"
                    ),
                    make_conv_block(
                        nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False), activation="leaky_relu"
                    ),
                ),
            ]
        )
        self.last_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            make_conv_block(nn.Conv2d(nf * 8, nh, 4, 1, 0, bias=False), activation="tanh"),
        )


class BaseDecoder(nn.Module):
    """Module implementing the decoders forward method.

    Attributes:
    ----------
    ny : int
        Number of dimensions of the output flat vector.
    skip : bool
        Whether to include skip connections into the decoder.
    """

    def __init__(self, ny, skip):
        """Initialize BaseDecoder.

        Parameters
        ----------
        ny : int
            Number of dimensions of the input flat vector.
        """
        super().__init__()
        self.ny = ny
        self.skip = skip

    def forward(self, z, skip=None, sigmoid=True):
        """Forward pass of the decoder.

        Parameters
        ----------
        z : torch.*.Tensor
            Decoder input.
        skip : list
            List of ``torch.*.Tensor`` representing skip connections in the same order as the
            decoder convolutional blocks. Must be ``None`` when skip connections are not allowed.
        sigmoid : bool
            Whether to apply a sigmoid at the end of the decoder.

        Returns:
        -------
        torch.*.Tensor
            Decoder output as a frame of shape (batch, channels, width, height).
        """
        assert skip is None and not self.skip or self.skip and skip is not None
        h = self.first_upconv(z.view(*z.shape, 1, 1))
        for i, layer in enumerate(self.conv):
            if skip is not None:
                h = torch.cat([h, skip[i]], 1)
            h = layer(h)
        x_ = h
        if sigmoid:
            x_ = torch.sigmoid(x_)
        return x_


class DCGAN64Decoder(BaseDecoder):
    """Module implementing the DCGAN decoder."""

    def __init__(self, nc, ny, nf, skip):
        """Initialize DCGAN64Decoder.

        Parameters
        ----------
        nc : int
            Number of channels in the output shape.
        ny : int
            Number of dimensions of the input flat vector.
        nf : int
            Number of filters per channel of the first convolution of the mirror encoder
            architecture.
        skip : list
            List of ``torch.*.Tensor`` representing skip connections in the same order as the
            decoder convolutional blocks. Must be ``None`` when skip connections are not allowed.
        """
        super().__init__(ny, skip)
        # decoder
        coef = 2 if skip else 1
        self.first_upconv = make_conv_block(
            nn.ConvTranspose2d(ny, nf * 8, 4, 1, 0, bias=False), activation="leaky_relu"
        )
        self.conv = nn.ModuleList(
            [
                make_conv_block(
                    nn.ConvTranspose2d(nf * 8 * coef, nf * 4, 4, 2, 1, bias=False),
                    activation="leaky_relu",
                ),
                make_conv_block(
                    nn.ConvTranspose2d(nf * 4 * coef, nf * 2, 4, 2, 1, bias=False),
                    activation="leaky_relu",
                ),
                make_conv_block(
                    nn.ConvTranspose2d(nf * 2 * coef, nf, 4, 2, 1, bias=False),
                    activation="leaky_relu",
                ),
                nn.ConvTranspose2d(nf * coef, nc, 4, 2, 1, bias=False),
            ]
        )


class VGG64Decoder(BaseDecoder):
    """Module implementing the VGG decoder."""

    def __init__(self, nc, ny, nf, skip):
        """Initialize VGG64Decoder.

        Parameters
        ----------
        nc : int
            Number of channels in the output shape.
        ny : int
            Number of dimensions of the input flat vector.
        nf : int
            Number of filters per channel of the first convolution of the mirror encoder
            architecture.
        skip : list
            List of ``torch.*.Tensor`` representing skip connections in the same order as the
            decoder convolutional blocks. Must be ``None`` when skip connections are not allowed.
        """
        super().__init__(ny, skip)
        # decoder
        coef = 2 if skip else 1
        self.first_upconv = nn.Sequential(
            make_conv_block(
                nn.ConvTranspose2d(ny, nf * 8, 4, 1, 0, bias=False), activation="leaky_relu"
            ),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.conv = nn.ModuleList(
            [
                nn.Sequential(
                    make_conv_block(
                        nn.Conv2d(nf * 8 * coef, nf * 8, 3, 1, 1, bias=False),
                        activation="leaky_relu",
                    ),
                    make_conv_block(
                        nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False), activation="leaky_relu"
                    ),
                    make_conv_block(
                        nn.Conv2d(nf * 8, nf * 4, 3, 1, 1, bias=False), activation="leaky_relu"
                    ),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                ),
                nn.Sequential(
                    make_conv_block(
                        nn.Conv2d(nf * 4 * coef, nf * 4, 3, 1, 1, bias=False),
                        activation="leaky_relu",
                    ),
                    make_conv_block(
                        nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=False), activation="leaky_relu"
                    ),
                    make_conv_block(
                        nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=False), activation="leaky_relu"
                    ),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                ),
                nn.Sequential(
                    make_conv_block(
                        nn.Conv2d(nf * 2 * coef, nf * 2, 3, 1, 1, bias=False),
                        activation="leaky_relu",
                    ),
                    make_conv_block(
                        nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=False), activation="leaky_relu"
                    ),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                ),
                nn.Sequential(
                    make_conv_block(
                        nn.Conv2d(nf * coef, nf, 3, 1, 1, bias=False), activation="leaky_relu"
                    ),
                    nn.ConvTranspose2d(nf, nc, 3, 1, 1, bias=False),
                ),
            ]
        )


class MLP(nn.Module):
    """Module implementing an MLP."""

    def __init__(self, n_inp, n_hid, n_out, n_layers, activation="relu"):
        """Initialize MLP.

        Parameters
        ----------
        n_inp : int
            Input dimension.
        n_hid : int
            Number of dimensions in intermediary layers.
        n_out : int
            Output dimension.
        n_layers : int
            Number of layers in the MLP.
        activation : str
            'relu', 'leaky_relu', 'elu', 'sigmoid', or 'tanh'. Adds the corresponding activation
            function before every linear operation but the first one.
        """
        super().__init__()
        assert n_hid == 0 or n_layers > 1
        modules = [
            make_lin_block(
                n_inp if il == 0 else n_hid,
                n_out if il == n_layers - 1 else n_hid,
                activation if il > 0 else "none",
            )
            for il in range(n_layers)
        ]
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        """Output of the MLP.

        Parameters
        ----------
        x : torch.*.Tensor
            Input of shape (batch, n_inp).

        Returns:
        -------
        torch.*.Tensor
            Output of shape (batch, n_out).
        """
        return self.module(x)


class StochasticLatentResidualVideoPredictor(nn.Module):
    """SRVP model. Please refer to the paper."""

    def __init__(
        self,
        nx,
        nc,
        nf,
        nhx,
        ny,
        nz,
        skipco,
        nt_inf,
        nh_inf,
        nlayers_inf,
        nh_res,
        nlayers_res,
        archi,
    ):
        """Initialize the SRVP model.

        Parameters
        ----------
        nx : int
            Width and height of the video frames.
        nc : int
            Number of channels in the video data.
        nf : int
            Number of filters per channel in the first convolution of the encoder.
        nhx : int
            Size of frames encoding (dimension of the encoder output).
        ny : int
            Number of dimensions of y (state space variable).
        nz : int
            Number of dimensions of z (auxiliary stochastic variable).
        skipco : bool
            Whether to include skip connections into the decoder.
        nt_inf : int
            Number of timesteps used to infer y_1 and to compute the content variable.
        nh_inf : int
            Size of inference MLP hidden layers.
        nlayers_inf : int
            Number of layers in inference MLPs.
        nh_res : int
            Size of residual MLP hidden layers.
        nlayers_res : int
            Number of layers in residual MLPs.
        archi : str
            Architecture to use for the encoder and decoder.
        """
        super().__init__()
        self.nx = nx
        self.nc = nc
        self.nf = nf
        self.nhx = nhx
        self.ny = ny
        self.nz = nz
        self.skipco = skipco
        self.nt_inf = nt_inf
        self.nh_inf = nh_inf
        self.nlayers_inf = nlayers_inf
        self.nh_res = nh_res
        self.nlayers_res = nlayers_res
        self.archi = archi

        self.encoder = encoder_factory(archi, nx, nc, nhx, nf)
        self.decoder = decoder_factory(archi, nx, nc, nh_inf + ny, nf, skipco)
        # -- Content
        self.w_proj = nn.Sequential(nn.Linear(nhx, nh_inf), nn.ReLU(inplace=True))
        self.w_inf = nn.Sequential(nn.Linear(nh_inf, nh_inf), nn.Tanh())
        # -- Inference of y
        self.q_y = MLP(n_inp=nhx * nt_inf, n_hid=nh_inf, n_out=ny * 2, n_layers=nlayers_inf)
        # -- Inference of z
        self.inf_z = nn.LSTM(nhx, nh_inf, 1)
        self.q_z = nn.Linear(nh_inf, nz * 2)
        # -- Prior
        self.p_z = MLP(n_inp=ny, n_hid=nh_res, n_out=nz * 2, n_layers=nlayers_res)
        # -- Prediction
        self.dynamics = MLP(
            n_inp=ny + nz, n_hid=nh_res, n_out=ny, n_layers=nlayers_res
        )  # Residual function

    def encode(self, x):
        """Frame-wise encoding of a sequence of images. Returns the encodings and skip connections.

        Parameters
        ----------
        x : torch.*.Tensor
            Video / sequence of images to encode, of shape (length, batch, channels, width, height).

        Returns:
        -------
        torch.*.Tensor
            Encoding of frames, of shape (length, batch, nhx), where length is the number of frames.
        list
            List of ``torch.*.Tensor`` representing skip connections. Must be ``None`` when
            skip connections are not allowed. Skip connections are extracted from the last frame
            in testing mode, and from a random frame during training.
        """
        nt, bsz, x_shape = x.shape[0], x.shape[1], x.shape[2:]
        # Flatten the temporal dimension (for convolution encoders)
        x_flat = x.reshape(nt * bsz, *x_shape)
        # Encode
        hx_flat, skips = self.encoder(x_flat, return_skip=True)
        # Reshape with temporal dimension
        hx = hx_flat.view(nt, bsz, self.nhx)
        # Skip connections
        if self.skipco:
            if self.training:
                # When training, take a random frame to compute the skip connections
                t = torch.randint(nt, size=(bsz,)).to(hx.device)
                index = torch.arange(bsz).to(hx.device)
                skips = [s.view(nt, bsz, *s.shape[1:])[t, index] for s in skips]
            else:
                # When testing, choose the last frame
                skips = [s.view(nt, bsz, *s.shape[1:])[-1] for s in skips]
        else:
            skips = None
        return hx, skips

    def infer_w(self, hx):
        """Computes the content variable from the data with a permutation-invariant network.

        Parameters
        ----------
        hx : torch.*.Tensor
            Encoding of frames, of shape (length, batch, nhx), where length is the number of frames.

        Returns:
        -------
        torch.*.Tensor
            Output sequence of frames, of shape (length, batch, channels, width, height).
        """
        nt, bsz = hx.shape[0], hx.shape[1]
        if self.training:
            # When training, pick w conditioning on random frames
            t = torch.stack([torch.randperm(nt)[: self.nt_inf] for _ in range(bsz)], 1).to(
                hx.device
            )
            index = torch.arange(bsz).repeat(self.nt_inf, 1).to(hx.device)
            h = hx[t.view(-1), index.view(-1)].view(self.nt_inf, bsz, self.nhx)
        else:
            # Otherwise, choose the last nt_inf random frames
            h = hx[-self.nt_inf :]
        # Permutation-invariant appplication
        h = self.w_proj(h)
        h = h.sum(0)
        return self.w_inf(h)

    def infer_y(self, hx):
        """Infers y_0 (first state variable) from the data.

        Parameters
        ----------
        hx : torch.*.Tensor
            Encoding of frames, of shape (length, batch, nhx), where ``length`` is the number
            of conditioning frames used to infer ``y_0``.

        Returns:
        -------
        torch.*.Tensor
            Initial state condition y_0, of shape (batch, ny).
        torch.*.Tensor
            Gaussian parameters of the approximate posterior for the initial state condition
            ``y_0``, of shape ``(batch, 2 * ny)``.
        """
        q_y_0_params = self.q_y(hx.permute(1, 0, 2).reshape(hx.shape[1], self.nt_inf * self.nhx))
        y_0 = rsample_normal(q_y_0_params)
        return y_0, q_y_0_params

    def infer_z(self, hx):
        """Infers a z variable from the data.

        Parameters
        ----------
        hx : torch.*.Tensor
            Encoding of frame t, of shape (batch, nhx), so that z is inferred from timestep t.

        Returns:
        -------
        torch.*.Tensor
            Inferred variable z, of shape (batch, nz).
        torch.*.Tensor
            Gaussian parameters of the approximate posterior of z, of shape (nt - 1, batch, 2 * nz).
        """
        q_z_params = self.q_z(hx)
        z = rsample_normal(q_z_params)
        return z, q_z_params

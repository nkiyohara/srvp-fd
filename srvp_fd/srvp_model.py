"""SRVP model implementation for feature extraction.

This module contains the implementation of the SRVP model encoder used for feature extraction
in the Fréchet distance calculation.
"""

import torch.nn as nn


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


class BaseEncoder(nn.Module):
    """Module implementing the encoders forward method."""

    def __init__(self, nh):
        """Initialize the base encoder.

        Parameters
        ----------
        nh : int
            Number of dimensions of the output flat vector.
        """
        super().__init__()
        self.nh = nh

    def forward(self, x, return_skip=False):
        """Process input through the encoder.

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
            Only if return_skip is True. List of skip connections represented as torch.*.Tensor
            corresponding to each convolutional block in reverse order (from the deepest to the
            shallowest convolutional block).
        """
        skips = []
        h = x

        for layer in self.conv:
            h = layer(h)
            skips.append(h)

        h = self.last_conv(h)
        h = h.view(-1, self.nh)

        if return_skip:
            return h, skips
        return h


class DCGAN64Encoder(BaseEncoder):
    """Module implementing the DCGAN encoder."""

    def __init__(self, nc, nh, nf):
        """Initialize the DCGAN encoder.

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
        """Initialize the VGG encoder.

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
            nn.Flatten(),
            nn.Linear(4 * 4 * nf * 8, nh),
            nn.Tanh(),
        )


def encoder_factory(name, _, nc, nh, nf):
    """Creates an encoder with the given parameters according the input architecture name."""
    if name == "dcgan":
        return DCGAN64Encoder(nc, nh, nf)
    if name == "vgg":
        return VGG64Encoder(nc, nh, nf)
    raise ValueError(f"Architecture {name} not supported")


class MLP(nn.Module):
    """Simple MLP with ReLU activations."""

    def __init__(self, nin, nout, nh=512, nlayers=3):
        """Initialize the MLP.

        Args:
            nin: Number of input dimensions
            nout: Number of output dimensions
            nh: Number of hidden dimensions
            nlayers: Number of hidden layers
        """
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.nh = nh
        self.nlayers = nlayers

        if nlayers == 0:
            self.net = nn.Linear(nin, nout)
        else:
            net = []
            for i in range(nlayers):
                if i == 0:
                    net.append(nn.Linear(nin, nh))
                    net.append(nn.ReLU(inplace=True))
                elif i == nlayers - 1:
                    net.append(nn.Linear(nh, nout))
                else:
                    net.append(nn.Linear(nh, nh))
                    net.append(nn.ReLU(inplace=True))
            self.net = nn.Sequential(*net)

    def forward(self, x):
        """Forward pass through the MLP."""
        return self.net(x)


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

        # Create encoder
        self.encoder = encoder_factory(archi, nx, nc, nhx, nf)

        # For the purpose of this package, we only need the encoder part
        # The rest of the model is not implemented here as we only need the encoder
        # for feature extraction

    def encode(self, x):
        """Encode a batch of frames.

        Parameters
        ----------
        x : torch.*.Tensor
            Tensor of shape (batch, length, nc, nx, nx) containing a batch of videos.

        Returns:
        -------
        torch.*.Tensor
            Encoding of frames, of shape (length, batch, nhx), where length is the number of frames.
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        # Reshape to (batch * seq_len, nc, nx, nx)
        x_reshape = x.reshape(-1, self.nc, self.nx, self.nx)
        # Encode
        h = self.encoder(x_reshape)
        # Reshape back to (seq_len, batch, nhx)
        h = h.reshape(batch_size, seq_len, self.nhx)
        h = h.permute(1, 0, 2)
        return h.clone()

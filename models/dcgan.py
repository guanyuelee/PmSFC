import torch
from models import BaseAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import numpy as np


class Autoencoder(BaseAE):
    # input: [N, input_size, input_size, in_channels]
    # hidden: [N, hidden_size, hidden_size, latent_channels]
    def __init__(self, input_size: int, in_channels: int, hidden_size: int,
                 latent_channels: int, channels: int) -> None:
        super(Autoencoder, self).__init__()

        self.depth = int(np.round(np.log(input_size // hidden_size) / np.log(2)))
        self.input_size = input_size
        self.hidden_size = hidden_size
        print('Switching input_size from %d to hidden_size %d needs %d depth.' % (input_size, hidden_size, self.depth))

        modules = []
        if channels is None:
            channels = 64

        hidden_dims = []
        for i in range(self.depth - 1):
            hidden_dims.append(channels)
        hidden_dims.append(latent_channels)

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    # add additional conv block make it deeper while keep the input output the same.
                    nn.Conv2d(h_dim, out_channels=h_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.encoder_parameters = [param for param in self.encoder.parameters()]

        # Build Decoder
        modules = []

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                    # add additional conv block while keep the input output the same
                    nn.Conv2d(hidden_dims[i + 1], out_channels=hidden_dims[i + 1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=hidden_dims[-1], kernel_size=3, padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=1, kernel_size=1))
        self.decoder_parameters = [param for param in self.decoder.parameters()] + \
                                 [param for param in self.final_layer.parameters()]

    def encode(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution

        return result

    def decode(self, z: Tensor):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs):
        hidden = self.encode(input)
        return self.decode(hidden)

    def get_parameters(self):
        return self.encoder_parameters + self.decoder_parameters


class Discriminator(BaseAE):
    # input: [N, input_size, input_size, in_channels]
    # hidden: [N, hidden_size, hidden_size, latent_channels]
    def __init__(self, input_size: int, in_channels: int, hidden_size: int,
                 latent_channels: int, channels: int) -> None:
        super(Discriminator, self).__init__()

        self.depth = int(np.round(np.log(input_size // hidden_size) / np.log(2)))
        self.input_size = input_size
        self.hidden_size = hidden_size
        print('In Discriminator: ')
        print('Switching input_size from %d to hidden_size %d needs %d depth.' % (input_size, hidden_size, self.depth))

        modules = []
        if channels is None:
            channels = 64

        hidden_dims = []
        for i in range(self.depth - 1):
            hidden_dims.append(channels)
        hidden_dims.append(latent_channels)

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    # add additional conv block make it deeper while keep the input output the same.
                    nn.Conv2d(h_dim, out_channels=h_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.encoder_parameters = [param for param in self.encoder.parameters()]

    def encode(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution

        return result

    def decode(self, z: Tensor):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        raise NotImplemented('Discriminator does not support decode method. ')

    def forward(self, input: Tensor, **kwargs):
        hidden = self.encode(input)
        return hidden

    def get_parameters(self):
        return self.encoder_parameters


class Encoder(nn.Module):
    def __init__(self, input_size: int, in_channels: int, hidden_size: int,
                 latent_channels: int, channels: int, with_classify=False, n_classes=None):
        super(Encoder, self).__init__()

        self.depth = int(np.round(np.log(input_size // hidden_size) / np.log(2)))
        self.input_size = input_size
        self.hidden_size = hidden_size
        print('In Discriminator: ')
        print('Switching input_size from %d to hidden_size %d needs %d depth.' % (input_size, hidden_size, self.depth))

        modules = []
        if channels is None:
            channels = 64

        hidden_dims = []
        for i in range(self.depth - 1):
            hidden_dims.append(channels)
        hidden_dims.append(latent_channels)

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    # add additional conv block make it deeper while keep the input output the same.
                    nn.Conv2d(h_dim, out_channels=h_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.with_classify = with_classify
        self.n_classes = n_classes
        if with_classify:
            self.classify_layer = nn.Linear(in_features=hidden_size * hidden_size * latent_channels,
                                            out_features=n_classes)
        else:
            self.classify_layer = None

        self.encoder_parameters = [param for param in self.encoder.parameters()]

        if with_classify:
            self.encoder_parameters += [param for param in self.classify_layer.parameters()]

    def forward(self, input: Tensor, **kwargs):
        hidden = self.encoder(input)
        if self.with_classify:
            hidden = torch.reshape(hidden, shape=[hidden.size(0), -1])
            hidden = self.classify_layer(hidden)
        return hidden

    def get_parameters(self):
        return self.encoder_parameters



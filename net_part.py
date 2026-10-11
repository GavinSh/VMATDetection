from .net_block import *


class EncoderMultConv(nn.Module):
    def __init__(
            self,
            in_features: int,
            latent_dim: int = 3,
            hidden_dim: list = None,
            image_shape: int = None,
            conv_dim: int = None
    ):

        super().__init__()
        self.image_shape = image_shape
        self.de_image_shape = self.image_shape
        if hidden_dim is None:
            hidden_dim = [64, 32, 16, 8, 4]
        assert len(hidden_dim) > 0, 'Number of conv kernel should be more than 0!'
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = conv_dim
        self.layers = []

        assert self.conv_dim in [2, 3], 'Dimension of the conv kernel should be 2 or 3!'
        if self.conv_dim == 2:
            DownConv = DownConv2d
        else:
            DownConv = DownConv3d

        self.layers.append(
            DownConv(in_features, hidden_dim[0], in_features)
        )
        self.de_image_shape /= 2

        for dim_id in range(len(hidden_dim) - 1):
            self.layers.append(
                DownConv(hidden_dim[dim_id], hidden_dim[dim_id + 1], hidden_dim[dim_id])
            )
            self.de_image_shape /= 2

        self.layers = nn.Sequential(*self.layers)
        self.de_image_shape = int(self.de_image_shape)
        self.linear_in_dim = self.de_image_shape * self.de_image_shape * self.hidden_dim[-1] if self.conv_dim == 2 \
            else self.de_image_shape * self.de_image_shape * self.de_image_shape * self.hidden_dim[-1]

        self.linear = nn.Sequential(
            Linear(self.linear_in_dim, self.latent_dim),
        )

    def forward(self, x):
        # define forward pass through the network
        x = self.layers(x)
        x = nn.Flatten()(x)
        x = self.linear(x)
        return x


class DecoderMultConv(nn.Module):
    def __init__(
            self,
            out_features: int,
            latent_dim: int = 3,
            hidden_dim: list = None,
            image_shape: int = None,
            conv_dim: int = None
    ):
        super().__init__()
        self.image_shape = image_shape
        self.de_image_shape = self.image_shape
        if hidden_dim is None:
            hidden_dim = [4, 8, 16, 32, 64]
        assert len(hidden_dim) > 0, 'number of dim should be more than 0'
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = conv_dim

        assert self.conv_dim in [2, 3], 'Dimension of the conv kernel should be 2 or 3!'
        if self.conv_dim == 2:
            UpConv = UpConv2d
            UpConvEnd = UpConv2dEnd
        else:
            UpConv = UpConv3d
            UpConvEnd = UpConv3dEnd

        self.layers = []

        for dim_id in range(len(hidden_dim) - 1):
            self.layers.append(
                UpConv(hidden_dim[dim_id], hidden_dim[dim_id + 1], hidden_dim[dim_id + 1])
            )
            self.de_image_shape /= 2

        self.layers.append(
            UpConvEnd(hidden_dim[-1], out_features, out_features)
        )
        self.de_image_shape /= 2

        self.layers = nn.Sequential(*self.layers)
        self.de_image_shape = int(self.de_image_shape)

        self.linear_out_dim = self.de_image_shape * self.de_image_shape * self.hidden_dim[0] if self.conv_dim == 2 \
            else self.de_image_shape * self.de_image_shape * self.de_image_shape * self.hidden_dim[0]

        self.conv_in_shape = [self.de_image_shape] * conv_dim

        self.linear = nn.Sequential(
            Linear(self.latent_dim, self.linear_out_dim),
        )

    def forward(self, x):
        # define forward pass through the network

        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, *self.conv_in_shape)
        x = self.layers(x)
        return x


class EncoderMult1d(nn.Module):
    """
    1d version
    """

    def __init__(self, in_features: int, latent_dim=3, hidden_dim=[64, 32, 16, 8, 4]):
        super(EncoderMult1d, self).__init__()
        self.layers = []

        if len(hidden_dim) == 0:
            self.layers.append(
                Linear(in_features, latent_dim)
            )
        else:
            # specify layer 4 - in 128, out 64
            encoder_L = Linear(in_features, hidden_dim[0])
            self.layers.append(encoder_L)

            for dim_id in range(len(hidden_dim) - 1):
                encoder_L = Linear(hidden_dim[dim_id], hidden_dim[dim_id + 1])
                self.layers.append(encoder_L)

            encoder_L = Linear(hidden_dim[-1], latent_dim)
            self.layers.append(encoder_L)

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        # define forward pass through the network
        x = self.layers(x)

        return x


class DecoderMult1d(nn.Module):
    """
    1d version
    """

    def __init__(self, out_features, latent_dim=3, hidden_dim=[4, 8, 16, 32, 64]):
        super(DecoderMult1d, self).__init__()
        self.layers = []

        if len(hidden_dim) == 0:
            self.layers.extend([
                LinearEnd(latent_dim, out_features)
            ])
        else:
            # specify layer 1 - in 3, out 4
            decoder_L = Linear(latent_dim, hidden_dim[0])  # add linearity
            self.layers.append(decoder_L)

            # for dim in hidden_dim[1:-1]:
            for dim_id in range(len(hidden_dim) - 1):
                # specify layer 2 - in 4, out 8
                decoder_L = Linear(hidden_dim[dim_id], hidden_dim[dim_id + 1])
                self.layers.append(decoder_L)

            decoder_L = LinearEnd(hidden_dim[-1], out_features)
            self.layers.extend([
                decoder_L
            ])

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        # define forward pass through the network
        x = self.layers(x)
        return x
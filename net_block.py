from torch import nn


class DownConv2d(nn.Module):
    """
    Conv 2d Block
        conv2d -> gelu -> conv2d -> gelu
    """

    def __init__(self, num_input_channels: int, num_output_channels: int, num_mid_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_input_channels, num_mid_channels, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            nn.BatchNorm2d(num_mid_channels),
            nn.LeakyReLU(negative_slope=0.4, inplace=True),
            nn.Conv2d(num_mid_channels, num_output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_output_channels),
            nn.LeakyReLU(negative_slope=0.4, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConv2d(nn.Module):
    """
    Conv 2d Block
        trans_conv2d -> gelu -> conv2d -> gelu
    """

    def __init__(self, num_input_channels: int, num_output_channels: int, num_mid_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(num_input_channels, num_mid_channels, kernel_size=3, output_padding=1, padding=1,
                               stride=2),  # 32x32 => 16x16
            nn.BatchNorm2d(num_mid_channels),
            nn.LeakyReLU(negative_slope=0.4, inplace=True),
            nn.Conv2d(num_mid_channels, num_output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_output_channels),
            nn.LeakyReLU(negative_slope=0.4, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConv2dEnd(nn.Module):
    """
    Conv 2d Block
        trans_conv2d -> gelu -> conv2d
    """

    def __init__(self, num_input_channels: int, num_output_channels: int, num_mid_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(num_input_channels, num_mid_channels, kernel_size=3, output_padding=1, padding=1,
                               stride=2),  # 32x32 => 16x16
            nn.BatchNorm2d(num_mid_channels),
            nn.LeakyReLU(negative_slope=0.4, inplace=True),
            nn.Conv2d(num_mid_channels, num_output_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class DownConv3d(nn.Module):
    def __init__(self, num_input_channels: int, num_output_channels: int, num_mid_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(num_input_channels, num_mid_channels, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            nn.BatchNorm3d(num_mid_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(num_mid_channels, num_output_channels, kernel_size=3, padding=1),  # 32x32 => 16x16
            nn.BatchNorm3d(num_output_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConv3d(nn.Module):
    def __init__(self, num_input_channels: int, num_output_channels: int, num_mid_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(num_input_channels, num_mid_channels, kernel_size=3, output_padding=1, padding=1,
                               stride=2),  # 32x32 => 16x16
            nn.BatchNorm3d(num_mid_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(num_mid_channels, num_output_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_output_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConv3dEnd(nn.Module):
    def __init__(self, num_input_channels: int, num_output_channels: int, num_mid_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(num_input_channels, num_mid_channels, kernel_size=3, output_padding=1, padding=1,
                               stride=2),  # 32x32 => 16x16
            nn.BatchNorm3d(num_mid_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(num_mid_channels, num_output_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class Linear(nn.Module):
    def __init__(self, num_input: int, num_output: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(num_input, num_output, bias=True),
            nn.LeakyReLU(negative_slope=0.4, inplace=True),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)


class LinearEnd(nn.Module):
    def __init__(self, num_input: int, num_output: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(num_input, num_output, bias=True),
        )

    def forward(self, x):
        return self.block(x)
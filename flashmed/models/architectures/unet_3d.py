"""3D UNet for volumetric medical image segmentation (CT/MRI)."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashmed.registry import MODELS


class ConvBlock3D(nn.Module):
    """Double convolution block for 3D UNet."""

    def __init__(self, in_channels: int, out_channels: int, spatial_dims: int = 3):
        super().__init__()
        Conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        Norm = nn.InstanceNorm3d if spatial_dims == 3 else nn.InstanceNorm2d

        self.conv = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Norm(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Norm(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Encoder block: downsample then double conv."""

    def __init__(self, in_channels: int, out_channels: int, spatial_dims: int = 3):
        super().__init__()
        Pool = nn.MaxPool3d if spatial_dims == 3 else nn.MaxPool2d
        self.pool = Pool(kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_channels, out_channels, spatial_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """Decoder block: upsample, concatenate skip, then double conv."""

    def __init__(self, in_channels: int, out_channels: int, spatial_dims: int = 3):
        super().__init__()
        ConvT = nn.ConvTranspose3d if spatial_dims == 3 else nn.ConvTranspose2d
        self.up = ConvT(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_channels, out_channels, spatial_dims)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear" if x.dim() == 5 else "bilinear",
                              align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


@MODELS.register("UNet3D")
class UNet3D(nn.Module):
    """3D UNet for volumetric segmentation of organs and lesions.

    Supports both 2D and 3D segmentation through spatial_dims parameter.
    Uses instance normalization and LeakyReLU for stable medical image training.

    Args:
        in_channels: Number of input channels (e.g., 4 for BraTS multi-modal)
        num_classes: Number of segmentation classes
        base_filters: Number of filters in the first layer (doubled at each level)
        depth: Number of encoder/decoder levels
        spatial_dims: 2 for 2D slices, 3 for full volumes
        deep_supervision: Whether to use deep supervision outputs
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_filters: int = 32,
        depth: int = 4,
        spatial_dims: int = 3,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.deep_supervision = deep_supervision
        self.spatial_dims = spatial_dims

        filters = [base_filters * (2 ** i) for i in range(depth + 1)]

        self.encoder_input = ConvBlock3D(in_channels, filters[0], spatial_dims)
        self.encoders = nn.ModuleList([
            DownBlock(filters[i], filters[i + 1], spatial_dims) for i in range(depth)
        ])
        self.decoders = nn.ModuleList([
            UpBlock(filters[i + 1], filters[i], spatial_dims) for i in range(depth - 1, -1, -1)
        ])

        Conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.final_conv = Conv(filters[0], num_classes, kernel_size=1)

        if deep_supervision:
            self.deep_outputs = nn.ModuleList([
                Conv(filters[i], num_classes, kernel_size=1) for i in range(1, depth)
            ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, (nn.InstanceNorm3d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        skips = []
        x = self.encoder_input(x)
        skips.append(x)

        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        skips = skips[:-1]
        skips = skips[::-1]

        deep_outputs = []
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[i])
            if self.deep_supervision and i < len(self.decoders) - 1:
                deep_outputs.append(x)

        output = self.final_conv(x)

        if self.deep_supervision and self.training:
            return output, deep_outputs

        return output

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

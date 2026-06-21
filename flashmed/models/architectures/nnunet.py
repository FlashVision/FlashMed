"""nnU-Net - Self-configuring segmentation network for medical imaging."""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashmed.registry import MODELS


class ResidualBlock(nn.Module):
    """Residual encoder block with instance normalization and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, spatial_dims: int = 3, stride: int = 1):
        super().__init__()
        Conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        Norm = nn.InstanceNorm3d if spatial_dims == 3 else nn.InstanceNorm2d

        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = Norm(out_channels, affine=True)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = Norm(out_channels, affine=True)
        self.activation = nn.LeakyReLU(0.01, inplace=True)

        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                Conv(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                Norm(out_channels, affine=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class StackedResidualBlocks(nn.Module):
    """Stack of residual blocks forming one encoder/decoder stage."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, spatial_dims: int = 3, stride: int = 1):
        super().__init__()
        blocks = [ResidualBlock(in_channels, out_channels, spatial_dims, stride)]
        for _ in range(num_blocks - 1):
            blocks.append(ResidualBlock(out_channels, out_channels, spatial_dims))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class UpsampleBlock(nn.Module):
    """Transposed convolution upsampling followed by residual blocks."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, spatial_dims: int = 3):
        super().__init__()
        ConvT = nn.ConvTranspose3d if spatial_dims == 3 else nn.ConvTranspose2d
        self.upsample = ConvT(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = StackedResidualBlocks(out_channels + skip_channels, out_channels, num_blocks=2,
                                                spatial_dims=spatial_dims)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear" if x.dim() == 5 else "bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


@MODELS.register("nnUNet")
class nnUNet(nn.Module):
    """Self-configuring U-Net for medical image segmentation.

    Implements the nnU-Net architecture with residual encoder blocks, instance
    normalization, and deep supervision. Supports automatic architecture configuration
    based on dataset properties such as spatial dimensions and patch size.

    Args:
        in_channels: Number of input channels (modalities).
        num_classes: Number of segmentation output classes.
        base_filters: Number of filters in the first encoder stage.
        max_filters: Maximum number of filters in any stage.
        spatial_dims: Spatial dimensionality (2 or 3).
        deep_supervision: Enable deep supervision outputs during training.
        num_pool_per_axis: Number of pooling operations per spatial axis.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_filters: int = 32,
        max_filters: int = 320,
        spatial_dims: int = 3,
        deep_supervision: bool = True,
        num_pool_per_axis: Tuple[int, ...] = (5, 5, 5),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.spatial_dims = spatial_dims
        self.deep_supervision = deep_supervision
        self.num_pool_per_axis = num_pool_per_axis

        num_stages = max(num_pool_per_axis) + 1
        encoder_filters = []
        for i in range(num_stages):
            filters = min(base_filters * (2 ** i), max_filters)
            encoder_filters.append(filters)

        self.encoder_stages = nn.ModuleList()
        for i in range(num_stages):
            in_ch = in_channels if i == 0 else encoder_filters[i - 1]
            out_ch = encoder_filters[i]
            stride = 1 if i == 0 else 2
            self.encoder_stages.append(
                StackedResidualBlocks(in_ch, out_ch, num_blocks=2, spatial_dims=spatial_dims, stride=stride)
            )

        self.decoder_stages = nn.ModuleList()
        for i in range(num_stages - 2, -1, -1):
            in_ch = encoder_filters[i + 1]
            skip_ch = encoder_filters[i]
            out_ch = encoder_filters[i]
            self.decoder_stages.append(UpsampleBlock(in_ch, skip_ch, out_ch, spatial_dims))

        Conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.seg_output = Conv(encoder_filters[0], num_classes, kernel_size=1)

        if deep_supervision:
            self.deep_supervision_heads = nn.ModuleList([
                Conv(encoder_filters[i], num_classes, kernel_size=1)
                for i in range(1, num_stages - 1)
            ])

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.InstanceNorm3d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with optional deep supervision.

        Args:
            x: Input tensor of shape (B, C, *spatial_dims).

        Returns:
            Segmentation logits. If deep_supervision=True and training, returns a list
            of outputs at multiple resolutions (highest-res first).
        """
        encoder_features = []
        for stage in self.encoder_stages:
            x = stage(x)
            encoder_features.append(x)

        skips = encoder_features[:-1][::-1]
        x = encoder_features[-1]

        decoder_outputs = []
        for i, decoder in enumerate(self.decoder_stages):
            x = decoder(x, skips[i])
            decoder_outputs.append(x)

        seg = self.seg_output(x)

        if self.deep_supervision and self.training:
            deep_outputs = [seg]
            for i, feat in enumerate(decoder_outputs[:-1]):
                idx = len(decoder_outputs) - 2 - i
                deep_outputs.append(self.deep_supervision_heads[idx](feat))
            return deep_outputs

        return seg

    @classmethod
    def fingerprint_dataset(cls, data_info: Dict) -> Dict:
        """Analyze dataset properties and return an optimal architecture configuration.

        Args:
            data_info: Dictionary with keys like 'median_shape', 'spacing',
                       'num_classes', 'in_channels', 'num_training_samples'.

        Returns:
            Configuration dictionary suitable for passing to auto_configure().
        """
        median_shape = data_info.get("median_shape", (128, 128, 128))
        spacing = data_info.get("spacing", (1.0, 1.0, 1.0))
        num_classes = data_info.get("num_classes", 4)
        in_channels = data_info.get("in_channels", 1)

        spatial_dims = 3 if len(median_shape) == 3 else 2
        num_pool_per_axis = tuple(
            max(1, min(5, int(math.log2(s)))) for s in median_shape
        )

        anisotropy_ratio = max(spacing) / (min(spacing) + 1e-8)
        base_filters = 32 if anisotropy_ratio < 3.0 else 24
        max_filters = 320 if spatial_dims == 3 else 512

        return {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "base_filters": base_filters,
            "max_filters": max_filters,
            "spatial_dims": spatial_dims,
            "num_pool_per_axis": num_pool_per_axis,
            "deep_supervision": True,
        }

    @classmethod
    def auto_configure(cls, dataset_info: Dict) -> "nnUNet":
        """Build an nnU-Net with architecture automatically configured from dataset properties.

        Args:
            dataset_info: Dictionary containing dataset metadata (median_shape, spacing,
                         num_classes, in_channels, etc.).

        Returns:
            Configured nnUNet instance.
        """
        config = cls.fingerprint_dataset(dataset_info)
        return cls(**config)

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

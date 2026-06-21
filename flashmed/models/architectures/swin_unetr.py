"""SwinUNETR - Swin Transformer encoder with UNet-style decoder for medical segmentation."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashmed.registry import MODELS


def _get_window_partition(x: torch.Tensor, window_size: int, spatial_dims: int) -> Tuple[torch.Tensor, List[int]]:
    """Partition spatial tensor into non-overlapping windows."""
    if spatial_dims == 3:
        B, D, H, W, C = x.shape
        pad_d = (window_size - D % window_size) % window_size
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        _, Dp, Hp, Wp, _ = x.shape
        x = x.view(B, Dp // window_size, window_size, Hp // window_size, window_size,
                   Wp // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, window_size ** 3, C)
        return windows, [Dp, Hp, Wp]
    else:
        B, H, W, C = x.shape
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = x.shape
        x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size ** 2, C)
        return windows, [Hp, Wp]


def _window_reverse(windows: torch.Tensor, window_size: int, padded_sizes: List[int],
                    batch_size: int, spatial_dims: int) -> torch.Tensor:
    """Reverse window partition back to spatial tensor."""
    if spatial_dims == 3:
        Dp, Hp, Wp = padded_sizes
        nD, nH, nW = Dp // window_size, Hp // window_size, Wp // window_size
        x = windows.view(batch_size, nD, nH, nW, window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(batch_size, Dp, Hp, Wp, -1)
        return x
    else:
        Hp, Wp = padded_sizes
        nH, nW = Hp // window_size, Wp // window_size
        x = windows.view(batch_size, nH, nW, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, Hp, Wp, -1)
        return x


class WindowAttention(nn.Module):
    """Multi-head attention within local windows with relative position bias."""

    def __init__(self, embed_dim: int, num_heads: int, window_size: int, spatial_dims: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.spatial_dims = spatial_dims

        window_size ** spatial_dims
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** spatial_dims, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_win, N, C = x.shape
        qkv = self.qkv(x).reshape(B_win, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_win, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with shifted window multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int, window_size: int = 7,
                 shift_size: int = 0, mlp_ratio: float = 4.0, spatial_dims: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.spatial_dims = spatial_dims

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WindowAttention(embed_dim, num_heads, window_size, spatial_dims)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor, spatial_shape: List[int]) -> torch.Tensor:
        B = x.shape[0]
        if self.spatial_dims == 3:
            D, H, W = spatial_shape
            x = x.view(B, D, H, W, -1)
            if self.shift_size > 0:
                x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
        else:
            H, W = spatial_shape
            x = x.view(B, H, W, -1)
            if self.shift_size > 0:
                x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        shortcut_spatial = x
        x = self.norm1(x.reshape(B, -1, self.embed_dim)).reshape(shortcut_spatial.shape)

        windows, padded_sizes = _get_window_partition(x, self.window_size, self.spatial_dims)
        windows = self.attn(windows)
        x = _window_reverse(windows, self.window_size, padded_sizes, B, self.spatial_dims)

        # Crop padding
        if self.spatial_dims == 3:
            D, H, W = spatial_shape
            x = x[:, :D, :H, :W, :].contiguous()
        else:
            H, W = spatial_shape
            x = x[:, :H, :W, :].contiguous()

        if self.shift_size > 0:
            if self.spatial_dims == 3:
                x = torch.roll(x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
            else:
                x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut_spatial[:, :x.shape[1], :x.shape[2], :x.shape[3], :] + x if self.spatial_dims == 3 else \
            shortcut_spatial[:, :x.shape[1], :x.shape[2], :] + x
        flat = x.reshape(B, -1, self.embed_dim)
        x = flat + self.mlp(self.norm2(flat))
        return x


class PatchMerging(nn.Module):
    """Patch merging layer for spatial downsampling (2x each axis)."""

    def __init__(self, embed_dim: int, spatial_dims: int = 3):
        super().__init__()
        self.spatial_dims = spatial_dims
        merge_factor = 2 ** spatial_dims
        self.reduction = nn.Linear(merge_factor * embed_dim, 2 * embed_dim, bias=False)
        self.norm = nn.LayerNorm(merge_factor * embed_dim)

    def forward(self, x: torch.Tensor, spatial_shape: List[int]) -> Tuple[torch.Tensor, List[int]]:
        B = x.shape[0]
        if self.spatial_dims == 3:
            D, H, W = spatial_shape
            x = x.view(B, D, H, W, -1)
            # Pad if odd
            pad_d = D % 2
            pad_h = H % 2
            pad_w = W % 2
            if pad_d or pad_h or pad_w:
                x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
                D, H, W = D + pad_d, H + pad_h, W + pad_w
            x0 = x[:, 0::2, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, 0::2, :]
            x3 = x[:, 0::2, 0::2, 1::2, :]
            x4 = x[:, 1::2, 1::2, 0::2, :]
            x5 = x[:, 0::2, 1::2, 1::2, :]
            x6 = x[:, 1::2, 0::2, 1::2, :]
            x7 = x[:, 1::2, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
            new_shape = [D // 2, H // 2, W // 2]
        else:
            H, W = spatial_shape
            x = x.view(B, H, W, -1)
            pad_h = H % 2
            pad_w = W % 2
            if pad_h or pad_w:
                x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
                H, W = H + pad_h, W + pad_w
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], dim=-1)
            new_shape = [H // 2, W // 2]

        x = x.reshape(B, -1, x.shape[-1])
        x = self.norm(x)
        x = self.reduction(x)
        return x, new_shape


class SwinStage(nn.Module):
    """A stage of Swin Transformer blocks followed by optional patch merging."""

    def __init__(self, embed_dim: int, depth: int, num_heads: int, window_size: int = 7,
                 downsample: bool = True, spatial_dims: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                spatial_dims=spatial_dims,
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging(embed_dim, spatial_dims) if downsample else None

    def forward(self, x: torch.Tensor, spatial_shape: List[int]) -> Tuple[torch.Tensor, List[int]]:
        for block in self.blocks:
            x = block(x, spatial_shape)
        if self.downsample is not None:
            x, spatial_shape = self.downsample(x, spatial_shape)
        return x, spatial_shape


class UNetDecoderBlock(nn.Module):
    """UNet-style decoder block with skip connection and conv layers."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, spatial_dims: int = 3):
        super().__init__()
        ConvT = nn.ConvTranspose3d if spatial_dims == 3 else nn.ConvTranspose2d
        Conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        Norm = nn.InstanceNorm3d if spatial_dims == 3 else nn.InstanceNorm2d

        self.upsample = ConvT(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            Conv(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Norm(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Norm(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear" if x.dim() == 5 else "bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


@MODELS.register("SwinUNETR")
class SwinUNETR(nn.Module):
    """Swin Transformer encoder with UNet-style decoder for medical image segmentation.

    Combines the hierarchical feature extraction of Swin Transformers with shifted
    window attention and the dense prediction capability of UNet decoders via skip
    connections. Supports both 2D and 3D inputs.

    Args:
        img_size: Input spatial resolution (isotropic).
        in_channels: Number of input channels.
        num_classes: Number of segmentation output classes.
        embed_dim: Base embedding dimension for the Swin encoder.
        depths: Number of Swin blocks at each stage.
        num_heads: Number of attention heads at each stage.
        window_size: Local window size for attention.
        spatial_dims: Spatial dimensionality (2 or 3).
    """

    def __init__(
        self,
        img_size: int = 96,
        in_channels: int = 4,
        num_classes: int = 4,
        embed_dim: int = 48,
        depths: Tuple[int, ...] = (2, 2, 2, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        spatial_dims: int = 3,
    ):
        super().__init__()
        self.img_size = img_size
        self.spatial_dims = spatial_dims
        self.num_stages = len(depths)
        self.embed_dim = embed_dim

        patch_size = 2
        Conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.patch_embed = nn.Sequential(
            Conv(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.LayerNorm(embed_dim) if False else nn.Identity(),
        )
        self.patch_grid = img_size // patch_size

        self.pos_drop = nn.Dropout(p=0.0)

        stage_dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            self.stages.append(SwinStage(
                embed_dim=stage_dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                downsample=(i < self.num_stages - 1),
                spatial_dims=spatial_dims,
            ))

        # Encoder feature projection to channel-first format for skip connections
        self.encoder_norms = nn.ModuleList([nn.LayerNorm(stage_dims[i]) for i in range(self.num_stages)])

        # Decoder
        decoder_channels = list(reversed(stage_dims))
        self.decoders = nn.ModuleList()
        for i in range(self.num_stages - 1):
            self.decoders.append(UNetDecoderBlock(
                in_channels=decoder_channels[i],
                skip_channels=decoder_channels[i + 1],
                out_channels=decoder_channels[i + 1],
                spatial_dims=spatial_dims,
            ))

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose3d(stage_dims[0], stage_dims[0], kernel_size=2, stride=2) if spatial_dims == 3
            else nn.ConvTranspose2d(stage_dims[0], stage_dims[0], kernel_size=2, stride=2),
            nn.InstanceNorm3d(stage_dims[0], affine=True) if spatial_dims == 3
            else nn.InstanceNorm2d(stage_dims[0], affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.seg_head = Conv(stage_dims[0], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _to_spatial(self, x: torch.Tensor, spatial_shape: List[int]) -> torch.Tensor:
        """Reshape sequence to spatial channel-first tensor."""
        B = x.shape[0]
        if self.spatial_dims == 3:
            D, H, W = spatial_shape
            return x.view(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        else:
            H, W = spatial_shape
            return x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, D, H, W) for 3D or (B, C, H, W) for 2D.

        Returns:
            Segmentation logits of same spatial size as input.
        """
        x = self.patch_embed(x)
        if self.spatial_dims == 3:
            B, C, D, H, W = x.shape
            spatial_shape = [D, H, W]
            x = x.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        else:
            B, C, H, W = x.shape
            spatial_shape = [H, W]
            x = x.permute(0, 2, 3, 1).reshape(B, -1, C)

        x = self.pos_drop(x)

        encoder_features = []
        current_shape = spatial_shape
        for i, stage in enumerate(self.stages):
            x_normed = self.encoder_norms[i](x)
            encoder_features.append(self._to_spatial(x_normed, current_shape))
            x, current_shape = stage(x, current_shape)

        # Bottleneck is the output of last stage (without downsample)
        x_spatial = encoder_features[-1]

        skips = encoder_features[:-1][::-1]
        out = x_spatial
        for i, decoder in enumerate(self.decoders):
            out = decoder(out, skips[i])

        out = self.final_upsample(out)
        out = self.seg_head(out)
        return out

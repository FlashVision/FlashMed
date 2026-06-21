"""Medical Vision Transformer (MedViT) for classification, detection, and pathology."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashmed.registry import MODELS


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional relative position bias."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@MODELS.register("MedViT")
class MedViT(nn.Module):
    """Medical Vision Transformer optimized for medical imaging tasks.

    Supports multi-label classification (e.g., 14 chest pathologies),
    single-label classification (e.g., pathology tissue types), and
    feature extraction for downstream tasks.

    Args:
        num_classes: Number of output classes
        in_channels: Input channels (1 for grayscale, 3 for RGB)
        img_size: Input image resolution
        patch_size: Size of each patch
        embed_dim: Transformer embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        dropout: Dropout rate
        pretrained: Load pretrained weights (if available)
        multi_label: Use sigmoid activation (True) or softmax (False)
    """

    def __init__(
        self,
        num_classes: int = 14,
        in_channels: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pretrained: bool = True,
        multi_label: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.multi_label = multi_label

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor, return_features: bool = False, **kwargs) -> torch.Tensor:
        features = self.forward_features(x)
        if return_features:
            return features
        logits = self.head(features)
        return logits

    def freeze_backbone(self):
        """Freeze everything except the classification head."""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract attention maps for visualization/GradCAM."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        attention_maps = []
        for block in self.blocks:
            x_norm = block.norm1(x)
            B_size, N, C = x_norm.shape
            qkv = block.attn.qkv(x_norm).reshape(B_size, N, 3, block.attn.num_heads, block.attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            attention_maps.append(attn)
            x = block(x)

        return torch.stack(attention_maps)

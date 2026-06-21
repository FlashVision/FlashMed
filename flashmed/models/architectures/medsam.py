"""MedSAM - Segment Anything Model adapted for medical image segmentation."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashmed.registry import MODELS


class PatchEmbed(nn.Module):
    """2D image to patch embedding with positional encoding support."""

    def __init__(self, img_size: int = 1024, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ImageEncoderBlock(nn.Module):
    """Transformer block for the image encoder with pre-norm and windowed attention."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
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
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PromptEncoder(nn.Module):
    """Encodes bounding boxes and optional point prompts into dense embeddings."""

    def __init__(self, embed_dim: int = 256, num_mask_tokens: int = 4, img_embedding_size: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_mask_tokens = num_mask_tokens
        self.box_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.point_embed = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.mask_tokens = nn.Embedding(num_mask_tokens, embed_dim)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

    def forward(
        self, boxes: torch.Tensor, points: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts into sparse and dense embeddings.

        Args:
            boxes: Bounding boxes (B, 4) normalized to [0, 1].
            points: Optional point prompts (B, N, 2) normalized to [0, 1].

        Returns:
            Tuple of (sparse_embeddings, dense_embeddings) where sparse has shape
            (B, num_tokens, embed_dim).
        """
        B = boxes.shape[0]
        sparse = [self.box_embed(boxes).unsqueeze(1)]

        if points is not None:
            point_embeds = self.point_embed(points)
            sparse.append(point_embeds)

        mask_tokens = self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        sparse.append(mask_tokens)

        sparse_embeddings = torch.cat(sparse, dim=1)
        return sparse_embeddings


class MaskDecoderBlock(nn.Module):
    """Cross-attention block for mask prediction."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.cross_attn_token_to_image = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_image_to_token = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, tokens: torch.Tensor, image_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.norm1(tokens)
        tokens = tokens + self.cross_attn_token_to_image(q, image_embedding, image_embedding)[0]
        tokens = tokens + self.mlp(self.norm2(tokens))

        q = self.norm3(image_embedding)
        image_embedding = image_embedding + self.cross_attn_image_to_token(q, tokens, tokens)[0]
        return tokens, image_embedding


class MaskDecoder(nn.Module):
    """Lightweight decoder that predicts segmentation masks from image embeddings and prompt tokens."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 2, num_mask_tokens: int = 4):
        super().__init__()
        self.num_mask_tokens = num_mask_tokens
        self.layers = nn.ModuleList([MaskDecoderBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.mask_prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, embed_dim // 8),
            )
            for _ in range(num_mask_tokens)
        ])
        self.iou_prediction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_mask_tokens),
        )

    def forward(
        self, image_embedding: torch.Tensor, sparse_prompt_embedding: torch.Tensor, image_grid_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = sparse_prompt_embedding
        for layer in self.layers:
            tokens, image_embedding = layer(tokens, image_embedding)

        B, N, C = image_embedding.shape
        image_2d = image_embedding.transpose(1, 2).reshape(B, C, image_grid_size, image_grid_size)
        upscaled = self.output_upscaling(image_2d)

        mask_tokens = tokens[:, -self.num_mask_tokens:, :]
        hyper_in = torch.stack([head(mask_tokens[:, i]) for i, head in enumerate(self.mask_prediction_heads)], dim=1)

        _, C_up, H, W = upscaled.shape
        masks = torch.einsum("bmc,bchw->bmhw", hyper_in, upscaled)

        iou_pred = self.iou_prediction_head(mask_tokens.mean(dim=1))
        return masks, iou_pred


@MODELS.register("MedSAM")
class MedSAM(nn.Module):
    """Segment Anything Model adapted for medical image segmentation.

    Uses a ViT-based image encoder with positional embeddings, a prompt encoder
    for bounding boxes and point prompts, and a lightweight cross-attention mask
    decoder for efficient segmentation.

    Args:
        img_size: Input image size (square).
        patch_size: Size of each image patch.
        encoder_embed_dim: Embedding dimension for the image encoder.
        encoder_depth: Number of transformer blocks in the encoder.
        encoder_num_heads: Number of attention heads in the encoder.
        decoder_embed_dim: Embedding dimension for the mask decoder.
        num_mask_tokens: Number of mask prediction tokens.
        in_channels: Number of input image channels.
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        encoder_embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        decoder_embed_dim: int = 256,
        num_mask_tokens: int = 4,
        in_channels: int = 3,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.grid_size = img_size // patch_size

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, encoder_embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            ImageEncoderBlock(encoder_embed_dim, encoder_num_heads) for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        self.neck = nn.Sequential(
            nn.Linear(encoder_embed_dim, decoder_embed_dim),
            nn.LayerNorm(decoder_embed_dim),
        )

        self.prompt_encoder = PromptEncoder(decoder_embed_dim, num_mask_tokens, self.grid_size)
        self.mask_decoder = MaskDecoder(decoder_embed_dim, num_heads=8, num_layers=2, num_mask_tokens=num_mask_tokens)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Extract image embeddings from the ViT encoder.

        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            Image embeddings of shape (B, num_patches, decoder_embed_dim).
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        x = self.neck(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        boxes: torch.Tensor,
        points: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting segmentation masks from images and prompts.

        Args:
            x: Input images (B, C, H, W).
            boxes: Bounding box prompts (B, 4) in normalized coordinates.
            points: Optional point prompts (B, N, 2) in normalized coordinates.

        Returns:
            Tuple of (masks, iou_predictions) where masks has shape
            (B, num_mask_tokens, H_up, W_up).
        """
        image_embeddings = self.encode_image(x)
        sparse_embeddings = self.prompt_encoder(boxes, points)
        masks, iou_pred = self.mask_decoder(image_embeddings, sparse_embeddings, self.grid_size)
        masks = F.interpolate(masks, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return masks, iou_pred

    def freeze_encoder(self) -> None:
        """Freeze the image encoder for fine-tuning only the decoder."""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.pos_embed.requires_grad = False
        for param in self.encoder_blocks.parameters():
            param.requires_grad = False
        for param in self.encoder_norm.parameters():
            param.requires_grad = False

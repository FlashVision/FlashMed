"""BiomedCLIP - Contrastive Language-Image Pre-training for biomedical zero-shot classification."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashmed.registry import MODELS


class VisionPatchEmbed(nn.Module):
    """Patch embedding for the vision encoder."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 512):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformerBlock(nn.Module):
    """Transformer block for the vision encoder."""

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
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """ViT-based vision encoder for BiomedCLIP."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 512,
                 depth: int = 12, num_heads: int = 8, in_channels: int = 3):
        super().__init__()
        self.patch_embed = VisionPatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            VisionTransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]


class TextTransformerBlock(nn.Module):
    """Causal transformer block for the text encoder."""

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

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class TextEncoder(nn.Module):
    """Transformer-based text encoder for BiomedCLIP."""

    def __init__(self, vocab_size: int = 30522, max_text_len: int = 77, embed_dim: int = 512,
                 depth: int = 6, num_heads: int = 8):
        super().__init__()
        self.max_text_len = max_text_len
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_text_len, embed_dim))

        self.blocks = nn.ModuleList([
            TextTransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode text tokens.

        Args:
            input_ids: Token IDs of shape (B, seq_len).

        Returns:
            Text embeddings of shape (B, embed_dim), taken from the EOS position.
        """
        B, seq_len = input_ids.shape
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :seq_len, :]

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)

        x = self.norm(x)
        # Use last non-padding token (EOS) as the text representation
        eos_indices = input_ids.argmax(dim=-1)
        text_features = x[torch.arange(B, device=x.device), eos_indices]
        return text_features


@MODELS.register("BiomedCLIP")
class BiomedCLIP(nn.Module):
    """Contrastive Language-Image Pre-training model for biomedical zero-shot classification.

    Jointly trains a vision encoder and text encoder in a shared embedding space using
    contrastive learning. Supports zero-shot classification by comparing image embeddings
    against text embeddings of class descriptions.

    Args:
        img_size: Input image resolution.
        patch_size: Patch size for the vision encoder.
        vision_embed_dim: Embedding dimension for the vision encoder.
        text_embed_dim: Embedding dimension for the text encoder.
        vision_depth: Number of transformer blocks in the vision encoder.
        text_depth: Number of transformer blocks in the text encoder.
        vision_heads: Number of attention heads in the vision encoder.
        text_heads: Number of attention heads in the text encoder.
        vocab_size: Vocabulary size for the text tokenizer.
        max_text_len: Maximum text sequence length.
        projection_dim: Dimension of the shared projection space.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        vision_embed_dim: int = 512,
        text_embed_dim: int = 512,
        vision_depth: int = 12,
        text_depth: int = 6,
        vision_heads: int = 8,
        text_heads: int = 8,
        vocab_size: int = 30522,
        max_text_len: int = 77,
        projection_dim: int = 256,
    ):
        super().__init__()
        self.projection_dim = projection_dim

        self.vision_encoder = VisionEncoder(
            img_size=img_size, patch_size=patch_size, embed_dim=vision_embed_dim,
            depth=vision_depth, num_heads=vision_heads,
        )
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size, max_text_len=max_text_len, embed_dim=text_embed_dim,
            depth=text_depth, num_heads=text_heads,
        )

        self.vision_projection = nn.Sequential(
            nn.Linear(vision_embed_dim, projection_dim, bias=False),
            nn.LayerNorm(projection_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_embed_dim, projection_dim, bias=False),
            nn.LayerNorm(projection_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

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
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into the shared projection space.

        Args:
            images: Input images of shape (B, C, H, W).

        Returns:
            Normalized image embeddings of shape (B, projection_dim).
        """
        features = self.vision_encoder(images)
        projected = self.vision_projection(features)
        return F.normalize(projected, dim=-1)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode text tokens into the shared projection space.

        Args:
            input_ids: Token IDs of shape (B, seq_len).

        Returns:
            Normalized text embeddings of shape (B, projection_dim).
        """
        features = self.text_encoder(input_ids)
        projected = self.text_projection(features)
        return F.normalize(projected, dim=-1)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute similarity logits between images and text.

        Args:
            images: Input images (B, C, H, W).
            input_ids: Text token IDs (B, seq_len).

        Returns:
            Logit matrix of shape (B, B) representing pairwise cosine similarities
            scaled by the learned temperature.
        """
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(input_ids)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (image_embeds @ text_embeds.T)
        return logits

    def zero_shot_classify(self, images: torch.Tensor, class_descriptions: torch.Tensor) -> torch.Tensor:
        """Perform zero-shot classification using text descriptions of classes.

        Args:
            images: Input images (B, C, H, W).
            class_descriptions: Token IDs for class descriptions (num_classes, seq_len).

        Returns:
            Classification probabilities of shape (B, num_classes).
        """
        image_embeds = self.encode_image(images)
        with torch.no_grad():
            text_embeds = self.encode_text(class_descriptions)
        logit_scale = self.logit_scale.exp()
        similarity = logit_scale * (image_embeds @ text_embeds.T)
        return similarity.softmax(dim=-1)

    def contrastive_loss(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute symmetric contrastive loss (InfoNCE).

        Args:
            images: Input images (B, C, H, W).
            input_ids: Corresponding text token IDs (B, seq_len).

        Returns:
            Scalar contrastive loss.
        """
        logits = self.forward(images, input_ids)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2.0

"""Medical Vision-Language Model for radiology report generation."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashmed.registry import MODELS


class VisualEncoder(nn.Module):
    """Lightweight visual encoder based on ViT for the VLM pipeline."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 512):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

        depth = 6
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4,
                dropout=0.1, activation="gelu", batch_first=True,
            )
            for _ in range(depth)
        ])

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class CrossAttention(nn.Module):
    """Cross-attention layer for attending from text tokens to visual tokens."""

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        residual = query
        query = self.norm(query)
        attn_out, _ = self.multihead_attn(query, key_value, key_value)
        return residual + self.dropout(attn_out)


class TextDecoder(nn.Module):
    """Autoregressive text decoder for report generation."""

    def __init__(self, vocab_size: int = 30522, embed_dim: int = 512, num_heads: int = 8,
                 num_layers: int = 6, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "self_attn": nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
                    dropout=dropout, activation="gelu", batch_first=True,
                ),
                "cross_attn": CrossAttention(embed_dim, num_heads, dropout),
            }))

        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embed(input_ids) + self.pos_embed(positions)

        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            x = layer["self_attn"](x, src_mask=causal_mask)
            x = layer["cross_attn"](x, visual_features)

        x = self.norm(x)
        logits = self.output_proj(x)
        return logits


@MODELS.register("MedVLM")
class MedVLM(nn.Module):
    """Medical Vision-Language Model for radiology report generation.

    Combines a visual encoder with a text decoder using cross-attention
    to generate structured radiology reports from medical images.

    Args:
        in_channels: Number of input image channels
        img_size: Input image resolution
        embed_dim: Model embedding dimension
        vocab_size: Vocabulary size for text generation
        max_seq_len: Maximum sequence length for generated reports
        num_encoder_layers: Visual encoder depth
        num_decoder_layers: Text decoder depth
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        embed_dim: int = 512,
        vocab_size: int = 30522,
        max_seq_len: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.visual_encoder = VisualEncoder(img_size=img_size, in_channels=in_channels, embed_dim=embed_dim)
        self.visual_proj = nn.Linear(embed_dim, embed_dim)
        self.text_decoder = TextDecoder(
            vocab_size=vocab_size, embed_dim=embed_dim,
            num_layers=num_decoder_layers, max_seq_len=max_seq_len,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into visual feature tokens."""
        visual_features = self.visual_encoder(images)
        visual_features = self.visual_proj(visual_features)
        return visual_features

    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for training (teacher-forced).

        Args:
            x: Input images [B, C, H, W]
            input_ids: Target token IDs [B, T] (shifted right for teacher forcing)

        Returns:
            Logits [B, T, vocab_size]
        """
        visual_features = self.encode_image(x)

        if input_ids is None:
            B = x.shape[0]
            input_ids = torch.zeros(B, 1, dtype=torch.long, device=x.device)

        logits = self.text_decoder(input_ids, visual_features)
        return logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        eos_token_id: int = 102,
        bos_token_id: int = 101,
    ) -> torch.Tensor:
        """Generate report tokens autoregressively.

        Args:
            images: Input images [B, C, H, W]
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            eos_token_id: End of sequence token
            bos_token_id: Beginning of sequence token

        Returns:
            Generated token IDs [B, seq_len]
        """
        self.eval()
        B = images.shape[0]
        visual_features = self.encode_image(images)

        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=images.device)

        for _ in range(max_length - 1):
            logits = self.text_decoder(generated, visual_features)
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                next_logits = torch.where(next_logits < min_values, torch.full_like(next_logits, -1e9), next_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return generated

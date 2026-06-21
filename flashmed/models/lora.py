"""LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning of medical models."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    Implements W' = W + (alpha/r) * B @ A where A and B are low-rank matrices.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def merge(self) -> nn.Linear:
        """Merge LoRA weights back into the original linear layer."""
        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None,
        )
        merged.weight.data = self.original.weight.data + (self.lora_B @ self.lora_A) * self.scaling
        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data
        return merged


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
    target_modules: Optional[list] = None,
) -> nn.Module:
    """Apply LoRA to all linear layers (or specified target modules) in a model.

    Args:
        model: The base model to adapt
        rank: LoRA rank (lower = fewer params)
        alpha: LoRA scaling factor
        dropout: Dropout on LoRA path
        target_modules: List of module name patterns to target (None = all Linear layers)

    Returns:
        Modified model with LoRA layers
    """
    if target_modules is None:
        target_modules = ["qkv", "proj", "fc1", "fc2", "query", "key", "value", "dense"]

    replacements = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                should_replace = any(t in child_name or t in name for t in target_modules)
                if should_replace and child.in_features >= rank and child.out_features >= rank:
                    replacements.append((module, child_name, child))

    replaced = 0
    for parent, child_name, child in replacements:
        lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, lora_layer)
        replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Replaced {replaced} layers | Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge all LoRA adaptations back into the base weights.

    After merging, the model runs at full speed with no LoRA overhead.
    """
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, LoRALinear):
                merged = child.merge()
                setattr(module, child_name, merged)
    return model

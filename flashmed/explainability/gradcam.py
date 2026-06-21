"""Advanced GradCAM variants for medical image model interpretability."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_target_layer(model: nn.Module, target_layer: Optional[str] = None) -> Optional[nn.Module]:
    """Resolve a target layer by name or auto-detect the last conv/norm layer.

    Args:
        model: Neural network model.
        target_layer: Dot-separated module name, or None for auto-detection.

    Returns:
        The resolved ``nn.Module`` or ``None`` if no suitable layer is found.
    """
    if target_layer:
        for name, module in model.named_modules():
            if name == target_layer:
                return module
        return None

    last_conv = None
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            last_conv = module

    if last_conv is None:
        for name, module in model.named_modules():
            if "norm" in name.lower() and isinstance(module, nn.LayerNorm):
                last_conv = module

    return last_conv


def _get_class_score(output: torch.Tensor, class_idx: Optional[int]) -> tuple:
    """Extract the scalar score for *class_idx* from model output.

    Args:
        output: Raw model output tensor.
        class_idx: Desired class index.  ``None`` picks the argmax.

    Returns:
        ``(score, class_idx)`` where *score* is a scalar tensor that can be
        back-propagated through, and *class_idx* is the resolved integer index.
    """
    if class_idx is None:
        if output.dim() > 1 and output.shape[1] > 1:
            class_idx = output.argmax(dim=1).item()
        else:
            class_idx = 0

    if output.dim() > 1 and output.shape[1] > 1:
        score = output[0, class_idx]
    else:
        score = output[0]

    return score, class_idx


def _resize_cam(cam: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a CAM array to ``(target_h, target_w)`` and normalise to [0, 1].

    Args:
        cam: 2-D numpy array with raw CAM values.
        target_h: Desired height.
        target_w: Desired width.

    Returns:
        Normalised heatmap of shape ``(target_h, target_w)`` in ``[0, 1]``.
    """
    from PIL import Image

    cam_resized = np.array(Image.fromarray(cam.astype(np.float32)).resize((target_w, target_h)))
    if cam_resized.max() > cam_resized.min():
        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())
    else:
        cam_resized = np.zeros_like(cam_resized)
    return cam_resized


def _handle_transformer_cam(cam: torch.Tensor) -> torch.Tensor:
    """Reshape a 1-D patch-sequence CAM (from a ViT) into a 2-D spatial map.

    If the first token is a [CLS] token it is dropped before reshaping.

    Args:
        cam: 1-D tensor of length ``num_patches`` (optionally ``+1`` for CLS).

    Returns:
        2-D tensor of shape ``(side, side)``.
    """
    if cam.dim() == 1:
        num_patches = cam.shape[0] - 1
        side = int(num_patches ** 0.5)
        if side * side == num_patches:
            cam = cam[1:].reshape(side, side)
        else:
            cam = cam[1:] if cam.shape[0] > 1 else cam
            side = int(len(cam) ** 0.5) or 1
            cam = cam[: side * side].reshape(side, side)
    return cam


class GradCAMPlusPlus:
    """Grad-CAM++ with pixel-wise gradient weighting for improved localisation.

    Uses second-order and third-order gradient terms to compute per-pixel
    alpha weights, yielding more complete object localisation than vanilla
    Grad-CAM.

    Args:
        model: The classification model.
        target_layer: Dot-separated name of the target layer, or ``None``
            to auto-detect the last convolutional / layer-norm layer.

    Example::

        gcpp = GradCAMPlusPlus(model, target_layer="backbone.layer4")
        heatmap = gcpp.generate(input_tensor)
        gcpp.cleanup()
    """

    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks: list = []

        layer = _find_target_layer(model, target_layer)
        if layer is not None:
            self._hooks.append(layer.register_forward_hook(self._forward_hook))
            self._hooks.append(layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate a Grad-CAM++ heatmap.

        Args:
            input_tensor: Input image tensor of shape ``[1, C, H, W]``.
            class_idx: Target class index.  Uses the predicted class when ``None``.

        Returns:
            Heatmap of shape ``[H, W]`` normalised to ``[0, 1]``.
        """
        self.model.eval()
        input_tensor = input_tensor.detach().requires_grad_(True)

        output = self.model(input_tensor)
        score, class_idx = _get_class_score(output, class_idx)

        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        grads = self.gradients
        acts = self.activations

        if grads.dim() == 4:
            grad_2 = grads.pow(2)
            grad_3 = grads.pow(3)
            sum_acts = acts.sum(dim=(2, 3), keepdim=True)
            denom = 2.0 * grad_2 + sum_acts * grad_3 + 1e-8
            alpha = grad_2 / denom
            alpha = alpha * F.relu(score.exp() * grads)
            weights = alpha.sum(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
        else:
            grad_2 = grads.pow(2)
            grad_3 = grads.pow(3)
            sum_acts = acts.sum(dim=-1, keepdim=True)
            denom = 2.0 * grad_2 + sum_acts * grad_3 + 1e-8
            alpha = grad_2 / denom
            alpha = alpha * F.relu(score.exp() * grads)
            weights = alpha.sum(dim=-1, keepdim=True)
            cam = (weights * acts).sum(dim=-1)

        cam = F.relu(cam).squeeze()
        cam = _handle_transformer_cam(cam)
        cam_np = cam.detach().cpu().numpy()
        return _resize_cam(cam_np, input_tensor.shape[2], input_tensor.shape[3])

    def cleanup(self) -> None:
        """Remove all registered forward / backward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class ScoreCAM:
    """Score-CAM — a gradient-free class activation mapping method.

    Each activation channel is up-sampled to input resolution, used as a
    mask on the input, and the resulting class score is treated as the
    channel weight.  This avoids gradient noise entirely.

    Args:
        model: The classification model.
        target_layer: Dot-separated name of the target layer, or ``None``
            to auto-detect the last convolutional / layer-norm layer.
        batch_size: Number of masked inputs to forward-pass simultaneously.

    Example::

        scam = ScoreCAM(model, target_layer="backbone.layer4")
        heatmap = scam.generate(input_tensor)
        scam.cleanup()
    """

    def __init__(self, model: nn.Module, target_layer: Optional[str] = None, batch_size: int = 16):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self._hooks: list = []
        self.batch_size = batch_size

        layer = _find_target_layer(model, target_layer)
        if layer is not None:
            self._hooks.append(layer.register_forward_hook(self._forward_hook))

    def _forward_hook(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        self.activations = output.detach()

    @torch.no_grad()
    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate a Score-CAM heatmap.

        Args:
            input_tensor: Input image tensor of shape ``[1, C, H, W]``.
            class_idx: Target class index.  Uses the predicted class when ``None``.

        Returns:
            Heatmap of shape ``[H, W]`` normalised to ``[0, 1]``.
        """
        self.model.eval()
        h, w = input_tensor.shape[2], input_tensor.shape[3]

        output = self.model(input_tensor)
        _, class_idx = _get_class_score(output, class_idx)

        if self.activations is None:
            return np.zeros((h, w))

        acts = self.activations

        if acts.dim() == 3:
            acts = _handle_transformer_cam(acts.squeeze(0))
            acts = acts.unsqueeze(0).unsqueeze(0)

        num_channels = acts.shape[1]
        upsampled = F.interpolate(acts, size=(h, w), mode="bilinear", align_corners=False)
        upsampled = upsampled.squeeze(0)

        # normalise each channel to [0, 1]
        for c in range(num_channels):
            ch = upsampled[c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                upsampled[c] = (ch - ch_min) / (ch_max - ch_min)
            else:
                upsampled[c] = torch.zeros_like(ch)

        scores = torch.zeros(num_channels, device=input_tensor.device)
        for start in range(0, num_channels, self.batch_size):
            end = min(start + self.batch_size, num_channels)
            masks = upsampled[start:end].unsqueeze(1)
            masked_inputs = input_tensor * masks
            out = self.model(masked_inputs)
            if out.dim() > 1 and out.shape[1] > 1:
                scores[start:end] = out[:, class_idx]
            else:
                scores[start:end] = out.squeeze()

        scores = F.softmax(scores, dim=0)
        cam = (scores.view(-1, 1, 1) * upsampled).sum(dim=0)
        cam = F.relu(cam)

        cam_np = cam.detach().cpu().numpy()
        return _resize_cam(cam_np, h, w)

    def cleanup(self) -> None:
        """Remove all registered forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class LayerCAM:
    """Layer-CAM — element-wise positive-gradient weighting with spatial detail.

    Computes the element-wise product of positive gradients and activations,
    preserving spatial resolution better than channel-pooled methods.

    Args:
        model: The classification model.
        target_layer: Dot-separated name of the target layer, or ``None``
            to auto-detect the last convolutional / layer-norm layer.

    Example::

        lcam = LayerCAM(model, target_layer="backbone.layer4")
        heatmap = lcam.generate(input_tensor)
        lcam.cleanup()
    """

    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks: list = []

        layer = _find_target_layer(model, target_layer)
        if layer is not None:
            self._hooks.append(layer.register_forward_hook(self._forward_hook))
            self._hooks.append(layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate a Layer-CAM heatmap.

        Args:
            input_tensor: Input image tensor of shape ``[1, C, H, W]``.
            class_idx: Target class index.  Uses the predicted class when ``None``.

        Returns:
            Heatmap of shape ``[H, W]`` normalised to ``[0, 1]``.
        """
        self.model.eval()
        input_tensor = input_tensor.detach().requires_grad_(True)

        output = self.model(input_tensor)
        score, class_idx = _get_class_score(output, class_idx)

        self.model.zero_grad()
        score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        grads = self.gradients
        acts = self.activations

        if grads.dim() == 4:
            positive_grads = F.relu(grads)
            cam = (positive_grads * acts).sum(dim=1, keepdim=True)
        else:
            positive_grads = F.relu(grads)
            cam = (positive_grads * acts).sum(dim=-1)

        cam = F.relu(cam).squeeze()
        cam = _handle_transformer_cam(cam)
        cam_np = cam.detach().cpu().numpy()
        return _resize_cam(cam_np, input_tensor.shape[2], input_tensor.shape[3])

    def cleanup(self) -> None:
        """Remove all registered forward / backward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

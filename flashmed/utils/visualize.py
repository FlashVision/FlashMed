"""Medical image visualization and GradCAM attention maps."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Gradient-weighted Class Activation Mapping for medical image interpretability.

    Generates heatmaps showing which image regions are most important
    for the model's prediction — critical for clinical trust.

    Args:
        model: The classification model
        target_layer: Name of the layer to extract activations from (auto-detected if None)
    """

    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hooks = []

        layer = self._find_target_layer(target_layer)
        if layer is not None:
            self._hooks.append(layer.register_forward_hook(self._forward_hook))
            self._hooks.append(layer.register_full_backward_hook(self._backward_hook))

    def _find_target_layer(self, target_layer: Optional[str] = None) -> Optional[nn.Module]:
        """Find the target layer for GradCAM."""
        if target_layer:
            for name, module in self.model.named_modules():
                if name == target_layer:
                    return module
            return None

        last_conv = None
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                last_conv = module

        if last_conv is None:
            for name, module in self.model.named_modules():
                if "norm" in name.lower() and isinstance(module, nn.LayerNorm):
                    last_conv = module

        return last_conv

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Generate GradCAM heatmap.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            class_idx: Target class (uses predicted class if None)
            normalize: Whether to normalize output to [0, 1]

        Returns:
            Heatmap array of shape [H, W]
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        if class_idx is None:
            if output.dim() > 1 and output.shape[1] > 1:
                class_idx = output.argmax(dim=1).item()
            else:
                class_idx = 0

        self.model.zero_grad()
        if output.dim() > 1 and output.shape[1] > 1:
            target = output[0, class_idx]
        else:
            target = output[0]
        target.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        gradients = self.gradients
        activations = self.activations

        if gradients.dim() == 3:
            weights = gradients.mean(dim=1, keepdim=True)
            cam = (weights * activations).sum(dim=-1)
        else:
            weights = gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = cam.squeeze()

        if cam.dim() == 1:
            num_patches = cam.shape[0] - 1
            side = int(num_patches ** 0.5)
            cam = cam[1:].reshape(side, side)

        cam_np = cam.cpu().numpy()

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        from PIL import Image
        cam_resized = np.array(Image.fromarray(cam_np).resize((w, h)))

        if normalize and cam_resized.max() > cam_resized.min():
            cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())

        return cam_resized

    def cleanup(self):
        """Remove hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


def visualize_medical_image(
    image: np.ndarray,
    title: str = "",
    save_path: Optional[str] = None,
    cmap: str = "gray",
):
    """Display or save a medical image with optional title.

    Args:
        image: Image as numpy array
        title: Plot title
        save_path: If provided, save instead of display
        cmap: Colormap for display
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Visualize] matplotlib not installed, skipping visualization")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Overlay a heatmap on a medical image.

    Args:
        image: Original image [H, W] or [H, W, 3]
        heatmap: Heatmap values [H, W] in range [0, 1]
        alpha: Opacity of heatmap overlay
        colormap: Matplotlib colormap name
        save_path: If provided, save the result

    Returns:
        Overlaid image as numpy array [H, W, 3]
    """
    import cv2

    if image.ndim == 2:
        image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif image.max() <= 1.0:
        image_rgb = (image * 255).astype(np.uint8)
    else:
        image_rgb = image.astype(np.uint8)

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    if heatmap_colored.shape[:2] != image_rgb.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (image_rgb.shape[1], image_rgb.shape[0]))

    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_colored, alpha, 0)

    if save_path:
        from PIL import Image
        Image.fromarray(overlay).save(save_path)

    return overlay


def visualize_segmentation(
    image: np.ndarray,
    mask: np.ndarray,
    num_classes: int = 4,
    alpha: float = 0.5,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Visualize segmentation mask overlaid on image.

    Args:
        image: Original image
        mask: Segmentation mask [H, W] with class indices
        num_classes: Number of classes for colormap
        alpha: Overlay opacity
        save_path: Output path

    Returns:
        Overlaid image
    """
    colors = np.array([
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [128, 128, 0],
    ], dtype=np.uint8)

    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1).astype(np.uint8)
    elif image.max() <= 1.0:
        image_rgb = (image * 255).astype(np.uint8)
    else:
        image_rgb = image.astype(np.uint8)

    mask_colored = colors[mask % len(colors)]
    overlay = (image_rgb * (1 - alpha) + mask_colored * alpha).astype(np.uint8)

    if save_path:
        from PIL import Image
        Image.fromarray(overlay).save(save_path)

    return overlay

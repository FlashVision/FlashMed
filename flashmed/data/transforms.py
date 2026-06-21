"""Medical-specific image transforms and augmentations."""

from typing import Tuple

import numpy as np
from torchvision import transforms as T

from flashmed.registry import TRANSFORMS


@TRANSFORMS.register("CLAHETransform")
class CLAHETransform:
    """Contrast Limited Adaptive Histogram Equalization for medical images."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image):
        import cv2
        img_np = np.array(image)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            img_np = clahe.apply(img_np)
        from PIL import Image
        return Image.fromarray(img_np)


@TRANSFORMS.register("WindowingTransform")
class WindowingTransform:
    """Apply radiological windowing (window center/width) to pixel data."""

    def __init__(self, window_center: float = 40.0, window_width: float = 400.0):
        self.window_center = window_center
        self.window_width = window_width

    def __call__(self, image):
        img_np = np.array(image, dtype=np.float32)
        lower = self.window_center - self.window_width / 2
        upper = self.window_center + self.window_width / 2
        img_np = np.clip(img_np, lower, upper)
        img_np = (img_np - lower) / (upper - lower)
        img_np = (img_np * 255).astype(np.uint8)
        from PIL import Image as PILImage
        if img_np.ndim == 2:
            return PILImage.fromarray(img_np, mode="L")
        return PILImage.fromarray(img_np)


@TRANSFORMS.register("ElasticDeformation")
class ElasticDeformation:
    """Elastic deformation for medical image augmentation."""

    def __init__(self, alpha: float = 50.0, sigma: float = 5.0, p: float = 0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image):
        if np.random.random() > self.p:
            return image
        import cv2
        img_np = np.array(image)
        shape = img_np.shape[:2]

        dx = cv2.GaussianBlur(
            (np.random.rand(*shape) * 2 - 1).astype(np.float32),
            ksize=(0, 0), sigmaX=self.sigma
        ) * self.alpha
        dy = cv2.GaussianBlur(
            (np.random.rand(*shape) * 2 - 1).astype(np.float32),
            ksize=(0, 0), sigmaX=self.sigma
        ) * self.alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        if img_np.ndim == 3:
            result = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR)
        else:
            result = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR)

        from PIL import Image as PILImage
        return PILImage.fromarray(result)


@TRANSFORMS.register("GammaCorrection")
class GammaCorrection:
    """Random gamma correction for intensity augmentation."""

    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.5), p: float = 0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, image):
        if np.random.random() > self.p:
            return image
        gamma = np.random.uniform(*self.gamma_range)
        img_np = np.array(image).astype(np.float32) / 255.0
        img_np = np.power(img_np, gamma)
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        from PIL import Image as PILImage
        return PILImage.fromarray(img_np)


@TRANSFORMS.register("RandomAnatomy")
class RandomAnatomyCrop:
    """Anatomy-aware random cropping that preserves medical ROI."""

    def __init__(self, output_size: Tuple[int, int] = (224, 224), scale: Tuple[float, float] = (0.8, 1.0)):
        self.output_size = output_size
        self.scale = scale

    def __call__(self, image):
        import cv2
        img_np = np.array(image)
        h, w = img_np.shape[:2]

        scale = np.random.uniform(*self.scale)
        new_h, new_w = int(h * scale), int(w * scale)
        top = np.random.randint(0, max(h - new_h, 1))
        left = np.random.randint(0, max(w - new_w, 1))

        cropped = img_np[top:top + new_h, left:left + new_w]
        resized = cv2.resize(cropped, self.output_size[::-1], interpolation=cv2.INTER_LINEAR)

        from PIL import Image as PILImage
        return PILImage.fromarray(resized)


def get_medical_transforms(
    split: str = "train",
    input_size: int = 224,
    modality: str = "xray",
    clahe: bool = True,
) -> T.Compose:
    """Get medical-specific transforms for training/validation.

    Args:
        split: "train" or "val"/"test"
        input_size: Target image size
        modality: One of "xray", "ct", "mri", "pathology"
        clahe: Whether to apply CLAHE preprocessing
    """
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if split == "train":
        transform_list = []
        if clahe:
            transform_list.append(CLAHETransform(clip_limit=2.0))
        transform_list.extend([
            T.Resize((input_size + 32, input_size + 32)),
            T.RandomCrop(input_size),
            T.RandomHorizontalFlip(p=0.5),
        ])
        if modality in ("xray", "ct"):
            transform_list.append(T.RandomRotation(degrees=10))
            transform_list.append(GammaCorrection(gamma_range=(0.8, 1.2), p=0.3))
        elif modality == "pathology":
            transform_list.extend([
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=90),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            ])
        transform_list.extend([T.ToTensor(), normalize])
        return T.Compose(transform_list)
    else:
        transform_list = []
        if clahe:
            transform_list.append(CLAHETransform(clip_limit=2.0))
        transform_list.extend([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            normalize,
        ])
        return T.Compose(transform_list)

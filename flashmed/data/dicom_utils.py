"""DICOM file utilities for medical image loading and processing."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def read_dicom(path: str) -> "pydicom.Dataset":
    """Read a DICOM file and return the dataset object.

    Args:
        path: Path to the .dcm file

    Returns:
        pydicom Dataset object with all DICOM metadata and pixel data.
    """
    import pydicom
    ds = pydicom.dcmread(path)
    return ds


def dicom_to_numpy(
    path: str,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Convert DICOM file to numpy array with optional windowing.

    Args:
        path: Path to DICOM file
        window_center: Center of the display window (uses DICOM metadata if None)
        window_width: Width of the display window (uses DICOM metadata if None)
        normalize: Whether to normalize output to 0-255 uint8

    Returns:
        2D numpy array of pixel values
    """
    import pydicom

    ds = pydicom.dcmread(path)
    pixel_array = ds.pixel_array.astype(np.float32)

    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    if window_center is None and hasattr(ds, "WindowCenter"):
        wc = ds.WindowCenter
        window_center = float(wc[0]) if isinstance(wc, (list, pydicom.multival.MultiValue)) else float(wc)
    if window_width is None and hasattr(ds, "WindowWidth"):
        ww = ds.WindowWidth
        window_width = float(ww[0]) if isinstance(ww, (list, pydicom.multival.MultiValue)) else float(ww)

    if window_center is not None and window_width is not None:
        pixel_array = apply_windowing(pixel_array, window_center, window_width)

    if normalize:
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax > pmin:
            pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
        pixel_array = pixel_array.astype(np.uint8)

    if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = 255 - pixel_array if normalize else pixel_array.max() - pixel_array

    return pixel_array


def apply_windowing(
    pixel_array: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """Apply windowing (contrast adjustment) to pixel data.

    Args:
        pixel_array: Raw pixel values (HU for CT)
        window_center: Center value of the window
        window_width: Width of the window

    Returns:
        Windowed pixel array clipped to [0, 1] range
    """
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    windowed = np.clip(pixel_array, lower, upper)
    windowed = (windowed - lower) / (upper - lower)
    return windowed


def get_dicom_metadata(path: str) -> Dict[str, str]:
    """Extract key metadata fields from a DICOM file.

    Returns dict with: PatientID, Modality, StudyDate, BodyPartExamined,
    ViewPosition, Rows, Columns, SliceThickness, etc.
    """
    import pydicom
    ds = pydicom.dcmread(path, stop_before_pixels=True)

    fields = [
        "PatientID", "PatientName", "PatientAge", "PatientSex",
        "Modality", "StudyDate", "StudyDescription",
        "BodyPartExamined", "ViewPosition",
        "Rows", "Columns", "PixelSpacing", "SliceThickness",
        "Manufacturer", "InstitutionName",
    ]

    metadata = {}
    for field in fields:
        if hasattr(ds, field):
            val = getattr(ds, field)
            metadata[field] = str(val)
    return metadata


def load_dicom_series(directory: str) -> np.ndarray:
    """Load a series of DICOM slices as a 3D volume (for CT/MRI).

    Args:
        directory: Path to directory containing .dcm slice files

    Returns:
        3D numpy array of shape (num_slices, H, W)
    """
    import pydicom

    dicom_dir = Path(directory)
    slices = []

    for dcm_file in sorted(dicom_dir.glob("*.dcm")):
        ds = pydicom.dcmread(str(dcm_file))
        slices.append(ds)

    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]) if hasattr(s, "ImagePositionPatient") else 0)

    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)

    if slices and hasattr(slices[0], "RescaleSlope"):
        volume = volume * float(slices[0].RescaleSlope) + float(slices[0].RescaleIntercept)

    return volume


def dicom_to_pil(path: str, **kwargs) -> "PIL.Image.Image":
    """Convert a DICOM file to a PIL Image for display/processing."""
    from PIL import Image
    pixel_array = dicom_to_numpy(path, normalize=True, **kwargs)
    if pixel_array.ndim == 2:
        return Image.fromarray(pixel_array, mode="L")
    return Image.fromarray(pixel_array)

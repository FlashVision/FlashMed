"""Medical imaging datasets for training and evaluation."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from flashmed.registry import DATASETS


@DATASETS.register("ChestXray14")
class ChestXray14Dataset(Dataset):
    """NIH ChestX-ray14 dataset — 112,120 frontal CXRs with 14 pathology labels.

    Expected directory structure:
        root/
        ├── images/
        │   ├── 00000001_000.png
        │   └── ...
        └── labels.csv  (columns: Image Index, Finding Labels, ...)
    """

    PATHOLOGIES = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia",
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        label_file: Optional[str] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_dir = self.root / "images"
        self.num_classes = 14

        label_path = Path(label_file) if label_file else self.root / "labels.csv"
        self.samples: List[Tuple[str, np.ndarray]] = []
        self._load_labels(label_path)

    def _load_labels(self, label_path: Path):
        """Parse the CSV label file into multi-hot encoded labels."""
        if not label_path.exists():
            image_dir = self.image_dir if self.image_dir.exists() else self.root
            if image_dir.exists():
                for img_file in sorted(image_dir.glob("*.png")):
                    self.samples.append((str(img_file), np.zeros(14, dtype=np.float32)))
            return

        import csv
        with open(label_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["Image Index"]
                findings = row["Finding Labels"]
                label_vec = np.zeros(self.num_classes, dtype=np.float32)
                if findings != "No Finding":
                    for finding in findings.split("|"):
                        finding = finding.strip()
                        if finding in self.PATHOLOGIES:
                            label_vec[self.PATHOLOGIES.index(finding)] = 1.0
                self.samples.append((str(self.image_dir / img_name), label_vec))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            from torchvision import transforms as T
            image = T.Compose([T.Resize((224, 224)), T.ToTensor()])(image)
        return image, torch.from_numpy(label)


@DATASETS.register("ISIC")
class ISICDataset(Dataset):
    """ISIC Skin Lesion Dataset for dermoscopy classification.

    Expected structure:
        root/
        ├── images/
        │   ├── ISIC_0000000.jpg
        │   └── ...
        └── metadata.csv
    """

    LESION_TYPES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.num_classes = len(self.LESION_TYPES)
        self.samples: List[Tuple[str, int]] = []
        self._scan_directory()

    def _scan_directory(self):
        """Scan for images and optionally parse metadata."""
        image_dir = self.root / "images"
        meta_path = self.root / "metadata.csv"

        if meta_path.exists():
            import csv
            with open(meta_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_name = row.get("image", row.get("image_id", ""))
                    dx = row.get("dx", "UNK").upper()
                    label = self.LESION_TYPES.index(dx) if dx in self.LESION_TYPES else len(self.LESION_TYPES) - 1
                    img_path = image_dir / f"{img_name}.jpg"
                    if img_path.exists():
                        self.samples.append((str(img_path), label))
        elif image_dir.exists():
            for img_file in sorted(image_dir.glob("*.jpg")):
                self.samples.append((str(img_file), 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            from torchvision import transforms as T
            image = T.Compose([T.Resize((224, 224)), T.ToTensor()])(image)
        return image, torch.tensor(label, dtype=torch.long)


@DATASETS.register("BraTS")
class BraTSDataset(Dataset):
    """BraTS Brain Tumor Segmentation Dataset (3D MRI volumes).

    Expected structure:
        root/
        ├── BraTS20_Training_001/
        │   ├── *_t1.nii.gz
        │   ├── *_t1ce.nii.gz
        │   ├── *_t2.nii.gz
        │   ├── *_flair.nii.gz
        │   └── *_seg.nii.gz
        └── ...
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        roi_size: Tuple[int, ...] = (128, 128, 128),
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.roi_size = roi_size
        self.samples: List[Dict[str, str]] = []
        self._scan_volumes()

    def _scan_volumes(self):
        """Find all patient volumes."""
        if not self.root.exists():
            return
        for patient_dir in sorted(self.root.iterdir()):
            if not patient_dir.is_dir():
                continue
            modalities = {}
            for nii_file in patient_dir.glob("*.nii.gz"):
                name = nii_file.stem.replace(".nii", "")
                if "_t1ce" in name:
                    modalities["t1ce"] = str(nii_file)
                elif "_t1" in name:
                    modalities["t1"] = str(nii_file)
                elif "_t2" in name:
                    modalities["t2"] = str(nii_file)
                elif "_flair" in name:
                    modalities["flair"] = str(nii_file)
                elif "_seg" in name:
                    modalities["seg"] = str(nii_file)
            if modalities:
                self.samples.append(modalities)

    def _load_nifti(self, path: str) -> np.ndarray:
        """Load a NIfTI volume as numpy array."""
        try:
            import nibabel as nib
            img = nib.load(path)
            return img.get_fdata().astype(np.float32)
        except ImportError:
            return np.zeros(self.roi_size, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        channels = []
        for mod in ["t1", "t1ce", "t2", "flair"]:
            if mod in sample:
                vol = self._load_nifti(sample[mod])
                vol = (vol - vol.mean()) / (vol.std() + 1e-8)
                channels.append(vol)
            else:
                channels.append(np.zeros(self.roi_size, dtype=np.float32))

        image = np.stack(channels, axis=0)
        seg = self._load_nifti(sample["seg"]) if "seg" in sample else np.zeros(self.roi_size, dtype=np.float32)

        if self.transform:
            result = self.transform({"image": image, "label": seg})
            return result["image"], result["label"]

        image_tensor = torch.from_numpy(image)
        seg_tensor = torch.from_numpy(seg).long()
        return image_tensor, seg_tensor


@DATASETS.register("PathMNIST")
class PathMNISTDataset(Dataset):
    """PathMNIST from MedMNIST — colon pathology classification (9 tissue types).

    Expected structure:
        root/
        └── pathmnist.npz  (keys: train_images, train_labels, val_images, ...)
    """

    TISSUE_TYPES = [
        "Adipose", "Background", "Debris", "Lymphocytes",
        "Mucus", "Smooth_Muscle", "Normal_Colon_Mucosa",
        "Cancer_Epithelium", "Stroma",
    ]

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.num_classes = 9
        self.images: np.ndarray = np.array([])
        self.labels: np.ndarray = np.array([])
        self._load_data()

    def _load_data(self):
        """Load from .npz file."""
        npz_path = self.root / "pathmnist.npz"
        if not npz_path.exists():
            return
        data = np.load(str(npz_path))
        split_key = {"train": "train", "val": "val", "test": "test"}.get(self.split, "train")
        self.images = data[f"{split_key}_images"]
        self.labels = data[f"{split_key}_labels"].flatten()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = int(self.labels[idx])

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        pil_image = Image.fromarray(image.astype(np.uint8))
        if self.transform:
            pil_image = self.transform(pil_image)
        else:
            from torchvision import transforms as T
            pil_image = T.Compose([T.Resize((28, 28)), T.ToTensor()])(pil_image)

        return pil_image, torch.tensor(label, dtype=torch.long)


@DATASETS.register("MIMIC-CXR")
class MIMICCXRDataset(Dataset):
    """MIMIC-CXR dataset for report generation and multi-label classification.

    Expected structure:
        root/
        ├── images/
        │   └── p10/p10000032/s50414267/*.dcm or *.jpg
        └── reports.csv  (columns: subject_id, study_id, report_text, labels)
    """

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None, task: str = "report"):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.task = task
        self.samples: List[Dict[str, str]] = []
        self._load_metadata()

    def _load_metadata(self):
        """Parse report/label CSV."""
        csv_path = self.root / "reports.csv"
        if not csv_path.exists():
            return

        import csv
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "image_path": str(self.root / "images" / row.get("dicom_path", "")),
                    "report": row.get("report_text", ""),
                    "labels": row.get("labels", ""),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        img_path = sample["image_path"]

        if img_path.endswith(".dcm"):
            from flashmed.data.dicom_utils import dicom_to_numpy
            image = Image.fromarray(dicom_to_numpy(img_path)).convert("RGB")
        else:
            try:
                image = Image.open(img_path).convert("RGB")
            except (FileNotFoundError, OSError):
                image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)
        else:
            from torchvision import transforms as T
            image = T.Compose([T.Resize((224, 224)), T.ToTensor()])(image)

        return {"image": image, "report": sample["report"], "labels": sample["labels"]}


def get_dataset(name: str, **kwargs) -> Dataset:
    """Get a dataset by name from the registry."""
    return DATASETS.build(name, **kwargs)

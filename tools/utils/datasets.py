# tools/utils/datasets.py
from __future__ import annotations

import os
from torchvision.datasets import ImageFolder


def _assert_dir(path: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset path not found: {path}")


def load_dataset(
    name: str,
    root: str,
):
    """
    Use only local dataset, no download.

    Directory structure requirements:
    - EuroSAT:          data/eurosat/2750/<class>/*.jpg
    - UCMerced_LandUse: data/UCMerced_LandUse/Images/<class>/*.jpg
    - NWPU_RESISC45:    data/NWPU_RESISC45/<class>/*.jpg
    """
    n = name.lower()

    if n in ("eurosat", "eurosat_rgb"):
        img_root = os.path.join(root, "eurosat", "2750")

    elif n in ("ucm", "ucmerced", "ucmerced_landuse"):
        img_root = os.path.join(root, "UCMerced_LandUse", "Images")

    elif n in ("resisc45", "nwpu_resisc45", "nwpu_resisc45"):
        img_root = os.path.join(root, "NWPU_RESISC45")

    else:
        raise ValueError(
            f"Unknown dataset name: {name}. "
            f"Supported: eurosat / ucm / resisc45"
        )

    _assert_dir(img_root)
    return ImageFolder(root=img_root, transform=None)
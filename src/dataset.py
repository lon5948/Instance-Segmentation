from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import skimage.io as sio
import torch
from torch.utils.data import Dataset


# ───────────────────────── helpers ──────────────────────────
def extract_instance_masks(
    mask_array: np.ndarray,
) -> Tuple[List[int], List[np.ndarray]]:
    inst_ids = np.unique(mask_array)
    inst_ids = inst_ids[inst_ids != 0]  # drop background
    ids, masks = [], []
    for i in sorted(inst_ids):
        ids.append(int(i))
        masks.append((mask_array == i).astype(np.uint8))
    return ids, masks


def compute_bbox(bin_mask: np.ndarray) -> List[int]:
    rows = np.any(bin_mask, axis=1)
    cols = np.any(bin_mask, axis=0)
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return [int(x0), int(y0), int(x1), int(y1)]


class CellInstanceDataset(Dataset):
    """
    Works for *both* training (folder-per-sample) and test (flat .tif files).

    When `train=False` you **must** supply `id_map` created from
    `test_image_name_to_ids.json` so the exported `image_id`
    matches the leaderboard.
    """

    def __init__(
        self,
        root: str | Path,
        classes: List[int] = [1, 2, 3, 4],
        transforms=None,
        train: bool = True,
        id_map: Dict[str, int] | None = None,
    ):
        self.root = Path(root)
        self.transforms = transforms
        self.classes = classes
        self.train = bool(train)
        self.id_map = id_map or {}

        self.samples = (
            sorted(p for p in self.root.iterdir() if p.is_dir())
            if self.train
            else sorted(self.root.glob("*.tif"))
        )

    def _official_id(self, fp: Path, fallback: int) -> int:
        return self.id_map.get(fp.name, fallback)

    def __getitem__(self, idx: int):
        if self.train:
            folder = self.samples[idx]
            img = sio.imread(folder / "image.tif")
            masks, labels = [], []

            for cls in self.classes:
                fp = folder / f"class{cls}.tif"
                if not fp.exists():
                    continue
                inst = sio.imread(fp).astype(np.int32)
                _, bin_masks = extract_instance_masks(inst)
                masks += [torch.as_tensor(m, dtype=torch.uint8) for m in bin_masks]
                labels += [cls] * len(bin_masks)

            if not masks:  # safeguard
                masks = [torch.zeros(img.shape[:2], dtype=torch.uint8)]
                labels = [0]

            # Convert image to correct format - ensure it's 3 channels
            if len(img.shape) == 2:  # If grayscale, convert to 3-channel
                img = np.stack([img, img, img], axis=2)
            elif img.shape[2] == 4:  # If RGBA, drop alpha channel
                img = img[:, :, :3]

            masks_t = torch.stack(masks)
            boxes_t = torch.as_tensor(
                [compute_bbox(m.numpy()) for m in masks], dtype=torch.float32
            )
            area_t = (masks_t > 0).flatten(1).sum(1).float()

            target = {
                "boxes": boxes_t,
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": masks_t,
                "image_id": torch.as_tensor([idx]),
                "area": area_t,
                "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
            }
        else:
            fp = self.samples[idx]
            img = sio.imread(fp)

            # Convert image to correct format - ensure it's 3 channels
            if len(img.shape) == 2:  # If grayscale, convert to 3-channel
                img = np.stack([img, img, img], axis=2)
            elif img.shape[2] == 4:  # If RGBA, drop alpha channel
                img = img[:, :, :3]

            img_id = self._official_id(fp, idx + 1)
            target = {"image_id": torch.as_tensor([img_id])}

        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.samples)

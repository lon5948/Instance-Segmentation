from __future__ import annotations

import random

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F


class ToTensor:
    """Convert (H,W,3) uint8 ndarray to *normalised* torch Tensor (0..1)."""

    def __call__(self, image):
        if image.ndim == 2:
            image = image[:, :, np.newaxis].repeat(3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        return F.to_tensor(image)


class AddGaussianNoise(torch.nn.Module):
    """Add i.i.d Gaussian noise to a tensor image with a given prob."""

    def __init__(self, mean: float = 0.0, std: float = 0.02, p: float = 0.5):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, tensor: torch.Tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            tensor = tensor + noise
            tensor = tensor.clamp(0.0, 1.0)
        return tensor


class RandomCellIntensityAdjustment(torch.nn.Module):
    """Randomly adjusts the intensity of cells to simulate staining variation."""

    def __init__(self, intensity_range=(0.85, 1.15), p=0.5):
        super().__init__()
        self.intensity_range = intensity_range
        self.p = p

    def forward(self, tensor: torch.Tensor):
        if torch.rand(1).item() < self.p:
            # Apply more to darker regions (likely cells)
            mask = tensor < 0.7
            factor = random.uniform(*self.intensity_range)
            tensor = tensor.clone()
            tensor[mask] = (tensor[mask] * factor).clamp(0.0, 1.0)
        return tensor


class RandomHistogramEqualization(torch.nn.Module):
    """Randomly applies histogram equalization to improve contrast."""

    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, tensor: torch.Tensor):
        if torch.rand(1).item() < self.p:
            # Convert to numpy for histogram processing
            img_np = tensor.permute(1, 2, 0).cpu().numpy()

            # Apply to each channel
            for c in range(img_np.shape[2]):
                img_flat = img_np[:, :, c].flatten()
                hist, bins = np.histogram(img_flat, 256, [0, 1])
                cdf = hist.cumsum()
                cdf_normalized = cdf * img_flat.max() / cdf.max()

                # Create lookup table
                lut = np.interp(img_flat, bins[:-1], cdf_normalized)
                img_np[:, :, c] = lut.reshape(img_np[:, :, c].shape)

            # Convert back to tensor
            tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
            tensor = tensor.clamp(0.0, 1.0)

        return tensor


def get_train_transforms():
    """Enhanced augmentation pipeline for training.

    Adds more sophisticated transformations appropriate for cell imaging.
    """
    return T.Compose(
        [
            ToTensor(),
            # Color adjustments
            T.RandomApply(
                [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03)],
                p=0.9,
            ),
            # Cell-specific intensity adjustments
            RandomCellIntensityAdjustment(intensity_range=(0.8, 1.2), p=0.7),
            # Contrast enhancement
            RandomHistogramEqualization(p=0.3),
            # Focus variation
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            # More aggressive noise to improve robustness
            AddGaussianNoise(mean=0.0, std=0.03, p=0.6),
        ]
    )


def get_test_transforms():
    """Deterministic preprocessing for validation / inference."""
    return T.Compose(
        [
            ToTensor(),
        ]
    )

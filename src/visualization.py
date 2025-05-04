import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
from skimage.io import imread, imsave

from src.utils import decode_maskobj


def to_rgb(img: np.ndarray) -> np.ndarray:
    """Return a 3-channel RGB copy of *img*.

    Handles grayscale (H,W) and images that include an alpha channel (H,W,4).
    """
    if img is None:
        return None
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 4:
        return img[..., :3]
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError(f"Unsupported image shape: {img.shape}")


def build_mask(preds):
    """Return a (H,W) int32 mask (pixel == instance-ID)."""
    preds = sorted(preds, key=lambda p: p["score"], reverse=True)
    H, W = preds[0]["segmentation"]["size"]
    inst = np.zeros((H, W), dtype=np.int32)
    for inst_id, p in enumerate(preds, 1):
        m = decode_maskobj(p["segmentation"]).astype(bool)
        inst[m] = inst_id
    return inst


def main():
    parser = argparse.ArgumentParser(description="Convert RLE preds to mask")
    parser.add_argument("--preds", required=True, type=Path,
                        help="JSON file with COCO-style predictions")
    parser.add_argument("--image", type=Path, default=None,
                        help="Original image (optional, for visualisation)")
    parser.add_argument("--id", required=True, type=int,
                        help="image_id to visualise")
    args = parser.parse_args()

    with args.preds.open() as f:
        preds_all = json.load(f)

    preds_img = [p for p in preds_all if p["image_id"] == args.id]
    if not preds_img:
        raise ValueError(f"No predictions for image_id={args.id} in {args.preds}")

    mask = build_mask(preds_img)

    img = imread(args.image) if args.image else None
    img_rgb = to_rgb(img)

    blend = label2rgb(mask, image=img_rgb, bg_label=0, alpha=0.4, kind="overlay")

    fig, ax = plt.subplots(1, 3, figsize=(14, 5), tight_layout=True)
    ax[0].imshow(img_rgb if img_rgb is not None else mask, cmap="gray")
    ax[0].set_title("raw" if img_rgb is not None else "instance mask")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="nipy_spectral", interpolation="nearest")
    ax[1].set_title("instance mask")
    ax[1].axis("off")

    ax[2].imshow(blend)
    ax[2].set_title("overlay")
    ax[2].axis("off")

    plt.show()


if __name__ == "__main__":
    main()

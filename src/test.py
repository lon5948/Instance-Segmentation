from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.utils import encode_mask


def _coco_bbox(box_tensor):
    x0, y0, x1, y1 = box_tensor.tolist()
    return [x0, y0, x1 - x0, y1 - y0]


def run_prediction(model, loader, device, outfile: str):
    model.eval()
    results: List[Dict] = []

    for imgs, tgts in loader:
        imgs = [i.to(device) for i in imgs]
        outs = model(imgs)

        for tgt, out in zip(tgts, outs):
            img_id = int(tgt["image_id"].item())
            masks = (out["masks"].squeeze(1).cpu() > 0.5).numpy()

            for m, box, score, label in zip(
                masks,
                out["boxes"].cpu(),
                out["scores"].cpu(),
                out["labels"].cpu(),
            ):
                results.append(
                    {
                        "image_id": img_id,
                        "bbox": _coco_bbox(box),
                        "score": float(score),
                        "category_id": int(label),
                        "segmentation": encode_mask(m.astype(np.uint8)),
                    }
                )

    Path(outfile).write_text(json.dumps(results))
    print(f"Wrote {len(results):,} predictions → {outfile}")

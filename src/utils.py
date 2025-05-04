import random
from typing import Dict

import numpy as np
import torch
from pycocotools import mask as mask_utils


# ───────────────────────── reproducibility ─────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ───────────────────────── RLE helpers ─────────────────────────────
def encode_mask(bin_mask: np.ndarray) -> Dict:
    rle = mask_utils.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def decode_maskobj(mask_obj):
    return mask_utils.decode(mask_obj)

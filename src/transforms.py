import numpy as np
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


def get_train_transforms():
    return T.Compose(
        [
            ToTensor(),
        ]
    )


def get_test_transforms():
    return T.Compose(
        [
            ToTensor(),
        ]
    )

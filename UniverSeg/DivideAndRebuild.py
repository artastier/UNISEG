# @author Arthur Astier
from skimage import exposure
from typing import List, Tuple
import numpy as np
import sys


def divide(img, sector_indices=None):
    divided_img = []
    img = exposure.rescale_intensity(img, out_range=(0., 1.))
    img_shape = np.shape(img)
    if img_shape[0] % 128 != 0 or img_shape[1] % 128 != 0:
        print("\n The width or the height of the image isn't a multiple of 128 \n", file=sys.stderr)
    if sector_indices is None:
        non_void = []
        width = img_shape[1] // 128
        height = img_shape[0] // 128
        for i in range(height):
            for j in range(width):
                extracted_image = img[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]
                if not np.all(extracted_image >= 0.95):
                    divided_img.append(extracted_image)
                    non_void.append((i, j))
        return divided_img, non_void
    else:
        for i, j in sector_indices:
            divided_img.append(img[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128])
        return divided_img


def rebuild(divided_imgs: List[np.ndarray], im_size: Tuple[int, int], sector_indices: List[Tuple[int, int]]):
    img = np.ones(im_size, dtype=float)
    for idx, (i, j) in enumerate(sector_indices):
        img[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = divided_imgs[idx]
    return img

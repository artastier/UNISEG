"""
@file
@brief This script provides functions for dividing and rebuilding images.

@author Arthur Astier

@section dependencies
- skimage.exposure
- torch
- typing.Tuple
- skimage.util.invert
- numpy

@section functions
- @b divide: Divides an image into smaller patches.
- @b rebuild: Rebuilds an image from smaller patches.

@section parameters Parameters
- @b img: Input image to be divided or rebuilt.
- @b to_invert: Boolean flag to invert the image patches (optional).
- @b divided_imgs: Divided image patches.
- @b img_shape: Shape of the original image.

"""

__author__ = "Arthur Astier"

from skimage import exposure
import torch
from typing import Tuple
from skimage.util import invert
import numpy as np
import sys


def divide(img, to_invert: bool = False):
    """
    Divides an image into smaller patches.

    @param img: Input image to be divided.
    @param to_invert: Boolean flag to invert the image patches (default is False).
    @return: Tensor containing divided image patches.
    """
    img = exposure.rescale_intensity(img, out_range=(0., 1.))
    img_shape = np.shape(img)
    if img_shape[0] % 128 != 0 or img_shape[1] % 128 != 0:
        print("\n The width or the height of the image isn't a multiple of 128 \n", file=sys.stderr)
        return None
    else:
        width = img_shape[1] // 128
        height = img_shape[0] // 128
        divided_img = torch.zeros(width * height, 1, 128, 128)
        idx = 0
        for i in range(height):
            for j in range(width):
                if to_invert:
                    divided_img[idx, 0] = torch.from_numpy(invert(img[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]))
                else:
                    divided_img[idx, 0] = torch.from_numpy(img[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128])
                idx += 1
        return divided_img


def rebuild(divided_imgs, img_shape: Tuple[int, int]):
    """
    Rebuilds an image from smaller patches.

    @param divided_imgs: Tensor containing divided image patches.
    @param img_shape: Shape of the original image.
    @return: Reconstructed image.
    """
    img = np.ones(img_shape, dtype=float)
    width = img_shape[1] // 128
    height = img_shape[0] // 128
    idx = 0
    for i in range(height):
        for j in range(width):
            img[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = divided_imgs[idx, 0]
            idx += 1
    return img

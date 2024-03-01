# @author Arthur Astier
import skimage.io as io
from skimage import color
import numpy as np
import os
import sys


def compare(mask_path: str, scans_path: str):
    enhanced_images = []
    filenames = os.listdir(os.path.join(os.getcwd(),scans_path))
    for filename in filenames:
        mask_file = os.path.join(mask_path, filename)
        scan_file = os.path.join(scans_path, filename)
        if not os.path.isfile(mask_file) or not os.path.isfile(scan_file):
            print("\n We can neither find the ground truth mask nor the test image \n", file=sys.stderr)
            continue
        mask = io.imread(mask_file, as_gray=True).astype(float)
        scan_img = io.imread(scan_file, as_gray=True).astype(float)
        enhanced_images.append(color_enhanced(scan_img, mask))
    return enhanced_images


def color_enhanced(test_image, gt_mask):
    enhanced_image = color.gray2rgb(test_image).copy()

    gt_mask_idx = np.where(gt_mask == 0)
    gt_mask_nb_pixels = gt_mask_idx[0].shape
    enhanced_image[gt_mask_idx] = np.repeat(np.array([[0, 1.0, 0]]), gt_mask_nb_pixels, axis=0)

    return enhanced_image

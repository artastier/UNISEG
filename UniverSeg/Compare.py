from Segmenter import Segmenter
import skimage.io as io
from skimage import color
import numpy as np
import os
import sys


def compare(ground_truth_path: str, test_folder_path: str, segmenter: Segmenter):
    enhanced_images = []

    for idx, filename in enumerate(segmenter.filenames):
        gt_mask_file = os.path.join(ground_truth_path, filename)
        test_mask_file = os.path.join(test_folder_path, filename)
        if not os.path.isfile(gt_mask_file) or not os.path.isfile(test_mask_file):
            print("\n We can neither find the ground truth mask nor the test image \n", file=sys.stderr)
            continue
        gt_mask = io.imread(gt_mask_file, as_gray=True).astype(float)
        test_img = io.imread(test_mask_file, as_gray=True).astype(float)
        predicted_mask = segmenter.segmented_images[idx]
        if gt_mask.shape != predicted_mask.shape:
            print("\n The shape of the ground truth mask doesn't match with the shape of the predicted mask \n",
                  file=sys.stderr)
            continue
        enhanced_images.append(color_enhanced(test_img, gt_mask, predicted_mask))
    return enhanced_images


def color_enhanced(test_image, gt_mask, predicted_mask):
    enhanced_image = color.gray2rgb(test_image).copy()

    gt_mask_idx = np.where(gt_mask == 0)
    gt_mask_nb_pixels = gt_mask_idx[0].shape
    enhanced_image[gt_mask_idx] = np.repeat(np.array([[0, 1.0, 0]]), gt_mask_nb_pixels, axis=0)

    predicted_mask_idx = np.where(predicted_mask == 0)
    predicted_mask_nb_pixels = predicted_mask_idx[0].shape
    enhanced_image[predicted_mask_idx] = np.repeat(np.array([[1.0, 0, 0]]), predicted_mask_nb_pixels, axis=0)

    intersection_idx = np.where(gt_mask+predicted_mask == 0)
    intersection_nb_pixels = intersection_idx[0].shape
    enhanced_image[intersection_idx] = np.repeat(np.array([[1.0, 1.0, 0]]), intersection_nb_pixels, axis=0)
    return enhanced_image

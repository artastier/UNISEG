"""
@file
@brief This script defines functions for comparing predicted segmentation masks with ground truth masks.

@author Arthur Astier

@section dependencies
- Segmenter
- skimage.util.invert
- skimage.io
- skimage.color
- skimage.exposure
- numpy
- os
- sys

@section functions
- @b compare: Compares predicted segmentation masks with ground truth masks.
- @b color_enhanced: Enhances images with color to visualize comparison and return the dice score of the predicted
 segmentation.

@section parameters
- @b ground_truth_path: Path to the folder containing ground truth masks.
- @b test_folder_path: Path to the folder containing test images.
- @b segmenter: Segmenter object containing segmented images.
- @b invert_label: Boolean flag to invert label images (optional, default is True).

"""

__author__ = "Arthur Astier"

from Segmenter import Segmenter
from skimage.util import invert
import skimage.io as io
from skimage import color
from skimage import exposure
import numpy as np
import os
import sys


def compare(ground_truth_path: str, test_folder_path: str, segmenter: Segmenter, invert_label=True):
    """
    Compares predicted segmentation masks with ground truth masks.

    @param ground_truth_path: Path to the folder containing ground truth masks.
    @param test_folder_path: Path to the folder containing test images.
    @param segmenter: Segmenter object containing segmented images.
    @param invert_label: Boolean flag to invert label images (default is True).
    @return: Enhanced images and corresponding thresholds.
    """
    enhanced_images = []
    dice_scores = []
    for idx, filename in enumerate(segmenter.filenames):
        gt_mask_file = os.path.join(ground_truth_path, filename)
        test_mask_file = os.path.join(test_folder_path, filename)
        if not os.path.isfile(gt_mask_file) or not os.path.isfile(test_mask_file):
            print("\n We can neither find the ground truth mask nor the test image \n", file=sys.stderr)
            continue
        if invert_label:
            gt_mask = invert(io.imread(gt_mask_file, as_gray=True).astype(float))
        else:
            gt_mask = io.imread(gt_mask_file, as_gray=True).astype(float)
        test_img = io.imread(test_mask_file, as_gray=True).astype(float)
        prediction = segmenter.segmented_images[idx]
        if gt_mask.shape != prediction.shape:
            print("\n The shape of the ground truth mask doesn't match with the shape of the predicted mask \n",
                  file=sys.stderr)
            continue
        enhanced_img, dice_score = color_enhanced(test_img, gt_mask, prediction)
        enhanced_images.append(enhanced_img)
        dice_scores.append(dice_score)
    return enhanced_images, dice_scores


def color_enhanced(test_image, gt_mask, predicted_mask):
    """
    Enhances images with color to visualize comparison.

    @param test_image: Test image.
    @param gt_mask: Ground truth mask.
    @param predicted_mask: Predicted mask.
    @return: Enhanced image and dice score.
    """
    enhanced_image = color.gray2rgb(test_image).copy()

    gt_mask_idx = np.where(gt_mask == 1)
    gt_mask_nb_pixels = gt_mask_idx[0].shape
    enhanced_image[gt_mask_idx] += np.repeat(np.array([[0, 0.5, 0]]), gt_mask_nb_pixels, axis=0)

    predicted_mask_idx = np.where(predicted_mask == 1)
    predicted_mask_nb_pixels = predicted_mask_idx[0].shape
    enhanced_image[predicted_mask_idx] += np.repeat(np.array([[0.5, 0, 0]]), predicted_mask_nb_pixels, axis=0)

    intersection_idx = np.where((gt_mask + predicted_mask) / 2 == 1)
    intersection_nb_pixels = intersection_idx[0].shape
    enhanced_image[intersection_idx] += np.repeat(np.array([[0.3, 0.3, 0]]), intersection_nb_pixels, axis=0)

    dice_score = 2 * len(intersection_idx[0]) / (len(gt_mask_idx[0]) + len(predicted_mask_idx[0]))

    return exposure.rescale_intensity(enhanced_image, out_range=(0., 1.)), dice_score

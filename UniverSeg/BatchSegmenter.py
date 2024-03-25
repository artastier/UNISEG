"""
@file
@brief BatchSegmenter class for segmenting batches of images.

@author Arthur Astier

@section dependencies
- Support
- Segmenter
- Compare
- skimage.exposure
- matplotlib.pyplot
- os
- sys

@section functions
- @b BatchSegmenter: Class for segmenting batches of images.
- @b create_and_segment_batches: Method to create and segment batches of images.
- @b save_results: Method to save the segmented batches and logs.

@section parameters
- @b map_paths: List of paths to map folders.
- @b label_paths: List of paths to label folders.
- @b test_path: Path to the test folder.
- @b gt_path: Path to the ground truth folder.
- @b batch_size: Number of batches to process.
- @b support_size: Size of the support.
- @b invert_label: Boolean flag to invert label images (optional, default is True).
"""
__author__ = "Arthur Astier"

import os
from skimage import exposure
import matplotlib.pyplot as plt
from Support import Support
from Segmenter import Segmenter
from Compare import compare
from random import shuffle
from typing import List
import sys


class BatchSegmenter:
    """
    BatchSegmenter class is responsible for batch processing of segmentation tasks.
    """

    def __init__(self, map_paths: List[str], label_paths: List[str], test_path: str, gt_path: str, batch_size: int,
                 support_size: int,
                 invert_label: bool = True):
        """
        Constructor method for BatchSegmenter.

        :param map_paths: List of paths to the maps.
        :param label_paths: List of paths to the labels.
        :param test_path: Path to the test data.
        :param gt_path: Path to the ground truth data.
        :param batch_size: Size of each batch.
        :param support_size: Size of the support.
        :param invert_label: Flag indicating whether to invert the label or not.
        """
        self.batch_size = batch_size
        self.segmented_batches, self.support_batches, self.test_filenames, self.dice_scores = BatchSegmenter.create_and_segment_batches(
            map_paths, label_paths, test_path, gt_path,
            invert_label,
            batch_size,
            support_size)

    @staticmethod
    def create_and_segment_batches(map_paths: List[str], label_paths: List[str], test_path: str, gt_path: str,
                                   invert_label: bool,
                                   batch_size: int,
                                   support_size: int):
        """
        Static method to create and segment batches.

        :param map_paths: List of paths to the maps.
        :param label_paths: List of paths to the labels.
        :param test_path: Path to the test data.
        :param gt_path: Path to the ground truth data.
        :param invert_label: Flag indicating whether to invert the label or not.
        :param batch_size: Size of each batch.
        :param support_size: Size of the support.
        :return: Segmented batches, support batches, test filenames, and dice scores.
        """
        segmented_batches = []
        support_batches = []
        dice_scores = []
        if len(map_paths) > 1 and len(label_paths) > 1 and (
                len(map_paths) != batch_size or len(label_paths) != batch_size):
            print("\n You haven't supplied as many folders as the number of batches requested \n", file=sys.stderr)
            return None, None, None
        unique_support = False
        if len(map_paths) == 1 and len(label_paths) == 1:
            unique_support = True
            filenames = os.listdir(map_paths[0])
        for batch in range(batch_size):
            if not unique_support:
                filenames = os.listdir(map_paths[batch])
            shuffle(filenames)
            if len(filenames) < support_size:
                print(
                    "\n The folders you supplied doesn't contain enough files compared to the support size requested. "
                    "\n",
                    file=sys.stderr)
                continue
            support_batch = filenames[:support_size]
            support_batches.append(support_batch)
            if unique_support:
                support = Support(map_paths[0], label_paths[0], support_batch, invert_label=invert_label)
            else:
                support = Support(map_paths[batch], label_paths[batch], support_batch, invert_label=invert_label)
            print(f"Batch n°{batch + 1} - Support size: {support.maps.shape[1]}")
            segmenter = Segmenter(test_path, support)
            enhanced_images, dice_batch = compare(gt_path, test_path, segmenter)
            segmented_batches.append(enhanced_images)
            dice_scores.append(dice_batch)
        return segmented_batches, support_batches, os.listdir(test_path), dice_scores

    def save_results(self):
        """
        Method to save the segmentation results.
        """
        if not os.path.exists(os.path.join(os.getcwd(), 'Results')):
            os.mkdir(os.path.join(os.getcwd(), 'Results'))
        for batch in range(self.batch_size):
            segmented_images = self.segmented_batches[batch]
            for idx, image in enumerate(segmented_images):
                filename = self.test_filenames[idx]
                subject_directory = 'Results/' + filename
                if not os.path.exists(os.path.join(os.getcwd(), subject_directory)):
                    os.mkdir(os.path.join(os.getcwd(), subject_directory))
                image = exposure.rescale_intensity(image, out_range=(0., 1.))
                plt.imsave(subject_directory + f'/Batch_n°{batch + 1}_' + filename, image)
                with open(os.path.join(subject_directory, "logs.txt"), 'a') as file:
                    file.write(f"\nSupport Batch n°{batch + 1}:\n")
                    for element in self.support_batches[batch]:
                        file.write(str(element) + '\n')
                    dice_score = self.dice_scores[batch][idx]
                    file.write(f'Dice score: {dice_score} \n')
                    file.close()

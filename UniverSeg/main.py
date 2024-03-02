"""
@file
@brief This script is used for batch segmentation of images.
@details It segments a batch of images using a specified batch size and support size.

@author Arthur Astier

@section intro Introduction
This script is designed to perform batch segmentation of images using the BatchSegmenter class.

@section usage
To use this script, specify the paths to the map, label, test, and ground truth folders, as well as other parameters like batch size and support size.

@section dependencies
- PyQt5.QtWidgets
- PathSelection
- BatchSegmenter

@section parameters
- @b app: QApplication object to handle application event loop.
- @b batch_size: Number of images to process in each batch.
- @b map_path: List of paths to the map folders.
- @b label_path: List of paths to the label folders.
- @b test_path: Path to the test folder.
- @b ground_truth_path: Path to the ground truth masks folder.
- @b invert_label: Boolean flag to invert the labels.
- @b batch_segmenter: BatchSegmenter object for performing batch segmentation.
"""

__author__ = "Arthur Astier"

from PyQt5.QtWidgets import QApplication
from PathSelection import PathSelection
import sys
from BatchSegmenter import BatchSegmenter

if __name__ == "__main__":
    app = QApplication(sys.argv)
    batch_size = 5
    # TODO: Adapt PathSelection to choose multiple folders depending on the batch_size and unique_support (bool)
    map_path = ["Support/Body" for i in range(batch_size)]  # PathSelection("Select the Map folder").directory
    label_path = ["Support/Body_mask" for i in range(batch_size)]  # PathSelection("Select the Label folder").directory
    test_path = "Test/Original/Body"  # PathSelection("Select the Test folder").directory
    ground_truth_path = "Test/Mask/Body"  # PathSelection("Select the Ground Truth Masks of the Test images folder").directory
    # For my laptop, I have SIGKILL because the support size is too big (>=70 128x128 images)
    batch_segmenter = BatchSegmenter(map_path, label_path, test_path, ground_truth_path, invert_label=True,
                                     batch_size=batch_size,
                                     support_size=32)
    batch_segmenter.save_results()

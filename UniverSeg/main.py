__author__ = "Arthur Astier"

from PyQt5.QtWidgets import QApplication
from PathSelection import PathSelection
import os
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

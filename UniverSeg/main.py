__author__ = "Arthur Astier"

from PyQt5.QtWidgets import QApplication
from skimage import exposure
from PathSelection import PathSelection
import os
import sys
import matplotlib.pyplot as plt
from BatchSegmenter import BatchSegmenter


def save_results(segmenter: BatchSegmenter):
    if not os.path.exists(os.path.join(os.getcwd(), 'Results')):
        os.mkdir(os.path.join(os.getcwd(), 'Results'))
    for batch in range(batch_size):
        segmented_images = segmenter.segmented_batches[batch]
        for idx, image in enumerate(segmented_images):
            filename = segmenter.test_filenames[idx]
            subject_directory = 'Results/' + filename
            if not os.path.exists(os.path.join(os.getcwd(), subject_directory)):
                os.mkdir(os.path.join(os.getcwd(), subject_directory))
            image = exposure.rescale_intensity(image, out_range=(0., 1.))
            plt.imsave(subject_directory + f'/Batch_n°{batch + 1}_' + filename, image)
            with open(os.path.join(subject_directory, "logs.txt"), 'a') as file:
                file.write(f"\nSupport Batch n°{batch + 1}:\n")
                for element in segmenter.support_batches[batch]:
                    file.write(str(element) + '\n')
                threshold = segmenter.thresholds[batch][idx]
                file.write(f'Threshold: {threshold} \n')
                file.close()


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
    save_results(batch_segmenter)

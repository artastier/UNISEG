__author__ = "Arthur Astier"

from PyQt5.QtWidgets import QApplication
from scipy.ndimage import zoom
from skimage import exposure
from PathSelection import PathSelection
import os
import sys
import matplotlib.pyplot as plt
from skimage.io import imsave
from BatchSegmenter import BatchSegmenter

if __name__ == "__main__":
    app = QApplication(sys.argv)
    batch_size = 1
    # TODO: Adapt PathSelection to choose multiple folders depending on the batch_size and unique_support (bool)
    map_path = ["Support/Body" for i in range(batch_size)]  # PathSelection("Select the Map folder").directory
    label_path = ["Support/Body_mask" for i in range(batch_size)]  # PathSelection("Select the Label folder").directory
    test_path = "Test/Original/Body"  # PathSelection("Select the Test folder").directory
    ground_truth_path = "Test/Mask/Body"  # PathSelection("Select the Ground Truth Masks of the Test images folder").directory
    # For my laptop, I have SIGKILL because the support size is too big (>=70 128x128 images)
    batch_segmenter = BatchSegmenter(map_path, label_path, test_path, ground_truth_path, invert_label=True,
                                     batch_size=batch_size,
                                     support_size=25)

    if not os.path.exists(os.path.join(os.getcwd(), 'Results')):
        os.mkdir(os.path.join(os.getcwd(), 'Results'))
    for batch in range(batch_size):
        batch_directory = 'Results/Batch' + str(batch + 1)
        if not os.path.exists(os.path.join(os.getcwd(), batch_directory)):
            os.mkdir(os.path.join(os.getcwd(), batch_directory))
        segmented_images = batch_segmenter.segmented_batches[batch]
        thresholds = batch_segmenter.thresholds[batch]
        with open(os.path.join(batch_directory, "support_batch_" + str(batch + 1) + "_.txt"), 'w') as file:
            for element in batch_segmenter.support_batches[batch]:
                file.write(str(element) + '\n')
            for idx, threshold in enumerate(thresholds):
                file.write(f'Query Image: {batch_segmenter.test_filenames[idx]} - Threshold: {threshold}')
        for idx, image in enumerate(segmented_images):
            filename = batch_segmenter.test_filenames[idx]
            rescale_factor = (512 / image.shape[0], 512 / image.shape[1], 1)
            image = zoom(image, rescale_factor)
            image = exposure.rescale_intensity(image, out_range=(0., 1.))
            plt.imsave(batch_directory + '/Segmented_' + filename, image)

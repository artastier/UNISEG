from PyQt5.QtWidgets import QApplication
from PathSelection import PathSelection
import os
import sys
import matplotlib.pyplot as plt
from BatchSegmenter import BatchSegmenter

if __name__ == "__main__":
    app = QApplication(sys.argv)
    batch_size = 2
    map_path = "Support/Original"  # PathSelection("Select the Map folder").directory
    label_path = "Support/Mask"  # PathSelection("Select the Label folder").directory
    test_path = "Test/Original"  # PathSelection("Select the Test folder").directory
    ground_truth_path = "Test/Mask"  # PathSelection("Select the Ground Truth Masks of the Test images folder").directory
    # For my laptop, I have SIGKILL because the support_size is too big (>22) and generates too much sub-images
    batch_segmenter = BatchSegmenter(map_path, label_path, test_path, ground_truth_path, invert_label=True,
                                     batch_size=batch_size,
                                     support_size=20)

    if not os.path.exists(os.path.join(os.getcwd(), 'Results')):
        os.mkdir(os.path.join(os.getcwd(), 'Results'))
    for batch in range(batch_size):
        batch_directory = 'Results/Batch' + str(batch+1)
        if not os.path.exists(os.path.join(os.getcwd(), batch_directory)):
            os.mkdir(os.path.join(os.getcwd(), batch_directory))
        segmented_images = batch_segmenter.segmented_batches[batch]
        for idx, image in enumerate(segmented_images):
            fig, ax = plt.subplots()
            plt.rcParams["figure.figsize"] = (5.12, 5.12)
            plt.rcParams["figure.dpi"] = 100
            ax.set_axis_off()
            plt.imshow(image, cmap='gray')
            filename = batch_segmenter.test_filenames[idx]
            plt.title(f" Batch n°{batch+1} - File: {filename}")
            fig.savefig(batch_directory+'/Segmented_' + filename, dpi=100)
            plt.close(fig)

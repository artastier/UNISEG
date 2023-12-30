from PyQt5.QtWidgets import QApplication
from PathSelection import PathSelection
import sys
import matplotlib.pyplot as plt
from Support import Support
from Segmenter import Segmenter
from Compare import compare

if __name__ == "__main__":
    app = QApplication(sys.argv)
    batch_size = 1
    map_path = PathSelection("Select the Map folder").directory
    label_path = PathSelection("Select the Label folder").directory
    # For my laptop, I have SIGKILL because the support_size is too big and generates too much sub-images
    support = Support(map_path, label_path, support_size=22, batch_size=batch_size, invert_label=True)
    print("Support size (number of sub-images): {}".format(support.maps.shape[1]))

    test_path = PathSelection("Select the Test folder").directory
    ground_truth_path = PathSelection("Select the Ground Truth Masks of the Test images folder").directory
    segmenter = Segmenter(test_path, support)
    enhanced_images = compare(ground_truth_path, segmenter=segmenter)
    for image, filename in zip(enhanced_images, segmenter.filenames):
        fid, ax = plt.subplots()
        plt.rcParams["figure.figsize"] = (512, 512)
        plt.rcParams["figure.dpi"] = 1
        ax.set_axis_off()
        plt.imshow(image, cmap='gray')
        plt.title(filename)
    plt.show()

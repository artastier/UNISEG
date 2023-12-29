import os

import matplotlib.pyplot as plt
from Support import Support
from Segmenter import Segmenter


if __name__ == "__main__":
    batch_size = 1
    support = Support("Support/Original", "Support/Mask", support_size=10, batch_size=batch_size)
    segmenter = Segmenter("Test/Original", support)
    segmented_images = segmenter.segmented_images
    for image in segmented_images:
        plt.figure()
        plt.imshow(image, cmap='gray')
    plt.show()

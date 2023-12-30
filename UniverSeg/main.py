import os

import matplotlib.pyplot as plt
from Support import Support
from Segmenter import Segmenter
from Compare import compare

if __name__ == "__main__":
    batch_size = 1
    support = Support("Support/Original", "Support/Mask", support_size=20, batch_size=batch_size, invert_label=True)
    print("Support size (number of sub-images): {}".format(support.maps.shape[1]))
    segmenter = Segmenter("Test/Original", support)
    enhanced_images = compare("Test/Mask", segmenter=segmenter)
    for image in enhanced_images:
        fid, ax = plt.subplots()
        ax.set_axis_off()
        plt.imshow(image, cmap='gray')
    plt.show()

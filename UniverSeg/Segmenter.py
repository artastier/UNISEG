__author__ = "Arthur Astier"

import os
import sys
import skimage.io as io
from skimage import exposure
from skimage.filters import threshold_otsu
import numpy as np
import torch
from DivideAndRebuild import divide, rebuild
from Support import Support
from universeg import universeg


class Segmenter:
    def __init__(self, test_folder_path: str, support: Support):
        self.model = universeg(pretrained=True)
        self.segmented_images, self.filenames = self.segment_from_path(test_folder_path, support)

    def segment_from_path(self, test_folder_path: str, support: Support):
        segmented_images = []
        filenames = os.listdir(test_folder_path)
        segmented_files = []
        if not filenames:
            print("\n No test images found \n", file=sys.stderr)
            return None
        for test_file in filenames:
            map_img = io.imread(os.path.join(test_folder_path, test_file), as_gray=True).astype(float)
            divided_map_img = divide(map_img)
            segmented_divided_img = self.apply_universeg(divided_map_img, support)
            if segmented_divided_img is None:
                continue
            segmented_files.append(test_file)
            segmented_img = exposure.rescale_intensity(rebuild(segmented_divided_img, np.shape(map_img)),
                                                       out_range=(0., 1.))
            segmented_images.append(segmented_img)
        return segmented_images, segmented_files

    def apply_universeg(self, divided_img, support: Support):
        if support.maps.shape[0] != divided_img.shape[0]:
            print("\n The support images and the query image aren't divided in the same number of 128x128 patches \n",
                  file=sys.stderr)
            return None
        # If we provide a torch tensor with the batch size (see UniverSeg documentation) equal to the number of 128x128
        # patches in the image, we are limited in the number of support, and we get unsatisfactory results. Hence, we
        # apply one model on 1 sub image to avoid SIGKILL with small support size.
        predictions = np.zeros(divided_img.shape)
        nb_subdivisions = divided_img.shape[0]
        for idx in range(nb_subdivisions):
            # We didn't use the sigmoid function as proposed in the Google Colab of UniverSeg because it was too
            # restrictive. Therefore, we optimized a manual threshold using the Dice score in Compare.py.
            predictions[idx] = self.model(divided_img[idx:idx + 1], support.maps[idx:idx + 1],
                                          support.labels[idx:idx + 1])[0].detach().numpy()
        return predictions

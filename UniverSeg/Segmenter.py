"""
@file
@brief This script defines a Segmenter class for segmenting images using the UniverSeg model.

@author Arthur Astier

@section dependencies
- os
- sys
- skimage.io
- skimage.exposure
- numpy
- DivideAndRebuild
- Support
- universeg

@section classes
- @b Segmenter: Class for segmenting images using the UniverSeg model.

"""

__author__ = "Arthur Astier"

import os
import sys
import skimage.io as io
from skimage import exposure
import numpy as np
from DivideAndRebuild import divide, rebuild
from Support import Support
from universeg import universeg
import torch


class Segmenter:
    """
    Class for segmenting images using the UNet model.
    """

    def __init__(self, test_folder_path: str, support: Support):
        """
        Initializes the Segmenter class.

        @param test_folder_path: Path to the test folder.
        @param support: Support object containing support data.
        """
        self.model = universeg(pretrained=True)
        self.segmented_images, self.filenames = self.segment_from_path(test_folder_path, support)

    def segment_from_path(self, test_folder_path: str, support: Support):
        """
        Segments images from a specified folder path.

        @param test_folder_path: Path to the test folder.
        @param support: Support object containing support data.
        @return: Segmented images and their filenames.
        """
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
        """
        Applies the UNet model for image segmentation.

        @param divided_img: Divided image patches.
        @param support: Support object containing support data.
        @return: Segmented image predictions.
        """
        if support.maps.shape[0] != divided_img.shape[0]:
            print("\n The support images and the query image aren't divided in the same number of 128x128 patches \n",
                  file=sys.stderr)
            return None
        predictions = np.zeros(divided_img.shape)
        nb_subdivisions = divided_img.shape[0]
        for idx in range(nb_subdivisions):
            predictions[idx] = torch.sigmoid(self.model(divided_img[idx:idx + 1], support.maps[idx:idx + 1],
                                                        support.labels[idx:idx + 1])[0]).round().clip(0,
                                                                                                      1).detach().numpy()
        return predictions

"""
@file
@brief This script defines a Support class for generating support data from image files.

@author Arthur Astier

@section dependencies
- os
- sys
- typing.List
- skimage.io
- torch
- DivideAndRebuild

@section classes
- @b Support: Class for generating support data from image files.

"""

__author__ = "Arthur Astier"

import os
import sys
from typing import List
import skimage.io as io
import torch
from DivideAndRebuild import divide


class Support:
    """
    Class for generating support data from image files.
    """

    def __init__(self, map_path: str, label_path: str, support_files: List[str], invert_label: bool = True):
        """
        Initializes the Support class.

        @param map_path: Path to the map folder.
        @param label_path: Path to the label folder.
        @param support_files: List of support files.
        @param invert_label: Boolean flag to invert label images (default is True).
        """
        self.maps, self.labels = None, None
        self.nb_division = None
        self.generate_support_from_path(map_path, label_path, support_files, invert_label)

    def generate_support_from_path(self, map_folder_path: str, label_folder_path: str, support_files: List[str],
                                   invert_label: bool = True):
        """
        Generates support data from image files.

        @param map_folder_path: Path to the map folder.
        @param label_folder_path: Path to the label folder.
        @param support_files: List of support files.
        @param invert_label: Boolean flag to invert label images (default is True).
        """
        support_size = len(support_files)
        for support_idx, map_file in enumerate(support_files):
            label_file = os.path.join(label_folder_path, map_file)
            if not os.path.isfile(label_file):
                print("\n We can't find the label image corresponding to the map image: " + map_file + "\n",
                      file=sys.stderr)
                continue
            map_img = io.imread(os.path.join(map_folder_path, map_file), as_gray=True).astype(float)
            label_img = io.imread(label_file, as_gray=True).astype(float)
            if map_img.shape != label_img.shape:
                print("\n The label and the map images for the image " + map_file + " don't have the same size \n",
                      file=sys.stderr)
                continue
            divided_label = divide(label_img, invert_label)
            divided_map = divide(map_img)
            if self.nb_division is None:
                self.nb_division = min(divided_label.shape[0], divided_map.shape[0])
                self.maps = torch.zeros((self.nb_division, support_size, 1, 128, 128))
                self.labels = torch.zeros((self.nb_division, support_size, 1, 128, 128))
            if (len(divided_label) != self.nb_division) or (len(divided_map) != self.nb_division):
                print("\n The label or the map images can't be divided into the same number of 128x128 patches than "
                      "the other support images provided.",
                      file=sys.stderr)
                continue
            self.maps[:, support_idx, :, :, :] = divided_map
            self.labels[:, support_idx, :, :, :] = divided_label
        if not bool(self.maps.count_nonzero()) and not bool(self.labels.count_nonzero()):
            print(
                "\n No support has been generated.\n Check if the name of the labels correspond to the name of the "
                "maps \n",
                file=sys.stderr)

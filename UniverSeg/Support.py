# @author Arthur Astier
import os
import sys
import numpy as np
from typing import List
import skimage.io as io
from skimage.util import invert
import torch
from DivideAndRebuild import divide


class Support:

    def __init__(self, map_path: str, label_path: str, support_files: List[str], invert_label: bool = True):
        self.maps, self.labels = Support.generate_support_from_path(map_path, label_path, support_files,
                                                                    invert_label)

    @staticmethod
    def generate_support_from_path(map_folder_path: str, label_folder_path: str, support_files: List[str],
                                   invert_label: bool = True):
        labels = []
        maps = []
        for map_file in support_files:
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
            divided_label_img, non_void_idx = divide(label_img)
            # In our support, the mask was black whereas the mask for UniverSeg is a white area
            if invert_label:
                divided_label_img = [invert(sub_im) for sub_im in divided_label_img]
            labels += divided_label_img
            maps += divide(map_img, non_void_idx)
        maps = np.array(maps)
        labels = np.array(labels)
        nb_sub_support = maps.shape[0] if maps.shape[0] == labels.shape[0] else None
        if nb_sub_support == 0 or nb_sub_support is None:
            print(
                "\n No support has been generated.\n Check if the name of the labels correspond to the name of the maps \n",
                file=sys.stderr)
            return None
        torch_maps = torch.zeros((1, nb_sub_support, 1, 128, 128))
        torch_labels = torch.zeros((1, nb_sub_support, 1, 128, 128))
        torch_maps[0, :, 0, :, :] = torch.from_numpy(maps)
        torch_labels[0, :, 0, :, :] = torch.from_numpy(labels)
        return torch_maps, torch_labels

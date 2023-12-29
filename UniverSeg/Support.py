import os
import sys
import numpy as np
from random import shuffle
import skimage.io as io
import torch
from DivideAndRebuild import divide


class Support:

    def __init__(self, map_path: str, label_path: str, support_size: int, batch_size: int):
        self.maps, self.labels = Support.generate_support_from_path(map_path, label_path, support_size, batch_size)

    @staticmethod
    def generate_support_from_path(map_folder_path: str, label_folder_path: str, support_size: int, batch_size: int):
        filenames = os.listdir(map_folder_path)
        shuffle(filenames)
        filenames = filenames[:support_size]
        labels = []
        maps = []
        for map_file in filenames:
            label_file = os.path.join(label_folder_path, map_file)
            if not os.path.isfile(label_file):
                print("\n We can't find the label image corresponding to the map image: " + map_file + "\n",
                      file=sys.stderr)
                continue
            map_img = io.imread(os.path.join(map_folder_path, map_file), as_gray=True).astype(float)
            divided_map_img, non_void_idx = divide(map_img)
            maps += divided_map_img
            label_img = io.imread(label_file, as_gray=True).astype(float)
            if map_img.shape != label_img.shape:
                print("\n The label and the map images don't have the same size \n", file=sys.stderr)
                continue
            labels += divide(label_img, non_void_idx)
        maps = np.array(maps)
        labels = np.array(labels)
        nb_sub_support = maps.shape[0] if maps.shape[0] == labels.shape[0] else None
        if nb_sub_support == 0:
            print(
                "\n No support has been generated.\n Check if the name of the labels correspond to the name of the maps \n",
                file=sys.stderr)
            return None
        torch_maps = torch.zeros((batch_size, nb_sub_support, 1, 128, 128))
        torch_labels = torch.zeros((batch_size, nb_sub_support, 1, 128, 128))
        torch_maps[0, :, 0, :, :] = torch.from_numpy(maps)
        torch_labels[0, :, 0, :, :] = torch.from_numpy(labels)
        return torch_maps, torch_labels

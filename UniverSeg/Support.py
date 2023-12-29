import os
import sys
import numpy as np
from random import shuffle
import skimage.io as io
from skimage import exposure
import torch


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
            divided_map_img, non_void_idx = Support.image_division(map_img)
            maps += divided_map_img
            label_img = io.imread(label_file, as_gray=True).astype(float)
            if map_img.shape != label_img.shape:
                print("\n The label and the map images don't have the same size \n", file=sys.stderr)
                continue
            labels += Support.image_division(label_img, non_void_idx)
        maps = np.array(maps)
        labels = np.array(labels)
        nb_sub_support = maps.shape[0] if maps.shape[0] == labels.shape[0] else None
        if nb_sub_support == 0:
            print(
                "\n No support has been generated. Check if the name of the labels correspond to the name of the maps \n",
                file=sys.stderr)
            return None
        torch_maps = torch.zeros((batch_size, nb_sub_support, 1, 128, 128))
        torch_labels = torch.zeros((batch_size, nb_sub_support, 1, 128, 128))
        torch_maps[0, :, 0, :, :] = torch.from_numpy(maps)
        torch_labels[0, :, 0, :, :] = torch.from_numpy(labels)
        return torch_maps, torch_labels

    @staticmethod
    def image_division(img, sector_indices=None):
        divided_img = []
        img = exposure.rescale_intensity(img, out_range=(0., 1.))
        img_shape = np.shape(img)
        if img_shape[0] % 128 != 0 or img_shape[1] % 128 != 0:
            print("\n The width or the height of the image isn't a multiple of 128 \n", file=sys.stderr)
        if sector_indices is None:
            non_void = []
            width = img_shape[1] // 128
            height = img_shape[0] // 128
            for i in range(height):
                for j in range(width):
                    extracted_image = img[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]
                    if not np.all(extracted_image == 1.0):
                        divided_img.append(extracted_image)
                        non_void.append((i, j))
            return divided_img, non_void
        else:
            for i, j in sector_indices:
                divided_img.append(img[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128])
            return divided_img

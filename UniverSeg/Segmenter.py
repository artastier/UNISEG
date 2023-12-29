import os
import sys
import skimage.io as io
import numpy as np
import torch
from DivideAndRebuild import divide, rebuild
from Support import Support
from universeg import universeg


class Segmenter:
    def __init__(self, test_folder_path: str, support: Support):
        self.segmented_images = Segmenter.segment_from_path(test_folder_path, support)

    @staticmethod
    def segment_from_path(test_folder_path: str, support: Support):
        segmented_images = []
        filenames = os.listdir(test_folder_path)
        if not filenames:
            print("\n No test images found \n", file=sys.stderr)
        for test_file in filenames:
            map_img = io.imread(os.path.join(test_folder_path, test_file), as_gray=True).astype(float)
            divided_map_img, non_void_idx = divide(map_img)
            segmented_divided_img = Segmenter.apply_universeg(divided_map_img, support)
            segmented_images.append(rebuild(segmented_divided_img, np.shape(map_img), non_void_idx))
        return segmented_images

    @staticmethod
    def apply_universeg(divided_img, support: Support):
        segmented_sub_img = []
        batch_size = support.maps.shape[0]
        model = universeg(pretrained=True)
        for sub_image in divided_img:
            target_image = torch.zeros((batch_size, 1, 128, 128))
            target_image[0, 0] = torch.from_numpy(sub_image)
            prediction = model(target_image, support.maps, support.labels)
            segmented_sub_img.append(prediction.detach().numpy())
        return segmented_sub_img

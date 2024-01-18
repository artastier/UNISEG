# @author Arthur Astier
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
        if not filenames:
            print("\n No test images found \n", file=sys.stderr)
            return None
        for test_file in filenames:
            map_img = io.imread(os.path.join(test_folder_path, test_file), as_gray=True).astype(float)
            divided_map_img, non_void_idx = divide(map_img)
            segmented_divided_img = self.apply_universeg(divided_map_img, support)
            segmented_img = exposure.rescale_intensity(rebuild(segmented_divided_img, np.shape(map_img), non_void_idx),
                                                       out_range=(0., 1.))
            # The segmented image is not a real binary mask
            # They explain in the publication that thresholding can decrease the performance of the Network
            segmented_images.append(segmented_img>0.90)
        return segmented_images, filenames

    def apply_universeg(self, divided_img, support: Support):
        segmented_sub_img = []
        batch_size = support.maps.shape[0]
        # We have tried to parallelize by duplicating the support and put each sub-image in a batch
        # But it led to SIGKILL because of too much memory requirements.
        for sub_image in divided_img:
            target_image = torch.zeros((batch_size, 1, 128, 128))
            target_image[0, 0] = torch.from_numpy(sub_image)
            prediction = self.model(target_image, support.maps, support.labels)
            segmented_sub_img.append(prediction.detach().numpy())
        return segmented_sub_img

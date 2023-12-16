import os

import matplotlib.pyplot as plt
import numpy as np
from universeg import universeg

import skimage.io as io
from skimage import exposure
from skimage.transform import resize
import torch

if __name__ == "__main__":
    # Set the dimensions
    batch_size = 1
    height = 128
    width = 128
    support_labels = torch.zeros((batch_size, 3, 1, height, width))
    support_map = torch.zeros((batch_size, 3, 1, height, width))
    idx_label = 0
    idx_map = 0
    for root, dir, filenames in os.walk("Support"):
        for filename in filenames:
            file = os.path.join(root, filename)
            # filter only image files with the following format
        if filenames and file.endswith('.png'):
            img = io.imread(file, as_gray=True).astype(float)
            img = resize(img, (height, width), anti_aliasing=True)
            img = exposure.rescale_intensity(img, out_range=(0., 1.))
            if "Mask" in root:
                support_labels[0][idx_label][0] = torch.from_numpy(img)
                idx_label += 1
            else:
                support_map[0][idx_map][0] = torch.from_numpy(img)
                idx_map += 1
    target_image = torch.zeros((batch_size, 1, height, width))
    img = io.imread("Test/Original/11011101021005.png", as_gray=True).astype(float)
    img = resize(img, (height, width), anti_aliasing=True)
    img = exposure.rescale_intensity(img, out_range=(0., 1.))
    target_image[0][0] = torch.from_numpy(img)

    model = universeg(pretrained=True)
    prediction = model(target_image, support_map, support_labels)
    plt.imshow(prediction[0][0].detach().numpy(), cmap='gray')
    plt.show()

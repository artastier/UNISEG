import os

import matplotlib.pyplot as plt
from universeg import universeg
from Support import Support
import skimage.io as io
from skimage import exposure
from skimage.transform import resize
import torch

if __name__ == "__main__":
    # Set the dimensions
    batch_size = 1
    support = Support("Support/Original", "Support/Mask", support_size=5, batch_size=batch_size)
    target_image = torch.zeros((batch_size, 1, 128, 128))
    img = io.imread("Test/Original/11011101021005.png", as_gray=True).astype(float)
    img = resize(img, (128, 128), anti_aliasing=True)
    img = exposure.rescale_intensity(img, out_range=(0., 1.))
    target_image[0][0] = torch.from_numpy(img)

    model = universeg(pretrained=True)
    prediction = model(target_image, support.maps, support.labels)
    plt.imshow(prediction[0][0].detach().numpy(), cmap='gray')
    plt.show()
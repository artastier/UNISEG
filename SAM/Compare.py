# @author Arthur Astier
import numpy as np
from skimage import exposure
from skimage import color

# test_image en gray
# gt_mask en gray en float
# predicted_mask en gray float

def color_enhanced(test_image, gt_mask, predicted_mask):
    enhanced_image = color.gray2rgb(test_image).copy()
    predicted_mask = 1*predicted_mask

    gt_mask_idx = np.where(gt_mask == 0)
    gt_mask_nb_pixels = gt_mask_idx[0].shape
    enhanced_image[gt_mask_idx] += np.repeat(np.array([[0, 0.5, 0]]), gt_mask_nb_pixels, axis=0)

    predicted_mask_idx = np.where(predicted_mask == 1)
    predicted_mask_nb_pixels = predicted_mask_idx[0].shape
    enhanced_image[predicted_mask_idx] += np.repeat(np.array([[0.5, 0, 0]]), predicted_mask_nb_pixels, axis=0)

    intersection_idx = np.where((gt_mask + predicted_mask) / 2 == 1)
    intersection_nb_pixels = intersection_idx[0].shape
    enhanced_image[intersection_idx] += np.repeat(np.array([[0.3, 0.3, 0]]), intersection_nb_pixels, axis=0)

    return exposure.rescale_intensity(enhanced_image,
                                      out_range=(0., 1.))

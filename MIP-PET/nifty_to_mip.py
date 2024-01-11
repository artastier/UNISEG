__author__ = "@paul-bd and Arthur Astier"

# https://github.com/paul-bd/MIP-PET.git
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
import nibabel
from skimage.transform import downscale_local_mean
import math


def nearest_power_of_2(number):
    return 2 ** math.ceil(math.log2(number))


def pad_images(images):
    shape = images.shape
    # We assume the input images are square
    next_power_of_2 = nearest_power_of_2(shape[0])
    pad_size = (next_power_of_2 - shape[0]) // 2
    padded_images = np.zeros((next_power_of_2, next_power_of_2, shape[2]))
    for i in range(shape[2]):
        # It will use zero-padding by default
        padded_images[:, :, i] = np.pad(images[:, :, i], (pad_size, pad_size))
    return padded_images


def downscale_images(images, downscale_size: tuple):
    shape = images.shape
    downsampling_factor = (shape[0] // downscale_size[0], shape[1] // downscale_size[1])
    downscaled_images = np.zeros((downscale_size[0], downscale_size[0], shape[2]))
    for i in range(shape[2]):
        downscaled_images[:, :, i] = downscale_local_mean(images[:, :, i], downsampling_factor)
    return downscaled_images


def transform_nifty(img, record_directory: str, patient_reference: str, nb_image=40,
                       is_mask=False,
                       borne_max=None,
                       pet_shapes=None):
    img_data = img.get_fdata()
    img_data += 1e-5
    # We assume the images are square, so shape[0] = shape[1]
    if not math.log2(img_data.shape[0]).is_integer():
        img_data = pad_images(img_data)
    if is_mask and img_data.shape[:2] != pet_shapes[patient_reference]:
        img_data = downscale_images(img_data, pet_shapes[patient_reference])
    if pet_shapes is None:
        img_size = (img_data.shape[0] / 100, img_data.shape[0] / 100)
    else:
        img_shape = pet_shapes[patient_reference]
        img_size = (img_shape[0] / 100, img_shape[1] / 100)
    create_mip_from_array(img_data, record_directory, patient_reference, img_size,
                          nb_image,
                          is_mask,
                          borne_max)


def create_mip_from_array(img_data, record_directory: str, patient_reference: str, img_size: tuple[float, float],
                          nb_image=40,
                          is_mask=False,
                          borne_max=None):
    ls_mip = []

    for angle in np.linspace(0, 360, nb_image):
        vol_angle = scipy.ndimage.interpolation.rotate(img_data, angle)

        MIP = np.amax(vol_angle, axis=1)
        MIP -= 1e-5
        MIP[MIP < 1e-5] = 0
        MIP = np.flipud(MIP.T)
        ls_mip.append(MIP)

    for i, mip in enumerate(ls_mip):
        fig, ax = plt.subplots(figsize=img_size, dpi=100)
        ax.set_axis_off()
        if borne_max is None:
            if is_mask:
                borne_max = 1
            else:
                borne_max = 15000
        ax.imshow(mip, cmap='Greys', vmax=borne_max)
        fig.savefig(record_directory + '/MIP_' + patient_reference + '%04d' % (i) + '.png', dpi=100)
        plt.close(fig)


def create_mip_from_path(pet_path: str, mask_path: str, record_folder: str, pet_borne_max=None,
                         mask_borne_max=None, nb_image=1):
    if not os.path.exists(os.path.join(os.getcwd(), record_folder)):
        os.mkdir(os.path.join(os.getcwd(), record_folder))
    pet_img_shapes = None
    if pet_path is not None:
        pet_img_shapes = generate_from_path(pet_path, record_folder=record_folder, mask=False, borne_max=pet_borne_max,
                                            nb_image=nb_image)
    if mask_path is not None:
        generate_from_path(mask_path, record_folder=record_folder, mask=True, borne_max=mask_borne_max,
                           nb_image=nb_image, pet_shapes=pet_img_shapes)
    print("MIP images generated !")


def generate_from_path(file_path: str, record_folder: str, mask=False, borne_max=None,
                       nb_image=1, pet_shapes=None):
    if not os.path.exists(file_path):
        print('The' + file_path + ' folder provided does not exist', file=sys.stderr)
    pet_files = [pet for pet in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, pet))]
    if not pet_files:
        print('The' + file_path + ' folder provided does not contain any file', file=sys.stderr)
    if mask:
        mip_directory = os.getcwd() + '/' + record_folder + '/Mask'
    else:
        mip_directory = os.getcwd() + '/' + record_folder + '/PET'
    if not os.path.exists(os.path.join(os.getcwd(), mip_directory)):
        os.mkdir(mip_directory + '/')

    if not mask:
        img_shapes = {}
    for pet in pet_files:
        file = os.path.join(file_path, pet)
        if pet.endswith('.nii'):
            img = nibabel.load(file)
            patient_name = pet.split(".")[0]
            img_data = img.get_fdata()
            img_data += 1e-5
            # We assume the images are square, so shape[0] = shape[1]
            if not math.log2(img_data.shape[0]).is_integer():
                img_data = pad_images(img_data)
            if mask and img_data.shape[:2] != pet_shapes[patient_name]:
                img_data = downscale_images(img_data, pet_shapes[patient_name])
            if pet_shapes is None:
                img_size = (img_data.shape[0] / 100, img_data.shape[0] / 100)
            else:
                img_shape = pet_shapes[patient_name]
                img_size = (img_shape[0] / 100, img_shape[1] / 100)
            create_mip_from_array(img_data, mip_directory, patient_name, img_size,
                                  nb_image,
                                  mask,
                                  borne_max)
            if not mask:
                img_shapes[patient_name] = (img_data.shape[0], img_data.shape[1])
    if not mask:
        return img_shapes

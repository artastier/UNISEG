__author__ = "@paul-bd and Arthur Astier"

# https://github.com/paul-bd/MIP-PET.git
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
import nibabel
from scipy.ndimage import zoom
import concurrent.futures


def rescale_images(images, rescale_size: tuple):
    shape = images.shape
    # With transpositions in create_mip_from_array, the first and the last dimensions of the original image represents
    # the wanted view for our problem
    rescale_factor = (rescale_size[0] / shape[0], rescale_size[1] / shape[2])
    unchanged_shape = shape[1]
    rescaled_images = np.zeros((rescale_size[0], unchanged_shape, rescale_size[1]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(zoom, images[:, i, :], rescale_factor) for i in range(unchanged_shape)]

    # Wait for all tasks to complete and retrieve the results
    results = concurrent.futures.as_completed(futures)
    for idx, zoomed_array in enumerate(results):
        rescaled_images[:, idx, :] = zoomed_array.result()
    return rescaled_images


def create_mip_from_array(img_data, record_directory: str, patient_reference: str,
                          nb_image=40,
                          is_mask=False,
                          borne_max=None):
    ls_mip = []
    img_data += 1e-5
    for angle in np.linspace(0, 360, nb_image):
        vol_angle = scipy.ndimage.interpolation.rotate(img_data, angle)

        MIP = np.amax(vol_angle, axis=1)
        MIP -= 1e-5
        MIP[MIP < 1e-5] = 0
        MIP = np.flipud(MIP.T)
        ls_mip.append(MIP)

    for i, mip in enumerate(ls_mip):
        if borne_max is None:
            if is_mask:
                borne_max = 1
            else:
                borne_max = 15000
        plt.imsave(record_directory + '/MIP_' + patient_reference + '%04d' % (i) + '.png', mip, cmap="Greys",
                   vmax=borne_max)


def create_mip_from_path(pet_path: str, mask_path: str, record_folder: str, pet_borne_max=None,
                         mask_borne_max=None, nb_image=1, output_size=None):
    if not os.path.exists(os.path.join(os.getcwd(), record_folder)):
        os.mkdir(os.path.join(os.getcwd(), record_folder))
    if pet_path is not None:
        generate_from_path(pet_path, record_folder=record_folder, mask=False, borne_max=pet_borne_max,
                           nb_image=nb_image, output_size=output_size)
    if mask_path is not None:
        generate_from_path(mask_path, record_folder=record_folder, mask=True, borne_max=mask_borne_max,
                           nb_image=nb_image, output_size=output_size)
    print("MIP images generated !")


def generate_from_path(file_path: str, record_folder: str, mask=False, borne_max=None,
                       nb_image=1, output_size=None):
    if output_size is None:
        output_size = [None]
    if not os.path.exists(file_path):
        print('The' + file_path + ' folder provided does not exist', file=sys.stderr)
        return
    pet_files = [pet for pet in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, pet))]
    if not pet_files:
        print('The' + file_path + ' folder provided does not contain any file', file=sys.stderr)
        return
    if mask:
        mip_directory = os.getcwd() + '/' + record_folder + '/Mask'
    else:
        mip_directory = os.getcwd() + '/' + record_folder + '/PET'
    if not os.path.exists(os.path.join(os.getcwd(), mip_directory)):
        os.mkdir(mip_directory + '/')
    # If only one output size is provided, it is applied to all images
    elif (len(output_size) != len(pet_files)) and len(output_size) != 1:
        print("You didn't provide the same number of output sizes and PET images.", file=sys.stderr)
        return
    for idx, pet in enumerate(pet_files):
        file = os.path.join(file_path, pet)
        if pet.endswith('.nii'):
            img = nibabel.load(file)
            patient_name = pet.split(".")[0]
            img_data = img.get_fdata()
            if len(output_size) != 1:
                rescale_size = output_size[idx]
            else:
                rescale_size = output_size[0]
            if (rescale_size is not None) and (img_data.shape != rescale_size):
                img_data = rescale_images(img_data, rescale_size)
            create_mip_from_array(img_data, mip_directory, patient_name, nb_image, mask, borne_max)

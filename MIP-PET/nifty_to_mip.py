# @author paul-bd and Arthur Astier
# https://github.com/paul-bd/MIP-PET.git
import os
import sys
import matplotlib.pylab as plt
import numpy as np
import scipy
from scipy import ndimage
import nibabel


def create_mip_from_3d(img, record_directory: str, patient_reference: str, nb_image=40, img_size=(5.12, 5.12),
                       is_mask=False,
                       borne_max=None):
    ls_mip = []

    img_data = img.get_fdata()
    img_data += 1e-5

    for angle in np.linspace(0, 360, nb_image):
        vol_angle = scipy.ndimage.interpolation.rotate(img_data, angle)

        MIP = np.amax(vol_angle, axis=1)
        MIP -= 1e-5
        MIP[MIP < 1e-5] = 0
        MIP = np.flipud(MIP.T)
        ls_mip.append(MIP)

    for mip, i in zip(ls_mip, range(len(ls_mip))):
        fig, ax = plt.subplots()
        plt.rcParams["figure.figsize"] = img_size
        plt.rcParams["figure.dpi"] = 100
        ax.set_axis_off()
        if borne_max is None:
            if is_mask:
                borne_max = 1
            else:
                borne_max = 15000
        plt.imshow(mip, cmap='Greys', vmax=borne_max)
        fig.savefig(record_directory + '/MIP_' + patient_reference + '%04d' % (i) + '.png', dpi=100)
        plt.close(fig)


def create_mip_from_path(pet_folder_name: str, mask_folder_name: str, pet_borne_max=None,
                         mask_borne_max=None,
                         img_size=(5.12, 5.12), nb_image=1):
    record_folder = 'MIP'
    if not os.path.exists(os.path.join(os.getcwd(), record_folder)):
        os.mkdir(os.path.join(os.getcwd(), record_folder))
    if pet_folder_name is not None:
        generate_from_path(pet_folder_name, record_folder=record_folder, mask=False, borne_max=pet_borne_max,
                           img_size=img_size,
                           nb_image=nb_image)
    if mask_folder_name is not None:
        generate_from_path(mask_folder_name, record_folder=record_folder, mask=True, borne_max=mask_borne_max,
                           img_size=img_size,
                           nb_image=nb_image)
    print("MIP images generated !")


def generate_from_path(file_folder_name: str, record_folder: str, mask=False, borne_max=None, img_size=(5.12, 5.12),
                       nb_image=1):
    if not os.path.exists(file_folder_name):
        print('The' + file_folder_name + ' folder provided does not exist', file=sys.stderr)

    pet_files = [pet for pet in os.listdir(file_folder_name) if os.path.isfile(os.path.join(file_folder_name, pet))]
    if not pet_files:
        print('The' + file_folder_name + ' folder provided does not contain any file', file=sys.stderr)
    if mask:
        mip_directory = os.getcwd() + '/' + record_folder + '/Mask'
    else:
        mip_directory = os.getcwd() + '/' + record_folder + '/PET'
    if not os.path.exists(os.path.join(os.getcwd(), mip_directory)):
        os.mkdir(mip_directory + '/')

    for pet in pet_files:
        file = os.path.join(file_folder_name, pet)
        if pet.endswith('.nii'):
            img = nibabel.load(file)
            patient_name = pet.split(".")[0]
            create_mip_from_3d(img, record_directory=mip_directory, patient_reference=patient_name,
                               nb_image=nb_image, img_size=img_size,
                               is_mask=mask,
                               borne_max=borne_max)

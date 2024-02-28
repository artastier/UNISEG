__author__ = "Arthur Astier"

from PyQt5.QtWidgets import QApplication
import nibabel
import sys
import os
import numpy as np
from PathSelection import PathSelection
from nifty_to_mip import create_mip_from_path
from Compare import compare
import matplotlib.pyplot as plt


def find_nearest_power_2(number):
    return 2 ** np.ceil(np.log2(number))


def extract_scan_size(scans_path: str):
    scan_sizes = dict()
    if not os.path.exists(scans_path):
        print('The' + scans_path + ' folder provided does not exist', file=sys.stderr)
        return
    pet_files = [pet for pet in os.listdir(scans_path) if os.path.isfile(os.path.join(scans_path, pet))]
    if not pet_files:
        print('The' + scans_path + ' folder provided does not contain any file', file=sys.stderr)
        return
    for idx, pet in enumerate(pet_files):
        file = os.path.join(scans_path, pet)
        if pet.endswith('.nii'):
            img = nibabel.load(file)
            patient_name = pet.split(".")[0]
            shape = find_nearest_power_2(img.shape)
            scan_sizes[patient_name] = (shape[:3].astype(int), img.affine)
    return scan_sizes


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pet_path = "patients/Test"  # PathSelection("(Optional) Select the image folder/file you want transform to MIP").directory
    mask_path = "patients/Test_mask"  # PathSelection("(Optional) Select the image folder/file you want transform to MIP").directory
    record_folder = "Test"
    scan_sizes = extract_scan_size(pet_path)
    create_mip_from_path(pet_path, mask_path, record_folder, pet_borne_max=7,
                         mask_borne_max=None, nb_image=1, output_size=scan_sizes)
    pet_path = "Test/PET"
    mask_path = "Test/Mask"
    enhanced_images = compare(mask_path, pet_path)
    for image in enhanced_images:
        plt.figure()
        plt.imshow(image)
    plt.show()

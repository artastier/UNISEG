__author__ = "Arthur Astier"

from PyQt5.QtWidgets import QApplication
import nibabel
import sys
import os
import numpy as  np
from PathSelection import PathSelection
from nifty_to_mip import create_mip_from_path


def find_nearest_power_2(number):
    return 2**np.ceil(np.log2(number))


def extract_scan_size(scans_path: str):
    scan_sizes = []
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
            img_data = img.get_fdata()
            shape = find_nearest_power_2(img_data.shape)
            scan_sizes.append((int(shape[0]), int(shape[2])))
    return scan_sizes


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pet_path = "patients/Full_body_scans"  # PathSelection("(Optional) Select the image folder/file you want transform to MIP").directory
    mask_path = "patients/Full_body_masks"  # PathSelection("(Optional) Select the image folder/file you want transform to MIP").directory
    record_folder = "Full_body"
    scan_sizes = extract_scan_size(pet_path)
    non_processed_files = create_mip_from_path(pet_path, mask_path, record_folder, pet_borne_max=7,
                                               mask_borne_max=None, nb_image=1, output_size=scan_sizes)

"""
@author Arthur Astier

This script provides functionality for processing medical image scans and generating Maximum Intensity Projection (MIP) images.

Requirements:
    - PyQt5.QtWidgets
    - nibabel
    - numpy
    - PathSelection
    - nifty_to_mip
    - Compare
    - matplotlib

Usage:
    Run this script to process medical image scans and generate MIP images. Specify the paths to the PET scans and masks,
    and the folder to save the MIP images. The script will extract the scan sizes, generate MIP images, and compare
    the enhanced images with masks.

"""
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
    """
    Find the nearest power of 2 to a given number.

    Args:
        number (int): The input number.

    Returns:
        int: The nearest power of 2 to the input number.
    """
    return 2 ** np.ceil(np.log2(number))


def extract_scan_size(scans_path: str):
    """
    Extract scan sizes from the provided scans folder.

    Args:
        scans_path (str): The path to the scans folder.

    Returns:
        dict: A dictionary mapping patient names to scan sizes.
    """
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
    # Initialize the PyQt application
    app = QApplication(sys.argv)

    # Specify paths and folders
    pet_path = PathSelection("(Optional) Select the image folder/file you want transform to MIP").directory
    mask_path = PathSelection("(Optional) Select the image folder/file you want transform to MIP").directory
    record_folder = "HECKTORUniverSeg"

    # Extract scan sizes
    scan_sizes = extract_scan_size(pet_path)

    # Generate MIP images
    create_mip_from_path(pet_path, mask_path, record_folder, pet_borne_max=7,
                         mask_borne_max=None, nb_image=1, output_size=scan_sizes)

    # Compare and plot enhanced images
    pet_path = "HECKTORUniverSeg/PET"
    mask_path = "HECKTORUniverSeg/Mask"
    enhanced_images = compare(mask_path, pet_path)
    for image in enhanced_images:
        plt.figure()
        plt.imshow(image)
    plt.show()

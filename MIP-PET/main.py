"""
@author Arthur Astier

This script provides functionality for generating Maximum Intensity Projection (MIP) images from PET scans.

Requirements:
    - PyQt5.QtWidgets
    - PathSelection
    - nifty_to_mip

Usage:
    Run this script and select the image folder/file(s) you want to transform into MIP using the PathSelection GUI.

"""
__author__ = "Arthur Astier"

from PyQt5.QtWidgets import QApplication
import sys
from PathSelection import PathSelection
from nifty_to_mip import create_mip_from_path

if __name__ == "__main__":
    # Initialize the PyQt application
    app = QApplication(sys.argv)

    # Prompt the user to select the PET and mask paths
    pet_path = PathSelection("(Optional) Select the image folder/file you want to transform to MIP").directory
    mask_path = PathSelection("(Optional) Select the image folder/file you want to transform to MIP").directory

    # Define the folder to save the MIP images
    record_folder = "Full_body"

    # Generate MIP images from the selected paths
    non_processed_files = create_mip_from_path(pet_path, mask_path, record_folder, pet_borne_max=7,
                                               mask_borne_max=None, nb_image=1, output_size=(300, 500))

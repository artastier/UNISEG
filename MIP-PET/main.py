__author__ = "Arthur Astier"

from PyQt5.QtWidgets import QApplication
import sys
from PathSelection import PathSelection
from nifty_to_mip import create_mip_from_path

if __name__ == "__main__":
    app = QApplication(sys.argv)
    pet_path = PathSelection("(Optional) Select the image folder/file you want transform to MIP").directory
    mask_path = PathSelection("(Optional) Select the image folder/file you want transform to MIP").directory
    record_folder = "Full_body"
    non_processed_files = create_mip_from_path(pet_path, mask_path, record_folder, pet_borne_max=7,
                                               mask_borne_max=None, nb_image=1, output_size=(300, 500))

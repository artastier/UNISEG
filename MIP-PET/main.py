# @author Arthur Astier
from PyQt5.QtWidgets import QApplication
import sys
from PathSelection import PathSelection
from nifty_to_mip import create_mip_from_path

if __name__ == "__main__":
    app = QApplication(sys.argv)
    pet_folder = PathSelection("(Optional) Select the image folder you want transform to MIP").directory
    mask_folder = PathSelection("(Optional) Select the image folder you want transform to MIP").directory
    record_folder = "MIP"
    create_mip_from_path(pet_folder, mask_folder, record_folder, pet_borne_max=10, mask_borne_max=None, nb_image=1,
                         img_size=(5.12, 5.12))

# @author Arthur Astier

from nifty_to_mip import create_mip_from_path

if __name__ == "__main__":
    patient_folder = "patients"
    pet_folder = "Original"
    mask_folder = "Mask"
    create_mip_from_path(patient_folder, pet_folder, None, pet_borne_max=10, mask_borne_max=None, nb_image=1,
                         img_size=(512, 512))

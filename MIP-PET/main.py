# @author Arthur Astier

import os

import nibabel

import nifty_to_mip

if __name__ == "__main__":
    for root, dir, filenames in os.walk("patients"):
        for filename in filenames:
            file = os.path.join(root, filename)
            # filter only image files with the following format
        if filenames and file.endswith('.nii'):
            img = nibabel.load(file)
            split_root = root.split("/")
            mip_directory = split_root[0] + "/" + split_root[1] + "/" + "MIP" + "/" + split_root[2]
            if "Mask" in split_root:
                mask = True
            else:
                mask = False
            nifty_to_mip.create_mip_from_3d(img, patient_reference=mip_directory, nb_image=1, duration=0.1,
                                            is_mask=mask,
                                            borne_max=None)

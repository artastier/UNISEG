# Maximum Intensity Projection from Nifti

This code aims to create MIP images from nifti images. It also enables to rescale the MIP images to a desired size.

# Usage

## Generate MIP image

```python
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
    record_folder = "MIP"

    # Generate MIP images from the selected paths
    create_mip_from_path(pet_path, mask_path, record_folder, pet_borne_max=7,
                         mask_borne_max=None, nb_image=1)
```

- The ```pet_path``` and ```mask_path``` are decoupled. If one of them is equal to ```None``` it will only process the
  supplied folder.
- ```pet_borne_max``` and ```mask_borne_max``` define the ```vmax``` parameter of the grayscale colormap used to record the
  MIP images.
- ```nb_image``` corresponds to the number images we want in output. Each image will be rotated of ```360/nb_image°```
  angle at each iteration.

## Rescale image

You need to provide a dictionary:

```python
output_size = {"patient1_filename":
                   ((256, 256, 512),  # Desired output 3D shape
                    affine1)  # Desired affine of the output 3D image
               }
create_mip_from_path(pet_path, mask_path, record_folder, pet_borne_max=7,
                     mask_borne_max=None, nb_image=1, output_size=output_size)
```

These two parameters are required to apply
the [```resample_from_to```](https://neuroimaging-data-science.org/content/005-nipy/003-transforms.html) method
contained in the ```nibabel.processing```
module. If the key is missing or wrong, the image won't be rescaled without generating errors.

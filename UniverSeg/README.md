# UniverSeg

We tested the UniverSeg model created by
Victor Ion Butoi,
Jose Javier Gonzalez Ortiz,
Tianyu Ma,
Mert R. Sabuncu,
John Guttag,
Adrian V. Dalca and detailed here: https://universeg.csail.mit.edu/.

This model has been trained on 2D 128x128 pixels images. The universal aspect of this model comes from its ability to
take into account a support.

A **support** is a set of images and their corresponding ground truth label of the tumor.

# Dataset

To test this model, we used the [HECKTOR dataset](https://hecktor.grand-challenge.org/) which was created to find a way to segment
head and neck tumors.

The images were PET and CT images that we transformed into 2D images thanks to the Maximum Intensity Projection (MIP)
(See the MIP-PET folder to transform these 3D images).

We made these 2D images divisible in several patches of 128x128 pixels.

# Architecture

The support images and the query image must be the same size. In fact, we divide the images in patches of 128x128
pixels. Next, the support corresponding to one 128x128 patch will be used to segment the same 128x128 patch on the query
image. For example, the support corresponding to the top left corner will be used to segment the top left
corner of the query image.

Once we made the prediction, which is a probability map, we optimize a manual threshold on this map based on the dice
score to create a binary mask. In
the [Google colab](https://colab.research.google.com/drive/1TiNAgCehFdyHMJsS90V9ygUw0rLXdW0r?usp=sharing) they use a
sigmoid to create a binary mask, but in our case it wasn't very efficient.

Finally, when all the patches have been segmented, we reconstruct the final mask and compare it to the ground truth.

The predicted mask is displayed in red, the ground truth in green and the intersection in orange.

# Setup

- Install ```universeg``` module:
    ```shell
    pip install git+https://github.com/JJGO/UniverSeg.git
    ```
- Clone this repository:
  ```shell
  git clone https://github.com/artastier/UNISEG.git
  ```

# Requirements

- All the images used must be divisible into 128x128 pixels images.
- The mask and the scan of the support must be the **same size** and have the **same filename**.
- All the scans and all the masks of the support must be the same size as the query image.

# Usage

```python
from BatchSegmenter import BatchSegmenter

batch_size = 2
map_path = ["Support1/Scans", "Support2/Scans"]
label_path = ["Support1/Masks", "Support2/Masks"]
test_path = "Test/Scans"
ground_truth_path = "Test/Masks"
batch_segmenter = BatchSegmenter(map_path, label_path, test_path, ground_truth_path,
                                 invert_label=True,
                                 batch_size=batch_size,
                                 support_size=32)
batch_segmenter.save_results()
```

- For each "batch", the ```BatchSegmenter``` will randomly pick ```support_size``` images in the label and map folders
  if the length of ```map_path``` and ```label_path``` is ```1```. It means these folders need to at least
  contain```support_size``` images.
- If you provide more than one folder in  ```map_path``` and ```label_path```, you need to supply as many folders as you
  have batches.
- The ```Test``` folder contains all the query images and their corresponding masks.
- ```invert_label``` must be set to ```True``` when the tumor corresponds to 0 in the binary mask. In fact, UniverSeg
  understands the area of interest in the binary mask of the support as being the area equal to 1.
- The results are stored in the ```Results``` folder. Inside, you can find as many folders as you supplied query images.

  The folders are named after the filename of the query images.
  For each query image you will find a ```logs.txt``` file which contains the thresholds and the filenames of the
  support used in each batch.

  In addition, you can see the predicted mask and the ground truth mask on the query image to
  evaluate the performance of each support batch.

# Improvements

Add the Dice score in the ```logs.txt``` file.
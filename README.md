# UNISEG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is the result of an academic project named UNISEG (Universal Segmentation) at Ecole Centrale in
Nantes, France, with my classmate [@damien-gautier-nantes](https://github.com/damien-gautier-nantes). The aim of this project was to evaluate the performance of
universal segmentation models for segmenting cancerous lesions. The UniverSeg and Segment Anything (SAM) models were
tested.

They were tested using the [HECKTOR dataset](https://hecktor.grand-challenge.org/) which regroups subjects with head and
neck tumors.

For confidentiality reasons we can't expose our results in images on GitHub.

# Usage :

## Download UniverSeg model

```shell
pip install git+https://github.com/JJGO/UniverSeg.git
```

## Download SAM model

- [Download a model checkpoint](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)

  WARNING: You may need to change where the program fetches the downloaded model for SAM use.
- Then:
    ```shell
    pip install git+https://github.com/facebookresearch/segment-anything.git
    pip install opencv-python pycocotools matplotlib onnxruntime onnx
    ```

## Clone this repository

```shell
git clone https://github.com/artastier/UNISEG.git
```

# Improvements:

To increase the automation of the segmentation of cancerous lesions, it may be useful to develop the following pipeline:

- Automatic detection with UniverSeg model
- Remove wrong predictions from UniverSeg
- Use of SAM to load an image where we can see what UniverSeg has segmented. Prompt points on the lesions
  non-segmented by UniverSeg and a background point. It can be interesting to try faster version of SAM.


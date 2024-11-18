# UNet_Cartridge_Segmentation
## 1. Background

This project is using Pytorch to implement a CNN-based model called UNET to do image segmentation.

The input are a pair of images including a cartridge image as well as a segmented image associate with it.

## 2. Requirements
- Python: 3.9.20
- numpy: 1.26.4
- opencv: 4.10.0.84
- omegaconf: 2.3.0
- Pytorch: 2.2.2
- dvc: 3.36.0
- For other package version, please check requirements.txt 

## 3. Data processing

- Step 1: Data Labelling

  Using OpenCV to extract the labelling colors from the segmented image and label diffent segments from 0 - 4 (5 segments in this project).


- Step 2: Data Augmentation

  Utilizing OpenCV to rotate, flip and shift input image to get more data for training and testing. 

  Step 1 and 2 are implemented in the same file and can be processed with command:
  ```
  python image_augment.py
  ```

- Step 3: Implementation of U-Net
  
  U-Net was first introduced by Olaf, Philipp and Thomasin 2015. The primary purpose of the model is to handle the medical images where the size of input data is quite limited.

  We implemented the U-Net based on [this paper](https://arxiv.org/abs/1505.04597) using Pytorch. For more details, check UNet.py

- Step 4: Model training

  More details on hyper-parameters for model training can be found in params.yaml. To run this step with command:

  ```
  python training.py
  ```
  
The whole process can also be controlled by Data Versioning Control (dvc). The setups is in dvc.yaml. To process the data using dvc:

```
dvc repro
```

To Check the dependencies:
```
dvc dag
```
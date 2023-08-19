![Alt text](./docs/figure_1.png?raw=true "Overall Framework")
![Alt text](./docs/figure_2.png?raw=true "NPVP")
# NPVP: A unified model for continuous conditional video prediction
https://openaccess.thecvf.com/content/CVPR2023W/Precognition/html/Ye_A_Unified_Model_for_Continuous_Conditional_Video_Prediction_CVPRW_2023_paper.html

# Continuous conditional video synthesis by neural processes
https://arxiv.org/abs/2210.05810v1

### Preparing Datasets
Processed KTH dataset: https://drive.google.com/file/d/1RbJyGrYdIp4ROy8r0M-lLAbAMxTRQ-sd/view?usp=sharing \
SM-MNIST: https://drive.google.com/file/d/1eSpXRojBjvE4WoIgeplUznFyRyI3X64w/view?usp=drive_link

For other datasets, please download them from the official website. Here we show the dataset folder structure.

#### BAIR
Please download the original BAIR dataset and utilize the "/utils/read_BAIR_tfrecords.py" script to convert it into frames as follows:

/BAIR \
  &nbsp;&nbsp;&nbsp;&nbsp; test/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; example_0/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0000.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0001.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; example_1/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0000.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0001.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; example_... \
&nbsp;&nbsp;&nbsp;&nbsp; train/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; example_0/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0000.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0001.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; example_... 

#### Cityscapes
Please download "leftImg8bit_sequence_trainvaltest.zip" from the official website. Center crop and resize all the frames to the size of 128X128. Save all the frames as follows:

/Cityscapes \
  &nbsp;&nbsp;&nbsp;&nbsp; test/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; berlin/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; berlin_000000_000000_leftImg8bit.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; berlin_000000_000001_leftImg8bit.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; bielefeld/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; bielefeld_000000_000302_leftImg8bit.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; bielefeld_000000_000302_leftImg8bit.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
&nbsp;&nbsp;&nbsp;&nbsp; train/\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; aachen/ \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; .... \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; bochum/ \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; .... \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
&nbsp;&nbsp;&nbsp;&nbsp; val/\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ....

#### KITTI
Please download the raw data (synced+rectified) from KITTI official website. Center crop and resize all the frames to the resolution of 128X128.
Save all the frames as follows:

/KITTI \
  &nbsp;&nbsp;&nbsp;&nbsp; 2011_09_26_drive_0001_sync/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0000000000.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0000000001.png \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp; 2011_09_26_drive_0002_sync/ \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... 

## Training
### Stage 1: CNN autoencoder training
Train the autoencoder and save the checkpoint. Configuration files for Stage 1 training are located in the "./configs" directory, with filenames ending in "*_Autoencoder.yaml". Before training, review the configuration file and adjust the dataset directory, checkpoint saving, TensorBoard log saving, etc., as needed.

Usage example:
```
python train_AutoEncoder_lightning.py --config_path ./configs/config_KTH_Autoencoder.yaml
```

### Stage 2: NP-based Predictor training

With a trained Autoencoder from stage 1, we can load it for the training of the NP-based Predictor in stage 2. Configuration files for Stage 2 training are located in the "./configs" directory, with filenames ending in "*_NPVP-D.yaml" or "_NPVP-S.yaml". Prior to training, review the configuration file and adjust the dataset directory, checkpoint saving, TensorBoard log saving, etc., according to your specific requirements.

Usage example:
```
python train_Predictor_lightning.py --config_path ./configs/config_KTH_Unified_NPVP-S.yaml
```

## Inference
Please read the inference.ipynb for the inference example of a KTH unified model.

Step 1: Download the process KTH dataset

Step 2: Download the Autoencoder checkpoint: https://drive.google.com/drive/folders/1eji1SxfT8do8TnWNPZqmhuOqxQZuaEpo?usp=sharing

Step 3: Download the Unified_NPVP-S checkpoint: https://drive.google.com/drive/folders/1knqw-KuWDSx6E-tG8jiOEG1G3BYMJJIf?usp=sharing

Step 4: Read and run "inference.ipynb".

### Citing
   
Please cite the paper if you find our work is helpful.
```
@inproceedings{ye2023unified,
  title={A Unified Model for Continuous Conditional Video Prediction},
  author={Ye, Xi and Bilodeau, Guillaume-Alexandre},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages={3603--3612},
  year={2023}
}
```

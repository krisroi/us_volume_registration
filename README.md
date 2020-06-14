# Ultrasound volume registration

A deep learning framework using PyTorch for unsupervised ultrasound to ultrasound volume registration.

This project is part of my master's thesis @NTNU during the spring of 2020.


# Network description

The network consists of two separate parts - an encoder and a fully connected regressor which regresses an affine transformation matrix.

The encoder used in the project is largely based on the encoder presented in

[Efficient 3D Fully Convolutional Networks for Pulmonary Lobe Segmentation in CT Images](https://arxiv.org/pdf/1909.07474v1.pdf) by H. Lee,
T.Matin, F. Gleeson and V. Grau.

Both training and prediction has the ability to run using the automatic [mixed precision (amp) strategy](https://arxiv.org/pdf/1710.03740.pdf) in addition to the regular
single precision (float32). This is done using Apex from NVIDIA, which can be found [here](https://github.com/nvidia/apex).


# Configuration file

In order to run the project, a configuration file needs to be placed inside the src/ folder. The structure of the configuratuion file should be as following:

    [Network]
    encoder_config = 4, 4, 4, 4, 4
    growth_rate = 8
    num_init_features = 8
    affine_config = 2048, 512, 256, 64
    num_init_parameters = 2048

    [Path]
    project_root = /home/krisroi/ # Root folder for project
    project_name = us_volume_registration # Root folder for git repo
    data_root = /mnt/EncryptedFastData/krisroi/ # Root folder for data
    procrustes = procrustes_analysis # Folder-name that runs the procrustes analysis.

    [Predict]
    batch_size = 16
    patch_size = 128
    stride = 50

The [Predict] group will be overwritten of the training-script to ensure that training and prediction is run on the same patch size. Changing stride and batch size would be optional.

# Command Line Usage

#### Description of Command Line options for training
| cl options            |   arg       |
| :-------------------- | :---------: |
| Usage help            |   -h        |
| Model name            |   -m        |
| Cardiac frame         |   -frame    |
| Learning rate         |   -lr       |
| Epochs                |   -e        |
| Batch size            |   -bs       |
| Patch size            |   -ps       |
| Stride                |   -st       |
| Number of datasets    |   -N        |
| Visible GPUs          |   -cvd      |
| Use regularization    |   -ur       |
| Memory efficient      |   -me       |
| Filter type           |   -ft       |
| Cross-validate        |   -cv       |
| Precision             |   -pr       |
| Drop rate             |   -dr       |
| Register hook         |   -rh       |

#### Description of Command Line options for prediction
| cl options            |   arg       |
| :-------------------- | :---------: |
| Usage help            |   -h        |
| Model name            |   -m        |
| Cardiac frame         |   -frame    |
| Prediction set number |   -PSN      |
| Visible GPUs          |   -cvd      |
| Plot patch-prediction |   -ppw      |
| Filter type           |   -ft       |
| Save predicted data   |   -sd       |
| Precision             |   -pr       |

#### Description of Command Line options for aligning volumes
| cl options            |   arg       |
| :-------------------- | :---------: |
| Usage help            |   -h        |
| Global theta filename |   -glob     |
| Cardiac frame         |   -frame    |
| Prediction set number |   -PSN      |
| Filter type           |   -aft      |
| Fixed volume opacity  |   -calph    |
| Moving volume opacity |   -galph    |


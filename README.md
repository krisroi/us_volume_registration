# Ultrasound volume registration

A deep learning framework for unsupervised ultrasound to ultrasound volume registration.

This project is part of my master's thesis @NTNU during the spring of 2020. 


# Network description

The network consists of two separate parts - an encoder and a fully connected regressor which regresses an affine transformation matrix. 

The encoder used in the project is largely based on the encoder presented in 

[Efficient 3D Fully Convolutional Networks for Pulmonary Lobe Segmentation in CT Images](https://arxiv.org/pdf/1909.07474v1.pdf) by H. Lee, 
T.Matin, F. Gleeson and V. Grau.

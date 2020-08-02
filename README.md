# DENAO: Monocular Depth Estimation Network With Auxiliary Optical Flow

This repo implements the multiview depth estimation network described in TPAMI 2020 paper:
[DENAO: Monocular Depth Estimation Network With Auxiliary Optical Flow](https://ieeexplore.ieee.org/document/9018142)

## Introduction
In this study, we demonstrate that learning a convolutional neural network (CNN) for depth estimation with an auxiliary optical flow network and the epipolar geometry constraint can greatly benefit the depth estimation task and in turn yield large improvements in both accuracy and speed. Our architecture is composed of two tightly-coupled encoder-decoder networks, i.e., an optical flow net and a depth net, the core part being a list of exchange blocks between the two nets and an epipolar feature layer in the optical flow net to improve predictions of both depth and optical flow. Our architecture allows to input arbitrary number of multiview images with a linearly growing time cost for optical flow and depth estimation.

## Requirements

The code was tested with Python 2.7, CUDA 9.0 and Ubuntu 16.04 using Caffe.

Add the custom layers into the corresponding folders of your own caffe.
Some layers like correlation layer could be found [here](https://github.com/lmb-freiburg/flownet2). Or you could just compile Caffe based on this repo.
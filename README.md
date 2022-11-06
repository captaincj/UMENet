# Towards Unified Multi-Excitation for Unsupervised Video Prediction


Code for our BMVC 2022 paper "Towards Unified Multi-Excitation for Unsupervised Video Prediction"

<img src="https://github.com/captaincj/UMENet/blob/master/images/umenet.png" width="500">

## Abstract
Unsupervised video prediction aims to forecast future frames conditioned on previ- ous frames with the absence of semantic labels. Most existing methods have applied conventional recurrent neural networks, which focus on past memory, while few draw at- tention to highlight motion and context information. In this work, we propose a Unified Multi-Excitation (UME) block to enhance long-short-term memory, specifically apply- ing an excitation mechanism to learn both channel-wise inter-dependencies and context correlations. Our contributions include: 1) introducing motion and channel excitation to enhance motion-sensitive channels of the features in the short term; and, 2) proposing an adaptive modeling scheme as context excitation inserted between (2+1)D convolution cells. The overall framework employs a multi-excitation block inserted into each Con- vLSTM layer to aggregate the motion, channel, and context excitations. The framework achieves state-of-the-art performance on a variety of spatio-temporal predictive datasets including the Moving MNIST, Sea Surface Temperature, Traffic BJ and Human 3.6 datasets. Extensive ablation studies demonstrate the effectiveness of each component of the method. 

## Code

In main.py, there is an example on how to run UMENet on the Moving MNIST dataset.

If you find this code useful for your research, please cite our paper.


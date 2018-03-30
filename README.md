# deep-res-net-analysis

Deep Learning Course Project: Deep Residual Network Property Analysis 

This project is to add a small CNN at the end of the modified Deep Residual Network (ResNet). 

We delete some blocks with self edge back to input of the layer in the ResNet and add a small CNN contatenated the last layer of ResNet.

This small CNN is then trained and we compared accuracy results to the ResNet.

cnn_cifar10_init.m file is to generate the CNN. And the other two are testing files.

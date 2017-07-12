# Convolutional Neural Network Cascade for Face-Detection

> These Codes reproduce the modal of CNN Cascade Architecture described in this [paper](http://users.eecs.northwestern.edu/~xsh835/assets/cvpr2015_cascnn.pdf)
> Implemented Using Theano Framework

*Total number of CNN = 6*
*Number of CNN for Binary Face Classification = 3*
*Number of CNN for Bounding Box Calibration = 3*

### Modal Flow

1. 12-Net CNN
2. 12-Calib CNN
3. 24-Net CNN
4. 24-Calib CNN
5. 48-Net CNN
6. 48-Calib CNN

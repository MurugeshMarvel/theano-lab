# EMOTION DETECTION

*This section contains programs with a CNN (Convolutional Neural Network) model for training the system on different human emotions, saving the trained model and testing with random images/ also from webcam.*

## Prerequistics

1. python 2.7 or more
2. theano 0.9.0
3. cv2
4. cPickle
5. numpy

## Steps Involved

1. Dataset Collection
2. Feature Extraction
3. Training from the Features
4. Testing the trained model
5. Say "Cheese"

### 1. Dataset Collection

* Face Cropped images
* All images should have same width and height
* Should collect in separate folders according to their Characteristics (i.e Happy, Sad, Anger,..etc)
* The name of the parent directory for images should be named as "dataset"

```

-dataset
  |
   --Happy
  |
   --Sad
  |
   --Anger
```
> The more accurate your training data is the more accurate your model will be

**You can obtain the dataset which I used for training my initial model in this [Link](www.google.com)

### 2. Feature Extraction

This is one of the Vital Process in training. As there are no system that directly takes images or text or sound for training or processing, those things should me represented in matrix, hot-vector or signal values which system can intepret. 

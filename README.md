# 2023-interns-cifar-cnn: CIFAR CNN with GradCAM Visualization

This repository contains a Convolutional Neural Network (CNN) model pipeline for a subset of the CIFAR-10 dataset, except only containing images of planes and ships. The pipeline consists of 3 Python scripts that perform data processing, model training and evaluation, visualization and inference. The scripts use PyTorch for model training and GradCAM visualization.

# Model Pipeline

train --> test --> inference

# Scripts

1. `gradcam.py`, `neuralnetwork.py`, `CIFAR2Class.py`: These modules define the classes and helper functions used in the model pipeline.
2. `train.py`: This module initializes the CNN model, defines the optimizer and loss function, sets the hyperparameters, loads & preprocesses the CIFAR dataset. Model training & evaluation on its performance on training and testing data is done here, accuracy values get stored.
3. `test.py`: This module evaluates the trained model on the test dataset, and generates heatmaps for a subset of images in the testing data, & also plots the training and testing accuracies across epochs, providing insights into the model's performance.
4. `inference.py`: This module allows a user to upload their own image of a plane or ship, and run our model on it. The actual & predicted label will be outputted, which provides insights on the model's performance on real world data.

## Hyperparameters

The following hyperparameters have been defined in the `model_load.py` file:

➡ Learning Rate = 0.0008

➡ Momentum = 0.4

➡ Batch Size = 50

➡ Number of Epochs = 50

These hyperparameters can be adjusted in the `train.py` file to optimize the model's performance.

## Performance

On average, this model achieves an overall test accuracy of about 85% on the CIFAR-2 dataset.

## Individual script

train.py: Arundeep
test.py: Siddharth
inference.py: Arundeep & Siddharth
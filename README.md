# Fashion MNIST

## Installation

Install Docker from: https://docs.docker.com/install/

Get dataset from: https://keras.io/datasets/#fashion-mnist-database-of-fashion-articles

Download Dockerfile from: https://hub.docker.com/r/gaarv/jupyter-keras/dockerfile

Use Docker pull command as: **docker pull gaarv/jupyter-keras**

## Models
TODO

## Training
### Preprocess the data
The data must be preprocessed before training the network. If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255. Scale these values to a range of 0 to 1 before feeding them to the neural network model. To do so, divide the values by 255. It's important that the training set and the testing set be preprocessed in the same way. If we don't preprocess data, when network grows, the parameters in network will overflow and losses and accuracies will be unpredictable.

### Cross validation
We randomly split the complete data into training and test sets, then perform the model training on the training set and use the test set for validation purpose, ideally split the data into 70:30 or 80:20, we used latter. With this approach there is a possibility of high bias if we have limited data, because we would miss some information about the data which we have not used for training. If our data is huge and our test sample and train sample has the same distribution then this approach is acceptable. To avoid overfitting, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test.

### Increasing model depth
The model needs to generalize the dataset to predict labels. However, when the model is small, there is not enough parameters to learn the dataset characteristics. In order to get rid of it, we can add more layers to model and by doing that we increase model depth.

## Results
TODO

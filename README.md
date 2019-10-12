# Fashion MNIST

## Installation

Install Docker from: https://docs.docker.com/install/

Get dataset from: https://keras.io/datasets/#fashion-mnist-database-of-fashion-articles

Download Dockerfile from: https://hub.docker.com/r/gaarv/jupyter-keras/dockerfile

Use Docker pull command as: **docker pull gaarv/jupyter-keras**

## Models
TODO
## Training
### Cross validation
We randomly split the complete data into training and test sets, then perform the model training on the training set and use the test set for validation purpose, ideally split the data into 70:30 or 80:20, we used latter. With this approach there is a possibility of high bias if we have limited data, because we would miss some information about the data which we have not used for training. If our data is huge and our test sample and train sample has the same distribution then this approach is acceptable. To avoid overfitting, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test.
### Increasing model parameters
TODO
## Results
TODO

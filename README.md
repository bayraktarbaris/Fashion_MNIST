# Fashion MNIST

## Installation

Install Docker from: https://docs.docker.com/install/

Get dataset from: https://keras.io/datasets/#fashion-mnist-database-of-fashion-articles

Download Dockerfile from: https://hub.docker.com/r/gaarv/jupyter-keras/dockerfile

Use Docker pull command as: ```docker pull gaarv/jupyter-keras```

## Models
***Model: "1_Layer"***

| Layer (type)  | Output Shape | Param # |
| ------------- | ------------ | ------- |
|Conv2D-1 (Conv2D)   |         (None, 26, 26, 32)    |    320 |      
|MaxPool (MaxPooling2D)  |     (None, 13, 13, 32)    |    0    |     
|Dropout (Dropout)       |     (None, 13, 13, 32)    |    0    |     
|flatten (Flatten)      |      (None, 5408)          |    0     |    
|Dense (Dense)          |      (None, 32)           |     173088  |  
|Output (Dense)         |      (None, 10)         |       330     |  


Total params: 173,738

Trainable params: 173,738

Non-trainable params: 0


***Model: "2_Layer"***

| Layer (type)  | Output Shape | Param # |
| ------------- | ------------ | ------- |
|Conv2D-1 (Conv2D)     |       (None, 26, 26, 32)  |      320       |
|MaxPool (MaxPooling2D)  |     (None, 13, 13, 32)  |      0         |
|Dropout-1 (Dropout)    |      (None, 13, 13, 32)  |      0         |
|Conv2D-2 (Conv2D)      |      (None, 11, 11, 64)  |      18496     |
|Dropout-2 (Dropout)     |     (None, 11, 11, 64) |       0         |
|flatten (Flatten)       |     (None, 7744)      |        0         |
|Dense (Dense)           |     (None, 64)       |         495680    |
|Output (Dense)          |     (None, 10)   |             650       |

Total params: 515,146

Trainable params: 515,146

Non-trainable params: 0


***Model: "3_layer"***

| Layer (type)  | Output Shape | Param # |
| ------------- | ------------ | ------- |
|Conv2D-1 (Conv2D)    |        (None, 26, 26, 32)    |    320       |
|MaxPool (MaxPooling2D) |      (None, 13, 13, 32)   |     0         |
|Dropout-1 (Dropout)    |      (None, 13, 13, 32)   |     0         |
|Conv2D-2 (Conv2D)      |      (None, 11, 11, 64)   |     18496     |
|Dropout-2 (Dropout)    |      (None, 11, 11, 64)   |     0         |
|Conv2D-3 (Conv2D)     |       (None, 9, 9, 128)    |     73856     |
|Dropout-3 (Dropout)   |       (None, 9, 9, 128)    |     0         |
|flatten (Flatten)     |       (None, 10368)        |     0         |
|Dense (Dense)        |        (None, 128)          |     1327232   |
|Dropout (Dropout)    |        (None, 128)          |     0         |
|Output (Dense)       |        (None, 10)           |     1290      |

Total params: 1,421,194

Trainable params: 1,421,194

Non-trainable params: 0

## Training
### Preprocessing data
The data must be preprocessed before training the network. If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255. Scale these values to a range of 0 to 1 before feeding them to the neural network model. To do so, divide the values by 255. It's important that the training set and the testing set be preprocessed in the same way. If we don't preprocess data, when network grows, the parameters in network will overflow and losses and accuracies will be unpredictable.

### Cross validation
We randomly split the complete data into training and test sets, then perform the model training on the training set and use the test set for validation purpose, ideally split the data into 70:30 or 80:20, we used latter. With this approach there is a possibility of high bias if we have limited data, because we would miss some information about the data which we have not used for training. If our data is huge and our test sample and train sample has the same distribution then this approach is acceptable. To avoid overfitting, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test.

### Increasing model depth
The model needs to generalize the dataset to predict labels. However, when the model is small, there is not enough parameters to learn the dataset characteristics. In order to get rid of it, we can add more layers to model and by doing that we increase model depth.
Increasing model depth, increases training time and learning since there are more parameters than small models.

## Learning rate
Large learning rates result in unstable training and tiny rates result in a failure to train.
Momentum can accelerate training and learning rate schedules can help to converge the optimization process.
Adaptive learning rates can accelerate training and alleviate some of the pressure of choosing a learning rate and learning rate schedule.

## Results
TODO

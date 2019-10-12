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

### Learning rate
Large learning rates result in unstable training and tiny rates result in a failure to train. Adaptive learning rates can accelerate training and alleviate some of the pressure of choosing a learning rate and learning rate schedule. If the learning rate is a big number then the optimum point can be missed, whereas if the learning rate is too small then it needs  more step to achieve the optimum point. However, if it misses the optimum point, then it is harder to get optimum point since it takes larger steps.

### Batch size
Using larger batches allows you to reduce the variance of your weight updates (by taking the average of the gradients in the mini-batch),and this in turn allows you to take biggerstep-sizes, which means the optimization algorithm will make progress faster. However, the amount of work done (in terms ofnumber of gradient computations) to reach a certain accuracy in the  objective will be the same. Low number of batch sizes should be chosen, since accuracy increases and loss drops regularly.

### Number of epochs
If you train so much on the training data,you may get over-fitted model since it memorizes the training set but not generalizes it. To deal with this problem, there is an approach called early stopping(not used in this project). You should stop training when the  error rate of validation data is minimum or when the error rate of training data is below some certain threshold. Consequently, if you increase the number of epochs, you will have an over-fitted model.

We set batch size as 512 and loss as cross-entropy loss. We used Adam optimizer with a learning rate 0.001 and trained the model for 50 epochs.

## Results
![Alt text](validation_accuracy_loss.png?raw=true "Title")

| Model  | Test Loss | Test Accuracy |
| ------------- | ------------ | ------- |
|1_Layer|0.2559|0.9137|
|2_Layer|0.2795|0.9205|
|3_Layer|0.2927|0.9197|


## Discussion
The increase in validation accuracy shows that our model does not overfit and continues learning since validation data is used to simulate test data. Also, 1_Layer model's accuracy is slightly lower than other two since it has lower learning capacity.

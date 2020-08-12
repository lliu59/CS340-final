import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

def create_model(input_shape, num_classes, learn_rate=0.1, fc_neurons=64, conv1_neurons=32, conv2_neurons=64):
    """ build function for sklearn KerasClassifier wrapper

    Parameters
    ----------
    input_shape: last three of dimensions of input (e.g. (28, 28, 1) for X, which is originally 50000*784)
    num_classes: number of output classes, i.e. output layer size
    learn_rate: SGD learning rate
    fc_neurons: number of neurons for the fully-connected layer
    conv1_neurons: number of neurons for the first convolutional layer
    conv2_neurons: number of neurons for the second convolutional layer

    Returns
    -------
    keras CNN model
    """
    model = Sequential()
    model.add(Conv2D(conv1_neurons, (3, 3),  # input layer
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(conv2_neurons, (3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(fc_neurons,
                    activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,
                    activation='softmax'))  # output layer

    # specify SGD
    optimizer = SGD(lr=learn_rate, momentum=0.9)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
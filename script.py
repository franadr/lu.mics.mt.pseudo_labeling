import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras import utils, losses, optimizers
from keras.datasets import fashion_mnist,mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 10
num_classes = 10
epochs = 300

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Used to add a dimensions to the sets from (60000,28,28) to (60000,28,28,1). Done because Tensor input needs 3
# dimensions structure (height, width, channels)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

default_model = Sequential()
default_model.add(Conv2D(20, kernel_size=(5, 5),
                         activation='relu',
                         input_shape=(28, 28, 1)))
default_model.add(MaxPooling2D(pool_size=(2, 2)))
default_model.add(Conv2D(40, (5, 5), activation='relu'))
default_model.add(MaxPooling2D(pool_size=(2, 2)))
default_model.add(Dropout(0.5))
default_model.add(Flatten())
default_model.add(Dense(650, activation='relu'))
default_model.add(Dropout(0.5))
default_model.add(Dense(150, activation='relu'))
default_model.add(Dense(10, activation='softmax'))

default_model.compile(loss=losses.categorical_crossentropy,
                      optimizer=optimizers.Adadelta(),
                      metrics=['accuracy'])

history = default_model.fit(x_train[:1000], y_train[:1000],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          )

print('\n History dict:', history.history)
print('\n Evaluate on test data')
score = default_model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc', score)

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os.path
from keras import utils, losses, optimizers, models
from keras.datasets import fashion_mnist, mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

batch_size = 10
num_classes = 10
epochs = 300
samples = 60000
labeled_samples = 1000
unlabeled_samples = samples - labeled_samples
# input image dimensions
img_rows, img_cols = 28, 28
model_checkpoint = ModelCheckpoint('MNIST{epoch:02d}.h5', period=100, save_weights_only=False)


def alpha(step):
    t1 = 100
    t2 = 600
    a = 3
    if step < 100:
        return 0.0
    elif step > 600:
        return a
    else:
        return ((step - t1) / (t2 - t1)) * a


def create_model():
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

    return default_model


def data_loading():
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
    x_train_labeled = x_train[:labeled_samples]
    x_train_unlabeled = x_train[labeled_samples:]
    return x_train_labeled, x_train_unlabeled, y_train, x_test, y_test


def train_supervised(model, x_train_labeled, y_train, batch_size):
    model.fit(x_train_labeled, y_train[:labeled_samples],
              batch_size=batch_size,
              epochs=100,
              verbose=1,
              callbacks=[model_checkpoint]
              )


x_train_labeled, x_train_unlabeled, y_train, x_test, y_test = data_loading()

if os.path.isfile('MNIST100.h5'):
    print('Fitted model found loading ...')
    model = models.load_model('MNIST100.h5')
else:
    print('Fitted model not found training ...')
    model = create_model()
    train_supervised(model, x_train_labeled, y_train, batch_size)

print('\n Evaluate on test data')
score = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc', )

import random
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import layers
import os.path
from keras import utils, losses, optimizers, models
from keras.datasets import fashion_mnist, mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, Callback

batch_size = 128
num_classes = 10
epochs = 300
samples = 60000
labeled_samples = 1000
unlabeled_samples = samples - labeled_samples
# input image dimensions
img_rows, img_cols = 28, 28
model_checkpoint = ModelCheckpoint('MNIST{epoch:02d}.h5', period=100, save_weights_only=False)


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

    return default_model


class PseudoLabeling(Callback):
    def __init__(self, model, labeled_samples, batch_size):
        self.x_train = np.zeros(shape=(60000, 28, 28, 1))
        self.y_train = np.zeros(shape=(60000, 10))
        self.unlabeled_indices = np.zeros(samples - labeled_samples)
        self.labeled_indices = np.zeros(labeled_samples)
        self.model = model
        self.t1 = 100
        self.t2 = 600
        self.num_classes = 1000
        self.n_samples = 60000
        self.x_train_l, self.y_train_l, self.x_train_u, self.y_train_u_real, self.y_train_u_pseudo, self.y_true_full, self.training_steps, self.testing_steps, self.y_test, self.x_test = self.data_preparation()
        self.alpha_s = 0.0
        self.unlabeled_proportion = self.x_train_u / self.n_samples
        self.unlabeled_proportion = self.x_train_l / self.n_samples

    def on_epoch_end(self, epoch, logs):
        self.alpha_s = self.alpha(epoch)
        self.y_train[self.unlabeled_indices] = self.model.predict(self.x_train[self.unlabeled_indices])
        random.shuffle(self.x_train)

    @staticmethod
    def alpha(step):
        t1 = 1
        t2 = 30
        a = 3
        if step < t1:
            return 0.0
        elif step > t2:
            return a
        else:
            return ((step - t1) / (t2 - t1)) * a

    def pseudo_loss(self, y_true, y_pred):
        y_true_item = y_true[:, :self.num_classes]
        unlabeled_flag = y_true[:, self.num_classes]
        crossentropy = losses.categorical_crossentropy(y_true=y_true_item, y_pred=y_pred)

        if unlabeled_flag == 1:
            return self.alpha_s * crossentropy
        else:
            return crossentropy

    def data_preparation(self):
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

        # getting indices
        indices = np.arange(x_train.shape[0])
        # shuffling
        np.random.shuffle(indices)

        # separating indices of labeled and unlabeled samples
        self.unlabeled_indices = indices[labeled_samples:]
        print(f"Unlabeled indices :{self.unlabeled_indices.shape[0]}")
        self.labeled_indices = indices[:labeled_samples]
        print(f"Labeled indices :{self.labeled_indices.shape[0]}")
        flag = np.zeros(x_train.shape[0])
        flag[self.unlabeled_indices] = 1

        # spliting the training sets
        # labeled features and target
        x_train_l = x_train[self.labeled_indices]
        y_train_l = y_train[self.labeled_indices]

        # unlabeled features , ground truth target
        x_train_u = x_train[self.unlabeled_indices]
        y_train_u_real = y_train[indices[labeled_samples:]]

        # init of pseudo labels array
        y_train_u_pseudo = np.zeros(y_train_u_real.shape[0])

        y_train[self.unlabeled_indices] = 0
        y_train_l = utils.to_categorical(y_train_l, num_classes)
        y_train_u_real = utils.to_categorical(y_train_u_real, num_classes)
        y_train = utils.to_categorical(y_train, 10)
        training_steps = x_train.shape[0] // batch_size
        testing_steps = x_test.shape[0] // batch_size
        self.x_train = x_train
        random.shuffle(self.x_train)
        self.y_train = y_train
        # # Flag array to distinguish unlabeled labels (1) from labeled labels (0)
        # flag_join = np.r_[np.repeat(0.0, x_train_l.shape[0]), np.repeat(1.0, x_train_u.shape[0])].reshape(-1, 1)
        # y_true_full = np.c_[utils.to_categorical(y_train, num_classes), flag_join]
        y_true_full = np.concatenate([y_train_l])
        return x_train_l, y_train_l, x_train_u, y_train_u_real, y_train_u_pseudo, y_train, training_steps, testing_steps, y_test, x_test

    def training_batch_gen(self):
        while True:
            nbatches = self.n_samples // batch_size
            for i in range(nbatches):
                start_batch = i * batch_size
                end_batch = start_batch + batch_size
                x = self.x_train[start_batch:end_batch]
                y = self.y_train[start_batch:end_batch]

                yield x, y

    # TODO validation generator
    def val_batch_gen(self):
        while True:
            indices = np.arange(self.y_test.shape[0])
            np.random.shuffle(indices)
            y_test_full = utils.to_categorical(self.y_test, 10)
            for i in range(len(indices) // batch_size):
                start_batch = i * batch_size
                end_batch = start_batch + batch_size
                x = self.x_test[start_batch:end_batch]
                y = y_test_full[start_batch:end_batch]

                yield x, y


def train_supervised(model, x_train_labeled, y_train_labeled, batch_size):
    model.fit(x_train_labeled, y_train_labeled,
              batch_size=batch_size,
              epochs=100,
              verbose=1,
              callbacks=[model_checkpoint]
              )


# if os.path.isfile('MNIST100.h5'):
#     print('Fitted model found loading ...')
#     model = models.load_model('MNIST100.h5')
# else:
# print('Fitted model not found training ...')
# model = create_model()
# train_supervised(model, x_train_labeled, y_train_labeled, batch_size)


# score = model.evaluate(pl.x_test, utils.to_categorical(pl.y_test, num_classes), batch_size=128)
# print('\n Evaluate on test data')
# print('test loss, test acc', score)
model = create_model()
pl = PseudoLabeling(model, labeled_samples, batch_size)

model.compile(loss=pl.pseudo_loss,
              optimizer=optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(pl.x_train[pl.labeled_indices], pl.y_train[pl.labeled_indices],
          batch_size=50,
          epochs=100,
          verbose=1,
          )
score = model.evaluate(pl.x_test, utils.to_categorical(pl.y_test, num_classes), batch_size=128)
print('\n Evaluate on test data')
print('test loss, test acc', score)

print('Pseudo labeling training :')
model.fit_generator(pl.training_batch_gen(), steps_per_epoch=pl.training_steps,
                    validation_data=pl.val_batch_gen(), callbacks=[pl],
                    validation_steps=pl.testing_steps, epochs=epochs).history

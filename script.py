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


class PseudoLabeling:
    def __init__(self, model, labeled_samples, batch_size):
        self.model = model
        self.t1 = 100
        self.t2 = 600
        self.num_classes = 10
        self.x_train_l, self.y_train_l, self.x_train_u, self.y_train_u_real, self.y_train_u_pseudo, self.y_true_full, self.training_steps, self.testing_steps = self.data_preparation()
        self.alpha_s = 0.0

    def on_epoch_end(self, epoch, logs):
        self.alpha_s = self.alpha(epoch)
        self.y_train_u_pseudo = self.model.predict(self.x_train_u)

    @staticmethod
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

    def pseudo_loss(self, y_true, y_pred):
        y_true_item = y_true[:, :self.num_classes]
        unlabeled_flag = y_true[:, self.num_classes]
        crossentropy = losses.categorical_crossentropy(y_true_item, y_pred)
        if unlabeled_flag == 1:
            return self.alpha_s * crossentropy
        else:
            return crossentropy

    @staticmethod
    def data_preparation():
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

        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train_l = x_train[indices[:labeled_samples]]
        y_train_l = y_train[indices[:labeled_samples]]
        x_train_u = x_train[indices[labeled_samples:]]
        y_train_u_real = y_train[indices[labeled_samples:]]
        y_train_u_pseudo = np.zeros(y_train_u_real.shape[0])
        y_train_l = utils.to_categorical(y_train_l, num_classes)
        y_train_u_real = utils.to_categorical(y_train_u_real, num_classes)

        training_steps = x_train.shape[0] // batch_size
        testing_steps = x_test.shape[0] // batch_size
        # Flag array to distinguish unlabeled labels (1) from labeled labels (0)
        flag_join = np.r_[np.repeat(0.0, x_train_l.shape[0]), np.repeat(1.0, x_train_u.shape[0])].reshape(-1, 1)
        y_true_full = np.c_[y_train, flag_join]
        return x_train_l, y_train_l, x_train_u, y_train_u_real, y_train_u_pseudo, y_true_full, training_steps, testing_steps

    # TODO training generator

    # TODO testing generator

def train_supervised(model, x_train_labeled, y_train_labeled, batch_size):
    model.fit(x_train_labeled, y_train_labeled,
              batch_size=batch_size,
              epochs=100,
              verbose=1,
              callbacks=[model_checkpoint]
              )


def train_semiupervised(model, x_train_labeled, x_train_unlabeled, y_train_labeled, y_train_unlabeled, batch_size):
    epochs = 300
    nbatches = len(x_train_unlabeled) / batch_size
    old_x_train_u = x_train_unlabeled
    print("Batches to be trained :", nbatches)
    for e in range(epochs):
        print("starting epoch")
        for b in range(nbatches):
            if b % 50 == 0:
                model.fit(x_train_labeled, y_train_labeled,
                          batch_size=batch_size,
                          epochs=1,
                          verbose=1)

            new_x_train_u = old_x_train_u[50:]
            current_batch = new_x_train_u[:50]
            old_x_train_u = new_x_train_u
            pseudo_labels = model.predict(current_batch)
            model.train_on_batch(current_batch, pseudo_labels, reset_metrics=True)
            u_loss = alpha(e) * losses.categorical_crossentropy(pseudo_labels, model.predict(current_batch))


if os.path.isfile('MNIST100.h5'):
    print('Fitted model found loading ...')
    model = models.load_model('MNIST100.h5')
else:
    print('Fitted model not found training ...')
    model = create_model()
    train_supervised(model, x_train_labeled, y_train_labeled, batch_size)

print('\n Evaluate on test data')
score = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc', score)
pl = PseudoLabeling(model, labeled_samples, batch_size)

# TODO run the training

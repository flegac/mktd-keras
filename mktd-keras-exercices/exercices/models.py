import os

import keras
from keras.layers import *
from keras.models import Sequential
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from exercices.dataset import Datasets


class Models:
    """
    A Model, in Machine Learning, is a function f: Tensor -> Tensor.
    The interesting fact about Models is that they can be trained to approximate another function.

    Most ML problems can be expressed as "How to approximate a function over Tensors".
    Ex:
        Original problem : "Tell me if a picture represents a cat or a dog"
        Encoded problem :
            - transform a picture into a Tensor of shape (width, height, 3) : the size 3 is for (R,G,B) color components
            - encode the known categories (cat, dog) as a Tensor of shape (2) : cat = [1,0] and dog = [0,1]
            - Give me the function F: (width,height,3) -> (2) that maps images to categories

    Given the reformulation above, the hard part of the problem is to write the F function.

    A Model purpose is exactly to solve this problem by approximating the F function.
    The training process consists in taking a Model (f function) and making it converge toward the F function.

    A Model usually has two function in its API :
        - fit(dataset) : modify the Model internal parameters to fit the target function F,
        - predict(tensor) : compute an approximation of the F function on some input Tensor.

    Documentation:
        https://keras.io/models/sequential/
    """

    @staticmethod
    def print_model_summary(model):
        model.summary()

    @staticmethod
    def load_model(path):
        # load json and create model
        json_file = open(os.path.join(path, 'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(os.path.join(path, 'model.h5'))
        return loaded_model

    @staticmethod
    def save_model(model, path):
        # serialize model to JSON
        model_json = model.to_json()
        with open(os.path.join(path, 'model.json'), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(os.path.join(path, 'model.h5'))
        print("Model saved to disk")

    @staticmethod
    def create_model(input_shape, num_classes):
        model = Sequential([
            Lambda(lambda x: x, input_shape=input_shape),
            #  TODO : use Convolutional Neural Network (Conv2D) to boost the training
            # https://keras.io/layers/convolutional/
            # Conv2D(filters=8, kernel_size=(3, 3), activation="relu"),

            Flatten(),
            # TODO : use batch normalization to allow the model to train on the dataset
            # https://keras.io/layers/normalization/
            # BatchNormalization(),

            Dense(units=8, activation='relu'),
            Dense(units=num_classes, activation='softmax')
        ])
        return model

    @staticmethod
    def train(model, dataset_provider):
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        (x_train, y_train), (_, _) = dataset_provider()

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        y_train = to_categorical(y_train)

        (x_train, y_train), (x_test, y_test) = Datasets.split_dataset(x_train, y_train, test_size=0.10)

        gen = ImageDataGenerator(rotation_range=8,
                                 width_shift_range=0.08,
                                 shear_range=0.3,
                                 height_shift_range=0.08,
                                 zoom_range=0.08)

        batches = gen.flow(x_train, y_train, batch_size=32)
        val_batches = gen.flow(x_test, y_test, batch_size=32)

        history = model.fit_generator(generator=batches,
                                      steps_per_epoch=int(batches.n / batches.batch_size),
                                      epochs=10,
                                      validation_data=val_batches,
                                      validation_steps=int(val_batches.n / val_batches.batch_size)
                                      )
        return history

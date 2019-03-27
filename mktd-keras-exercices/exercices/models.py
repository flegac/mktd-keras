import os
from typing import Tuple

from keras.layers import *
from keras.models import Sequential, load_model


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
    def load_model(path: str):
        filename = os.path.join(path, 'model.h5')
        loaded_model = load_model(filename)
        return loaded_model

    @staticmethod
    def save_model(model, path: str):
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, 'model.h5')
        model.save(filename)
        print("Model saved to disk : {}".format(filename))

    @staticmethod
    def create(input_shape: Tuple[int, int, int], num_classes: int):
        model = Sequential([
            Lambda(lambda x: x, input_shape=input_shape),
            #  TODO : use Convolutional Neural Network (Conv2D) to boost the training
            # https://keras.io/layers/convolutional/
            # Conv2D(filters=8, kernel_size=(3, 3), activation="relu"),

            Flatten(),
            # TODO : use batch normalization to allow the model to train on the dataset
            # https://keras.io/layers/normalization/
            BatchNormalization(),

            Dense(units=8, activation='relu'),
            Dense(units=num_classes, activation='softmax')
        ])
        return model

    @staticmethod
    def train(model, train, validation):
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        history = model.fit_generator(
            generator=train,
            steps_per_epoch=int(train.n / train.batch_size),
            epochs=2,
            validation_data=validation,
            validation_steps=int(validation.n / validation.batch_size)
        )
        return history

    @staticmethod
    def predict(model, x: np.ndarray):
        x = x.astype(np.float32)
        return model.predict(x.reshape(1, *x.shape))

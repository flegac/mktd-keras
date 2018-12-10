# 3.1 Metrics : evaluate model
# 3.2 Loss function (mean square error, cross entropy)
# 3.3 Optimizer function (stochastic gradient descent)
# 3.4 Batch size, epoch number
# 3.5 model.compile(training_parameters)
# 3.6 model.fit()
# 3.7 ImageDataGenerator : data
# 3.8 model.fit_generator()
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Reshape
from keras.utils import to_categorical

from exercices import utils


def create_sequential_model(input_shape=(28, 28, 1), output=10):
    model = Sequential([
        Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape),
        MaxPooling2D(),
        BatchNormalization(),

        Flatten(),
        BatchNormalization(),
        Dense(units=8, activation='relu'),
        Dense(units=output, activation='softmax')
    ])
    return model


def model_training():
    model = create_sequential_model()

    (x_train, y_train), (x_test, y_test) = utils.get_mnist()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1)
    # evaluate the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("{}: {}" .format(model.metrics_names[1], scores[1] * 100))


model_training()

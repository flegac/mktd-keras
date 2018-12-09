# 3.1 Metrics : evaluate model
# 3.2 Loss function (mean square error, cross entropy)
# 3.3 Optimizer function (stochastic gradient descent)
# 3.4 Batch size, epoch number
# 3.5 model.compile(training_parameters)
# 3.6 model.fit()
# 3.7 ImageDataGenerator : data
# 3.8 model.fit_generator()
from keras import Sequential
from keras.layers import Dense

from exercices import utils


def create_sequential_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    return model


def model_training():
    model = create_sequential_model()

    (x_train, y_train), (x_test, y_test) = utils.get_mnist()

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
    # evaluate the model
    scores = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))




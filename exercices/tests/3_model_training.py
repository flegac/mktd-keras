# 3.1 Metrics : evaluate model
# 3.2 Loss function (mean square error, cross entropy)
# 3.3 Optimizer function (stochastic gradient descent)
# 3.4 Batch size, epoch number
# 3.5 model.compile(training_parameters)
# 3.6 model.fit()
# 3.7 ImageDataGenerator : data
# 3.8 model.fit_generator()


def model_training():
    model = None
    X, Y = None, None

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
    # evaluate the model
    scores = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

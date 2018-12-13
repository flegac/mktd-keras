from keras.utils import to_categorical

from exercices.dataset import Datasets
from exercices.models import Models


def test_model_training():
    # TODO : Model is create here. Modify the Model to improve performances ! (failed tests will give you some advices)
    # https://keras.io/layers/about-keras-layers/
    model = Models.create_model(input_shape=(28, 28, 1), num_classes=10)
    dataset_provider = Datasets.mnist

    Models.train(model, dataset_provider)

    # evaluate the model
    (_, _), (x_test, y_test) = dataset_provider()
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_test = to_categorical(y_test)

    scores = model.evaluate(x_test, y_test, verbose=0)
    accuracy_percentage = scores[1] * 100
    print("evaluation on unseen dataset : {} = {}".format(model.metrics_names[1], accuracy_percentage))

    # TODO : add batch normalization to Models.create_model() and see real improvements
    # https://keras.io/layers/normalization/
    assert accuracy_percentage >= 70, "Bad accuracy ({}%) : {}".format(
        accuracy_percentage,
        "add a batch normalization layer to your model !")

    # TODO : add convolutional layers to to Models.create_model() and boost the efficiency of the model
    # https://keras.io/layers/convolutional/
    assert accuracy_percentage >= 90, "Bad accuracy : ({}%) : {}".format(
        accuracy_percentage,
        "use a Convolutional layer !")

    # TODO : try to get the best score, learn from state of the art networks !
    # https://keras.io/applications/
    assert accuracy_percentage >= 90, "Bad accuracy : ({}%) : {}".format(
        accuracy_percentage,
        "Loonk at states of the art networks : https://keras.io/applications/")


test_model_training()

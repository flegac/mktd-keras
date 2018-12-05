# - 2.1 Load existing model using h5 files
# - 2.2 Print summary (layers) of a model
# - 2.3 Create sequential model, add some layers
# - 2.4 Save model as h5 file


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json


def load_existing_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")


def print_model_summary():
    model = None
    model.summary()


def create_sequential_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))


def save_model_as_h5():
    model = None
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

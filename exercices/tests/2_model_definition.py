# - 2.1 Load existing model using h5 files
# - 2.2 Print summary (layers) of a model
# - 2.3 Create sequential model, add some layers
# - 2.4 Save model as h5 file
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json


def load_model(path):
    # load json and create model
    json_file = open(os.path.join(path, 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(path, 'model.h5'))
    return loaded_model


def create_sequential_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    return model


def save_model(model, path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(path, 'model.json'), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(path, 'model.h5'))
    print("Saved model to disk")


def print_model_summary(model):
    model.summary()


path = '.'
print(os.path.curdir)
model = create_sequential_model()
save_model(model, path)
print_model_summary(model)

model2 = load_model(path)
print_model_summary(model2)

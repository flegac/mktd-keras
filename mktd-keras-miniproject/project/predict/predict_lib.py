import json
import os

import numpy as np
from keras import Model
from keras.engine.network import Network

from exercices.models import Models


def init_model(path: str) -> Network:
    with open(os.path.join(path, 'model.json')) as _:
        cfg = json.load(_)
    model = Model.from_config(cfg)
    model.load_weights(os.path.join(path, 'model.h5'))
    return model


def predict(model: Network, image: np.ndarray):
    return Models.predict(model, image)

import os

import requests

from exercices.dataset import Tensors
from exercices.visualize import show_image
# from project.training.training_script import dataset_path, train_path, test_path


def test_prediction():
    print('test predictions ...')
    root_dir = '../resources/images'
    # root_dir = os.path.join(test_path, 'n8')
    for file in os.listdir(root_dir):
        path = os.path.join(root_dir, file)
        with open(path, 'rb') as f:
            r = requests.post('http://localhost:5000/process', files={'input': f})
            print('expected = {}, response = {}'.format(file[1], r.text))
            if str(file[1]) != r.text:
                show_image(Tensors.from_image(path))

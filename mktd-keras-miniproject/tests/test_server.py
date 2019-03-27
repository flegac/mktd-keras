import os

import requests


def test_prediction():
    root_dir = '../resources/images'
    for file in os.listdir(root_dir):
        path = os.path.join(root_dir, file)
        with open(path, 'rb') as f:
            r = requests.post('http://localhost:5000/process', files={'input': f})
            print('expected = {}, response = {}'.format(file[1], r.text))


test_prediction()

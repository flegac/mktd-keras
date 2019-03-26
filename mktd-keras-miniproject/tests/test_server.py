import requests


def test_prediction():
    with open('../resources/image.png', 'rb') as f:
        r = requests.post('http://localhost:5000/process', files={'input': f})
        print('response = {}'.format(r.text))

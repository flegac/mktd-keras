import requests

with open('../resources/image.jpg', 'rb') as f:
    r = requests.post('http://localhost:5000/process', files={'input': f})
    print(r.text)

import os
import scipy
from flask import Flask, request
from werkzeug.utils import secure_filename

import cv2
from exercices.models import Models

print('loading model ...')
model = Models.load_model('../resources/model')
print('model loaded !')

print("processing !")
filename = '../resources/images/n2140.jpg'

img = cv2.imread(filename)
img = img / 255.
img = scipy.resize(img, (128, 128, 3))
print(type(img))
print(img.shape)

print('predict with model ...')
result = Models.predict(model, img)

print(result)

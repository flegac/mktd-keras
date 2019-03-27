import cv2

import scipy

data  =scipy.resize(cv2.imread('/tmp/uploads/n2140.jpg'), (128, 128, 3))
print(data)
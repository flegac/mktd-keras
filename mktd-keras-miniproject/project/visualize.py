from typing import Tuple

from exercices.visualize import show_samples
import numpy as np


def random_sample(category: int) -> Tuple[np.ndarray, int]:
    return (np.random.rand(28, 28, 1), category)


#  select random images of each monkey category
'''
- iterate over train folder (n0,n1,n2,...)
- for each folder take N random images
'''

folders = list(range(10))
N = 10

samples = []
for category in folders:
    samples.extend([random_sample(category)] * N)

# plot the selected images
show_samples(samples, len(folders), N)

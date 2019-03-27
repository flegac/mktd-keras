from sklearn.utils import shuffle

from exercices.dataset import Datasets
from exercices.visualize import show_samples


def test_show_images():
    sample_provider = Datasets.mnist()

    samples = [_ for _ in sample_provider()]
    samples = shuffle(samples, n_samples=25)

    show_samples(samples, 5, 5)

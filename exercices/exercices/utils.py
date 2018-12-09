# Helpers for the exercices goes here


from keras.datasets import mnist


def get_mnist():
    return mnist.load_data()

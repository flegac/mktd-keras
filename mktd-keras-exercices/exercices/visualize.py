from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def show_image(image: np.ndarray):
    #  https://matplotlib.org/users/image_tutorial.html
    plt.imshow(image.squeeze())


def show_samples(samples, nrows=1, ncols=1):
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind, (x, y) in enumerate(samples):
        title = str(np.argmax(y)) if y.size > 1 else y
        axeslist.ravel()[ind].imshow(x.squeeze(), cmap=plt.jet())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    # plt.tight_layout()  # optional
    plt.show(block=True)


def show_hist(image: np.ndarray):
    plt.hist(image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')


def show_confusion_matrix(expected, predictions):
    #  TODO: compute confusion matrix
    #  https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    # cm = None
    cm = confusion_matrix(expected, predictions)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Show confusion matrix in a separate window
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return cm

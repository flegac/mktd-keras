# [Keras API](https://keras.io/)


# [Model](https://keras.io/models/about-keras-models/)

## Training
- compile(optimizer, loss, metrics)
- fit(x, y, batch_size, epochs)

## Prediction
- predict(x, batch_size) -> y

## Creation : stacking [Layers](https://keras.io/layers/about-keras-layers/)
- [Sequential](https://keras.io/models/sequential/): simple, linear workflow
- [Functional](https://keras.io/models/model/): more general workflow

## [Default models](https://keras.io/applications/)
- VGG
- Inception
- ResNet
- DenseNet

# Dataset

## [Image preprocessing](https://keras.io/preprocessing/image/)
- ImageDataGenerator(rescale, zoom_range, rotation_range, ...) : data augmentation
- ImageDataGenerator.flow(x,y, batch_size, shuffle, )

## [Default datasets](https://keras.io/datasets/)
- CIFAR10 : 50k (32,32) images on 10 categories
- MNIST: 60k (28,28) images of hand written digits


# Optimization

## [Loss function](https://keras.io/losses/)

## [Optimizer](https://keras.io/optimizers/)

## [Metrics](https://keras.io/metrics/)
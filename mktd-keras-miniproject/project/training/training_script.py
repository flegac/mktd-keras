import os

import keras
from keras import Model
from keras.callbacks import ModelCheckpoint, CSVLogger, LambdaCallback
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator

from exercices.models import Models

# define dataset
# TODO: modify the path to the dataset folder
dataset_path = 'path/to/dataset/10-monkey-species'  # https://www.kaggle.com/slothkong/10-monkey-species
dataset_path = 'D:\\Datasets\\10-monkey-species'
train_path = os.path.join(dataset_path, 'training')
test_path = os.path.join(dataset_path, 'validation')


def basic_model(input_shape, output_class_number):
    base_model = None

    # TODO: adapt to your specific problem a genrec pretrained keras model
    # ---> https://keras.io/applications/
    # base_model = keras.applications.xxx

    # transfert learning :
    # add new input layers to match your problem input
    inputs = Input(input_shape)
    x = base_model(include_top=False, input_shape=input_shape)(inputs)

    # add new output layers to match your problem output
    out = Flatten()(x)
    out = Dense(output_class_number, activation="softmax", name="output_full_3")(out)

    model = Model(inputs, out)

    model.summary()
    return model


def prepare_dataset(input_shape, batch_size=128):

    train_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    ).flow_from_directory(
        directory=train_path,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical')

    test_gen = ImageDataGenerator(
        rescale=1. / 255,
    ).flow_from_directory(
        directory=test_path,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical')
    return train_gen, test_gen


def train():
    input_shape = (96, 96, 3)

    model = basic_model(input_shape=input_shape, output_class_number=10)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-4),
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy'])

    # display model summary
    with open('model_summary.txt', 'w') as _:
        model.summary(print_fn=lambda x: _.write(str(x) + '\n'))

    train_gen, test_gen = prepare_dataset(model.input_shape[1:])

    # define training hyper parameters
    epochs = 10

    # train model with dataset
    history = model.fit_generator(train_gen,
                                  # steps_per_epoch=np.ceil(train_gen.n / train_gen.batch_size),
                                  epochs=epochs,
                                  validation_data=test_gen,
                                  # validation_steps=np.ceil(test_gen.n / test_gen.batch_size),
                                  verbose=1,
                                  callbacks=[
                                      ModelCheckpoint(
                                          filepath='model_epoch{epoch:02d}_loss{val_loss:.2f}.h5',
                                          monitor='val_loss',
                                          verbose=1,
                                          save_best_only=True),
                                      CSVLogger('training_logs.csv', append=False),
                                      LambdaCallback(on_train_end=lambda logs: Models.save_model(model, 'model_final'))
                                  ])


if __name__ == "__main__":
    train()

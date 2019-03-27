import os

import keras
from keras import Model
from keras.callbacks import ModelCheckpoint, CSVLogger, LambdaCallback
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator

# define model
from exercices.models import Models


def basic_model(input_shape, output_class_number):
    inputs = Input(input_shape)
    base_model = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, input_shape=input_shape)
    x = base_model(inputs)

    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(output_class_number, activation="sigmoid", name="output_full_3")(out)

    model = Model(inputs, out)

    model.summary()
    return model


input_shape = (128, 128, 3)
model = basic_model(input_shape=input_shape, output_class_number=10)
model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-4),
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy'])

# display model summary
with open('model_summary.txt', 'w') as _:
    model.summary(print_fn=lambda x: _.write(str(x) + '\n'))

# define training hyper parameters
epochs = 10
batch_size = 128

# define dataset
dataset_path = 'D:/Datasets/10-monkey-species'  # https://www.kaggle.com/slothkong/10-monkey-species
train_path = os.path.join(dataset_path, 'training')
test_path = os.path.join(dataset_path, 'validation')

augmentation = ImageDataGenerator(rescale=1. / 255)
train_gen = augmentation.flow_from_directory(directory=train_path,
                                             target_size=input_shape[:2],
                                             batch_size=batch_size,
                                             class_mode='categorical')
test_gen = augmentation.flow_from_directory(directory=test_path,
                                            target_size=input_shape[:2],
                                            batch_size=32,
                                            class_mode='categorical')

# train model with dataset
history = model.fit_generator(train_gen,
                              epochs=epochs,
                              validation_data=test_gen,
                              verbose=1,
                              callbacks=[
                                  ModelCheckpoint(
                                      filepath='model_epoch{epoch:02d}_loss{val_loss:.2f}.h5',
                                      monitor='val_loss',
                                      verbose=1,
                                      save_best_only=True),
                                  CSVLogger('training_logs.csv', append=False),
                                  LambdaCallback(on_train_end=lambda logs: Models.save_model(model, 'model_final.h5'))
                              ])

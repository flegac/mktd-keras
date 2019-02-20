import json
import os

import keras
from keras import Sequential
from keras.callbacks import ModelCheckpoint, CSVLogger, LambdaCallback
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator

# define dataset
dataset_path = 'path/to/10-monkey-species'  # https://www.kaggle.com/slothkong/10-monkey-species
train_path = os.path.join(dataset_path, 'training')
test_path = os.path.join(dataset_path, 'validation')


# define model
def basic_model(input_shape, output_class_number):
    kernel_size = (3, 3)
    filters = 16
    dropout = 0.25
    dense_size = 64

    model = Sequential([
        Conv2D(filters, kernel_size, activation='relu', input_shape=input_shape),
        Conv2D(filters, kernel_size, activation='relu'),
        MaxPooling2D(),
        Dropout(dropout),

        Conv2D(2 * filters, kernel_size, activation='relu'),
        Conv2D(2 * filters, kernel_size, activation='relu'),
        MaxPooling2D(),
        Dropout(dropout),

        Flatten(),
        Dense(dense_size, activation="relu"),
        Dropout(dropout),
        Dense(output_class_number, activation="softmax")
    ])

    model.summary()
    return model


input_shape = (128, 128, 3)
model = basic_model(input_shape=input_shape, output_class_number=10)
model.compile(optimizer=keras.optimizers.SGD(lr=1e-3), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

# display model summary
with open('model_summary.txt', 'w') as _:
    model.summary(print_fn=lambda x: _.write(str(x) + '\n'))

# export model architecture (without weights)
with open('model_config.json', 'w') as _:
    json.dump(model.get_config(), _, indent=2, sort_keys=True)

# define training hyper parameters
epochs = 30
batch_size = 32

# train model with dataset
augmentation = ImageDataGenerator()
train_gen = augmentation.flow_from_directory(directory=train_path,
                                             target_size=input_shape,
                                             batch_size=batch_size,
                                             class_mode='categorical')
test_gen = augmentation.flow_from_directory(directory=test_path,
                                            target_size=input_shape,
                                            batch_size=32,
                                            class_mode='categorical')
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
                                  LambdaCallback(on_train_end=lambda logs: model.save_weights('model_final.h5'))
                              ])

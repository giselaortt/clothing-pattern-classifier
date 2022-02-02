import numpy as np
#import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt


# Load training and validation sets
ds_train = image_dataset_from_directory(
    '../database/FingerCamera',
    labels='inferred',
    image_size=[224, 224], #decreasing the images to increase speed
    interpolation='nearest', #nearest requires smaller computational power
    batch_size=64,
    shuffle=True,
)

ds_valid = image_dataset_from_directory(
    '../database/FingerCamera',
    labels='inferred',
    image_size=[224, 224], #decreasing the images to increase speed
    interpolation='nearest', #nearest requires smaller computational power
    batch_size=64,
    shuffle=False,
)

pretrained_layers =VGG16()
pretrained_layers.trainable = False

model = keras.Sequential( pretrained_layers.layers )
#model.add(layers.Flatten())
'''
for i, layer in enumerate(model.layers):
    ...:     # layer.name = 'layer_' + str(i)    <-- old way
    ...:     layer._name = 'layer_' + str(i)
In [10]: model.summary()
'''
model.add( layers.Dense(6, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=30,
    verbose=0,
)
plt.figure()
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
plt.show()
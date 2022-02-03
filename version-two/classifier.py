import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.metrics as metrics


# Load training and validation sets
ds_train = image_dataset_from_directory(
    '../database/train_folder',
    labels='inferred',
    image_size=[224, 224], #decreasing the images to increase speed
    interpolation='nearest', #nearest requires smaller computational power
    batch_size=64,
    shuffle=True,
)

ds_valid = image_dataset_from_directory(
    '../database/test_folder',
    labels='inferred',
    image_size=[224, 224], #decreasing the images to increase speed
    interpolation='nearest', #nearest requires smaller computational power
    batch_size=64,
    shuffle=False,
)

pretrained_layers =VGG16()
pretrained_layers.trainable = False

model = keras.Sequential( pretrained_layers.layers )

for i, layer in enumerate(model.layers):
    layer._name = 'layer_' + str(i)

model.add(layers.Flatten())
model.add( layers.Dense(6, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[metrics.CategoricalAccuracy()],
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
plt.savefig('loss_evolution.png')
history_frame.loc[:, ['categorical_accuracy', 'val_categorical_accuracy']].plot()
plt.savefig('accuracy_evolution.png')


#plt.show()

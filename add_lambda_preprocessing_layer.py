from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from PIL import Image

# restore saved model
restored_model = keras.models.load_model('cats_dogs_classifier/01')

# add lambda preprocessing layer
def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

model = keras.Sequential([
    keras.layers.Lambda(preprocess, input_shape = (160, 160, 3)),
    restored_model
])
model.summary()

# export saved model
'''
model_version = '02'
model_name = 'cats_dogs_classifier'
model_path = os.path.join(model_name, model_version)
tf.saved_model.save(model, model_path)
'''

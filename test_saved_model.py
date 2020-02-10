from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from PIL import Image

# restore saved model
restored_model = keras.models.load_model('cats_dogs_classifier_2')
restored_model.build(input_shape = (None, 160, 160, 3))
restored_model.trainable = False
restored_model.summary()

# format function
def format_photo(image):
    IMG_SIZE = 160 # All images will be resized to 160x160
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image

# load test photos into numpy arrays
cat_img_path = 'test_data/cat.jpg'
cat_img_path_2 = 'test_data/cats_2.jpg'
false_cat_path = 'test_data/false_cat.jpg'
dog_img_path = 'test_data/dog.jpg'

cat_img = np.array(Image.open(cat_img_path))
cat_img_2 = np.array(Image.open(cat_img_path_2))
false_cat = np.array(Image.open(false_cat_path))
dog_img = np.array(Image.open(dog_img_path))

print(cat_img.shape)
print(cat_img_2.shape)
print(false_cat.shape)
print(dog_img.shape)

# format arrays
cat_img = format_photo(cat_img)
cat_img_2 = format_photo(cat_img_2)
false_cat = format_photo(false_cat)
dog_img = format_photo(dog_img)

print(cat_img.shape)
print(cat_img_2.shape)
print(false_cat.shape)
print(dog_img.shape)

input_batch = tf.stack([cat_img, cat_img_2, false_cat, dog_img])
print(input_batch.shape)

# make predictions
prediction = restored_model.predict(input_batch)
print(prediction) # positive = class 1, negative = class 0

'''
new_model = keras.Sequential([
    restored_model,
    keras.layers.Activation(activation = 'sigmoid')
])
prediction = new_model.predict(input_batch)
print(prediction)
prediction = tf.math.round(prediction)
print(prediction)
'''
























#

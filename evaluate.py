from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# download data
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split = [
        tfds.Split.TRAIN.subsplit(tfds.percent[:80]), # train
        tfds.Split.TRAIN.subsplit(tfds.percent[80:90]), # validation
        tfds.Split.TRAIN.subsplit(tfds.percent[90:]) # test
    ],
    with_info = True,
    as_supervised = True,
)

# format data
IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# restore saved model
classifier = keras.models.load_model('cats_dogs_classifier_2')
classifier.build(input_shape = (None, 160, 160, 3))
classifier.trainable = False
classifier.summary()

# evaluate performance
classifier.evaluate(test_batches)




































#

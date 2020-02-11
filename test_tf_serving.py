from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from PIL import Image
from tensorflow_serving.apis.predict_pb2 import PredictRequest

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

# make query
request = PredictRequest()
request.model_spec.name = 'cats_dogs'
request.model_spec.signature_name = 'serving_default'
input_name = 'mobilenetv2_1.00_160_input'
request.inputs[input_name].CopyFrom(tf.make_tensor_proto(input_batch))

# send query to model server
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

channel = grpc.insecure_channel('192.168.86.56:8500')
predict_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
response = predict_service.Predict(request, timeout = 10.0)

# output response
output_name = 'dense'
outputs_proto = response.outputs[output_name]
output = tf.make_ndarray(outputs_proto)
print(output.round(2))

























#

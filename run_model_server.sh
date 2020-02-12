#!/bin/bash
docker run -it --rm -p 8500:8500 -p 8501:8501 -v "cats_dogs_classifier:/models/cats_dogs" -e MODEL_NAME=cats_dogs tensorflow/serving

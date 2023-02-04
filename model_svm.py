import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import applications, layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures

# Feature extractor function
def get_features_and_labels(dataset, input_shape):
  base_model = applications.InceptionV3(weights="imagenet",
                                        include_top=False,
                                        input_shape=input_shape)
  all_features = []
  all_labels = []
  for images, labels in dataset:
    preprocessed_images = applications.inception_v3.preprocess_input(images)
    features = base_model.predict(preprocessed_images)
    all_features.append(features)
    all_labels.append(labels)
  # Return feature and label for dataset
  return np.concatenate(all_features), keras.utils.to_categorical(np.concatenate(all_labels))

"""
Defining SVM classifier:
Note: kernel_initializer parameter in RandomFourierFeature layer can be either a string identifier or a Keras Initializer instance.
Currently only 'gaussian' and 'laplacian' are supported string identifiers (case insensitive).
"""

def architecture(input_shape):
  inputs = keras.Input(shape=input_shape)
  x = layers.Flatten()(inputs)
  x =  RandomFourierFeatures(output_dim=4096, scale=10.0, kernel_initializer="gaussian")(x)
  outputs = layers.Dense(4, activation="linear")(x)
  model = keras.Model(inputs, outputs)
  
  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-2),
      loss=keras.losses.hinge,
      metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
  ) 
  return model
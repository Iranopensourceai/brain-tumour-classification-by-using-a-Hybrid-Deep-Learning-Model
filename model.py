import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionV3

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

 
def initialize_model(input_shape, n_classes, fine_tune=0):

  base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)

  if fine_tune > 0 :
    for layer in base_model.layers[:fine_tune]:
      layer.trainable = False
  else:
    for layer in base_model.layers:
      layer.trainable = True
  
  x = base_model.output 
  gp = layers.GlobalAveragePooling2D()(x)
  fc = layers.Dense(1072, activation='relu')
  d = layers.Dropout(0.2)(fc)
  prediction = layers.Dense(n_classes, activation='softmax')(d)

  return keras.Model(inputs= base_model.input, outputs=prediction)


def compile_model(model):
  model.compile(Adam(lr=0.001),
  loss='categorical_crossentropy', metrics=['val_accuracy'])
  return model
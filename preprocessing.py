import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_augmentation():
   train_aug=ImageDataGenerator(
        rotation_range=0.2,
        zoom_range=0.01,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 1.5])
   return train_aug

def test_augmentation():
   test_aug=ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True)
   return test_aug

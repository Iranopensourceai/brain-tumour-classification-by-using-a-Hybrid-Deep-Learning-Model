import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
IMG_HEIGHT = 224
IMG_WIDTH = 224

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    


def data_augmentation():
   model=keras.Sequential(
    [
      layers.RandomFlip("horizontal"),
      layers.RandomRotation(0.1),
      layers.RandomZoom(0.2),
    ])
   return model
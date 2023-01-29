'''
Creating a data frame and saving the path and label of each image in the data frame
'''
import os 
import pandas as pd
import numpy as np
from tqdm import tqdm 
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
BATCH_SIZE=32
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
path = '/kaggle/input/brain-tumor-classification-mri/'
class_label= ["no_tumor", "pituitary_tumor", "meningioma_tumor", "glioma_tumor"]

train_df,test_df=generate_datafram(path)
train_set,test_set = Read_flow_from_dataframe(train_df,test_df)
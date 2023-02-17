from model import *
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau

epochs = 50
batch_size = 32

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

model = compile_model(model)

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, 
        callbacks=[reduce_lr])

os.mkdir('MODEL_PATH')
model.save('MODEL_PATH/model.h5')
'''
 1.2 Use flow_from_dataframe for read data
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def Read_flow_from_dataframe(train_df,test_df):
    train_datagen= ImageDataGenerator()
    train_set= train_datagen.flow_from_dataframe(train_df,
                                               x_col='images',
                                               y_col='labels',
                                               target_size=(224, 224),
                                               classes=class_label,
                                               shuffle=True,
                                               batch_size=BATCH_SIZE,
                                               class_mode='categorical')

    # test_datagen= ImageDataGenerator(rescale = 1./255)
    test_datagen= ImageDataGenerator()

    test_set= test_datagen.flow_from_dataframe(test_df,
                                               x_col='images',
                                               y_col='labels',
                                               target_size=(224, 224),
                                               classes=class_label,
                                               shuffle=True,
                                               batch_size=BATCH_SIZE,
                                               class_mode='categorical')
    return train_set,test_set
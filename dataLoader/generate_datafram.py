import os
import pandas as pd
'''
 1.1 add image path and labels in datafram
'''
def generate_datafram(path):
    train_labels = []
    train_images = []

    test_labels = []
    test_images = []

    for sub_dir,dir,files in os.walk(path):
        for file in files:
            img_path = os.path.join(sub_dir,file)
            if img_path.split('/')[-3] == 'Training':
                train_images.append(img_path)
                train_labels.append(img_path.split('/')[-2])
            elif img_path.split('/')[-3] == 'Testing':
                test_images.append(img_path)
                test_labels.append(img_path.split('/')[-2])
    train_df = pd.DataFrame(list(zip(train_images,train_labels)),columns = ['images','labels'])
    test_df = pd.DataFrame(list(zip(test_images,test_labels)),columns = ['images','labels'])
    return train_df,test_df
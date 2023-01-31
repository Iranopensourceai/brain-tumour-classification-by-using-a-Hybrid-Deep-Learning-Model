'''
Creating a data frame and saving the path and label of each image in the data frame
'''
import pandas as pd
from generate_datafram import generate_datafram
from read_data import Read_flow_from_dataframe
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
BATCH_SIZE=32
path = '/kaggle/input/brain-tumor-classification-mri/'
df=generate_datafram(path)
labelencoder = LabelEncoder()
df=generate_datafram(path)
df['diagnosis'] = labelencoder.fit_transform(df['diagnosis'])
X_train,X_test,Y_train,Y_test=train_test_split(df['id_code'],df['diagnosis'],test_size=.2)
train_df = pd.DataFrame(list(zip(X_train,Y_train)),columns = ['id_code','diagnosis'])
test_df = pd.DataFrame(list(zip(X_test,Y_test)),columns = ['id_code','diagnosis'])
train_set,test_set = Read_flow_from_dataframe(train_df,test_df,BATCH_SIZE)
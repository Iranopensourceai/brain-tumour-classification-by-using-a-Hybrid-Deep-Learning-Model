'''
Creating a data frame and saving the path and label of each image in the data frame
'''
import pandas as pd
import torch
from generate_datafram import generate_datafram
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from read_data import Data_Loader
import torch.nn as nn
from Model import init_model
from torch import optim
from train import train
from test import test
n_epochs =30
BATCH_SIZE=32
path = '/kaggle/input/brain-tumor-classification-mri/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labelencoder = LabelEncoder()
df=generate_datafram(path)
df['diagnosis'] = labelencoder.fit_transform(df['diagnosis'])
X_train,X_test,Y_train,Y_test=train_test_split(df['id_code'],df['diagnosis'],test_size=.2)
train_df = pd.DataFrame(list(zip(X_train,Y_train)),columns = ['id_code','diagnosis'])
test_df = pd.DataFrame(list(zip(X_test,Y_test)),columns = ['id_code','diagnosis'])

# create datasets
train_loader,valid_loader = Data_Loader(train_df,test_df,BATCH_SIZE,path)

model = init_model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train(train_loader,valid_loader,model,n_epochs,device,optimizer,criterion)
test(valid_loader,model,device)
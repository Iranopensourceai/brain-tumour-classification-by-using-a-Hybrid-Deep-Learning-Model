'''
Creating a data frame and saving the path and label of each image in the data frame
'''
from generate_datafram import generate_datafram
from read_data import Read_flow_from_dataframe
BATCH_SIZE=32
path = '/kaggle/input/brain-tumor-classification-mri/'
train_df,test_df=generate_datafram(path)
train_set,test_set = Read_flow_from_dataframe(train_df,test_df,BATCH_SIZE)
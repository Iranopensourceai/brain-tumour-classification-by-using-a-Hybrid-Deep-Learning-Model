'''
 1.2 Use flow_from_dataframe for read data
'''
import torch
from torchvision import transforms
from TomurDataset import TomurTrainData ,TomurTestData

class_label= ["no_tumor", "pituitary_tumor", "meningioma_tumor", "glioma_tumor"]
# train transformations
train_trans = transforms.Compose([transforms.ToPILImage(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# validation transformations
valid_trans = transforms.Compose([transforms.ToPILImage(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
def Data_Loader(train_df,test_df,BATCH_SIZE,path):

    train_dataset = TomurTrainData(data      = train_df, 
                             directory = path,
                             transform = train_trans)
    valid_dataset = TomurTestData(data       = test_df, 
                                directory  = path,
                                transform  = valid_trans)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size  = BATCH_SIZE, 
                                            shuffle     = True, 
                                            num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                                            batch_size  = BATCH_SIZE, 
                                            shuffle     = True, 
                                            num_workers = 4)
    return train_loader,valid_loader
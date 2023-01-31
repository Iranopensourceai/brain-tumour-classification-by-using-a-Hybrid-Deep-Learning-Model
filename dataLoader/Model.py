import torchvision
from torchvision import models
import torch.nn as nn
def init_model():
                 
    # load pre-trained model
    model = models.googlenet(pretrained = True)
    in_feature = model.fc.in_features 
    for param in model.parameters():
        param.requires_grad = True
    model.fc =nn.Sequential(
    nn.Linear(in_features=in_feature, out_features=4, bias=True),
  )

            
    return model


# check architecture
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report
def test(valid_loader,model):
    y_pred = []
    y_true = []

    # iterate over test data
    for data in valid_loader:
            inputs, labels = data['image'].to(device), data['label'].to(device)# on G
            output = model(inputs) # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

    # constant for classes
    classes = ["glioma_tumor", "meningioma_tumor","no_tumor", "pituitary_tumor"]

    # Build confusion matrix
    print(classification_report(y_true, y_pred))
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/ np.sum(cf_matrix, axis=1) , index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')
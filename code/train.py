

import torch
import numpy as np
import time



def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()
def train(train_loader,valid_loader,model,n_epochs,device,optimizer,criterion):

    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_loader)
    since = time.time()
    for epoch in range(1, n_epochs+1):
        running_loss = 0.0
        # scheduler.step(epoch)
        correct = 0
        total=0
        print(f'Epoch {epoch}\n')
        for batch_idx, data in enumerate(train_loader):
            data_, target_ = data['image'].to(device), data['label'].to(device)# on GPU
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            _,pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==target_).item()
            total += target_.size(0)
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss/total_step)
        time_elapsed = time.time() - since
        print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')
        batch_loss = 0
        total_t=0
        correct_t=0
        with torch.no_grad():
            model.eval()
            for data in (valid_loader):
                data_t, target_t = data['image'].to(device), data['label'].to(device)# on GPU

                outputs_t = model(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _,pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t==target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t / total_t)
            val_loss.append(batch_loss/len(valid_loader))
            network_learned = batch_loss < valid_loss_min

            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')    
       
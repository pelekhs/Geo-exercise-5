import numpy as np
import torch
from sklearn import metrics
import os

def trainNN (model, patience, best_model_name, epochs, trainloader, valloader, device,
             optimizer, criterion):
    # Initialisations
    epoch = 0
    epoch_loss_val = []
    epoch_loss_train = []
    countdown = patience    

    # Training
    while epoch < epochs and countdown > 0:
        # Training
        epoch +=1
        batch_loss = []
        y_pred = []
        batch_acc = []
        # enumerate fetches a batch of the data for training!
        for i, data in enumerate(trainloader):
            inputs = data['bands'].to(device)
            labels = data['labels'].long().to(device)
            print(inputs)
            # reset gradients for each batch
            optimizer.zero_grad()
            # forward step
            out = model(inputs)
            # compute loss and save it to list
            loss = criterion(out, labels)
            batch_loss.append(loss.item())
            # backpropagate the loss
            loss.backward()
            # update weights according to the selected optimizer
            optimizer.step()
            # calculate batch accuracy for this epoch
            batch_pred = [np.argmax(prediction.
                        to(torch.device('cpu')).
                        detach().numpy()) 
                        for prediction in out]
            y_pred.append(batch_pred)
            batch_acc.append(metrics.accuracy_score(
                                    labels.
                                    to(torch.device('cpu')).
                                    detach().numpy(),
                                    batch_pred))
        print("Epoch: {}:\nTraining loss  : {:.5f}, Training accuracy  : {:.3f}".
            format(epoch, np.mean(batch_loss), np.mean(batch_acc)))
        epoch_loss_train.append(np.mean(batch_loss))
        # Validation
        batch_loss_val = []
        batch_acc_val = []
        y_pred_val = []
        
        # Validation
        # no need to store gradients here (Validation purposes only)
        with torch.no_grad():
            for i, data in enumerate(valloader):
                inputs_val = data['bands'].to(device) 
                labels_val = data['labels'].long().to(device)
                out_val = model(inputs_val)
                loss_val = criterion(out_val, labels_val)
                batch_loss_val.append(loss_val.item())
                
                # calculate batch accuracy for this epoch
                batch_pred = [np.argmax(prediction.
                            to(torch.device('cpu')).
                            detach().numpy()) 
                            for prediction in out_val]
                y_pred_val.append(batch_pred)
                batch_acc_val.append(metrics.accuracy_score(
                                        labels_val.
                                        to(torch.device('cpu')).
                                        detach().numpy(),
                                        batch_pred))
        epoch_loss_val.append(np.mean(batch_loss_val))
        print("Validation loss: {:1.3f}, Validation accuracy: {:1.3f}\nCountdown: {} \n".
              format(epoch_loss_val[-1], np.mean(batch_acc_val), countdown))
        # Early stopping condtion: N epochs without achieving loss less than the
        # present minimum. No need to save models before patience
        if epoch_loss_val[-1] <= min(epoch_loss_val):
            countdown = patience #start countdown
        #checkpoint 
            if epoch >= patience: # no need to save before that
        #I ovewrite models so as to keep the last to trigger the countdown
                torch.save(model, os.path.join(os.getcwd(),
                        "models" + os.path.sep + best_model_name + ".pt"))
        else:
            countdown -= 1
    print("Finished Training!")
    return epoch_loss_train, epoch_loss_val, epoch
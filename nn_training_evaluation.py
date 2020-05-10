import numpy as np
import torch
from sklearn import metrics
import os
import matplotlib.pyplot as plt



def logit_accuracy(logits, y_true):
    """This function computes the accuracy based on the logit outputs
    of the neural network (tensor) and the true y labels in integer form"""
    max_vals, y_pred = torch.max(logits, 1)
    acc = (y_pred == y_true).sum().item()/y_pred.size()[0]
    return acc, y_pred

def trainNN (model, best_model_name, epochs, trainloader, valloader,
             optimizer, criterion, patience=100, device="cpu", plot_upper_y=1.5):
    """This function trains a neural network (model) and then plot the loss
    diagrams on the training and validation sets through the epochs """ 
    # Initialisations
    epoch = 0
    epoch_loss_val = []
    epoch_acc_val = []
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
            inputs = data['bands'].float().to(device)
            labels = data['labels'].long().to(device)
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
            batch_acc.append(logit_accuracy(out, labels)[0])
        print("Epoch {}:\nTraining loss  : {:.4f}, Training accuracy  : {:.4f}".
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
                out_val = model(inputs_val.float())
                loss_val = criterion(out_val, labels_val)
                batch_loss_val.append(loss_val.item())
                # calculate batch accuracy for this epoch
                batch_acc_val.append(logit_accuracy(out_val, labels_val)[0])
        epoch_loss_val.append(np.mean(batch_loss_val))
        epoch_acc_val.append(np.mean(batch_acc_val))
        print("Validation loss: {:1.4f}, Validation accuracy: {:1.4f}\nCountdown: {} \n".
              format(epoch_loss_val[-1], epoch_acc_val[-1], countdown))
    
        # Early stopping condtion: N epochs without achieving loss less than the
        # present minimum. No need to save models before patience
        if epoch_loss_val[-1] <= min(epoch_loss_val):
            countdown = patience #start countdown
        #Checkpoint: I ovewrite models so as to keep the last to trigger the countdown
            torch.save(model, os.path.join(os.getcwd(),
                       "models" + os.path.sep + best_model_name + ".pt"))
        else:
            countdown -= 1
    print("Finished Training!")
    # Plot Train / Validation Loss

    plt.rcParams["figure.figsize"] = (10,4)
    plt.figure()
    plt.title("Relative Loss")
    plt.plot(list(range(1,epoch+1)), epoch_loss_train, label='Training set')
    plt.plot(list(range(1,epoch+1)), epoch_loss_val,  label='Validation set')
    plt.scatter(np.argmin(np.array(epoch_loss_val))+1, 
                min(epoch_loss_val), color="red", label="Minimum validation loss")
    plt.grid()
    plt.ylim(0, plot_upper_y)
    plt.legend(fancybox=True)
    plt.show()
    the_epoch = np.argmin(np.array(epoch_loss_val)) + 1
    print("The accuracy achieved is: {:.3f} at epoch {}".
        format(epoch_acc_val[the_epoch - 1], the_epoch))
    return epoch_loss_train, epoch_loss_val, epoch

from sklearn.metrics import confusion_matrix, classification_report

def NN_evaluation(model, testloader, criterion, patience=100, device="cpu"):
    """This function evaluates the models on a test set"""
    y_pred_test = []
    y_test = []
    batch_loss = []
    batch_accs = []
    num_classes=16
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs = data['bands'].float().to(device) 
            labels = data['labels'].long().to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            batch_loss.append(loss.item())
            batch_acc, batch_pred = logit_accuracy(logits, labels)
            y_pred_test.append(batch_pred)
            y_test.append(data['labels'])
            batch_accs.append(batch_acc)
    #    print("Validation loss: {:1.3f}, Validation Acc: {:1.3f} \n".
    #          format(np.mean(batch_loss), np.mean(batch_accs)))
    # Predicted labels to numpy array
    y_pred_test = np.concatenate([y_pred_test[i].to("cpu").numpy()
                                 for i in range(len(y_pred_test))]).reshape(-1)
    y_test = np.concatenate([y_test[i].to("cpu").numpy() for i in range(len(y_test))]).reshape(-1)
    # =============================================================================
    # Confusion Matrix testidation Set
    # =============================================================================
    from confusion_matrix import plot_confusion_matrix
    # testidation
    labels = list(set(y_test))
    print(classification_report(y_test, y_pred_test, digits=3))
    cm = confusion_matrix(y_test, y_pred_test, 
                               labels=list(range(num_classes)))
    print("\n")
    plt.rcParams["figure.figsize"] = (10,6)
    plt.figure()
    plot_confusion_matrix(cm, classes=labels, title='Confusion matrix - Validation set', cmap=plt.cm.Greens)
    return y_test, y_pred_test, cm
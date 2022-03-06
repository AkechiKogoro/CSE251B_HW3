from basic_fcn import *
from dataloader import *
from utils import *
import torch.optim as optim
import time
from torch.utils.data import DataLoader
import torch
import gc
import copy
import yaml
import torch
from depict import *
import os

def train(Init, cnn_model, optimizer, train_loader, val_loader):

    device, fname, tname, criterion, epochs, early_stop = Init['processor'], Init['fname'], \
        Init['tname'], Init['loss'], Init['epochs'], Init['early_stop_epoch'];

    best_iou_score = 0.0
    pre_iou_score = 0.0
    num_increase = 0
    
    iou_list=[];
    pixel_list=[];
    loss_list=[];

    if (Init['loss'] == 'Diceloss'):
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-6e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    for epoch in range(epochs):
        ts = time.time(); print('\n');
        for iter, (inputs, labels) in enumerate(train_loader):
            # reset optimizer gradients
            optimizer.zero_grad();

            # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device) #transfer the input to the same device as the model's
            labels = labels.to(device) #transfer the labels to the same device as the model's

            outputs = cnn_model(inputs) #we will not need to transfer the output, it will be automatically in the same device as the model's!
            
            
            targets=labels.long();
            loss = criterion(outputs, targets)#calculate loss
            
            # backpropagate
            loss.backward();
            # update the weights
            optimizer.step();

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        
        if (Init['loss'] == 'Diceloss'):
            scheduler.step()

        current_miou_score, current_pixel_acc, current_loss = val(Init, epoch, cnn_model, val_loader)
        
        if (early_stop > 0):
            if (current_miou_score < pre_iou_score):
                num_increase += 1;
            else:
                num_increase = 0;
            
            pre_iou_score = current_miou_score
            
            if (num_increase > early_stop):
                print('\n'*3 + 'EARLY STOP !');
                break;

        iou_list.append(current_miou_score)
        pixel_list.append(current_pixel_acc)
        loss_list.append(current_loss)

        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            file_name = './model/mlp' + fname + '.pth';
            torch.save(cnn_model.state_dict(), file_name);

    print(f"Best IoU in training is : {best_iou_score}\n")

    info_name = './info/info' + fname +'.txt';
    info_file = open(info_name, 'w');
    print(f"Best IoU in training is : {best_iou_score}\n", file = info_file)
    info_file.close();
            
    acc_pic_name = './img/acc' + fname + '.png';
    loss_pic_name = './img/loss' + fname + '.png';

    BasicPlot(loss_list, loss_pic_name, 'epochs', 'loss', 'loss vs epochs' + tname)
    MultiplePlot([iou_list, pixel_list], acc_pic_name, 'epochs', 'accuracy', \
        'accuracy vs epochs' + tname, ['IoU', 'pixel accuracy'], ['red', 'green'])
    print('\n');


def val(Init, epoch, cnn_model, val_loader):

    n_class, device, criterion = Init['n_class'], Init['processor'], Init['loss']
    cnn_model.eval() # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(val_loader):

            # both inputs and labels have to reside in the same device as the model's
            input = input.to(device) #transfer the input to the same device as the model's
            label = label.to(device) #transfer the labels to the same device as the model's

            output = cnn_model(input)
            target = label.long();

            loss = criterion(output, target) #calculate the loss
            losses.append(loss.item()) #call .item() to get the value from a tensor. The tensor can reside in gpu but item() will still work 
            
            pred = torch.argmax(output, axis = 1) # Make sure to include an argmax to get the prediction from the outputs of your model

            mean_iou_scores.append(np.nanmean(iou(pred, label, n_class)))  # Complete this function in the util, notice the use of np.nanmean() here
        
            accuracy.append(pixel_acc(pred, label)) # Complete this function in the util


    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    cnn_model.train() #DONT FORGET TO TURN THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores), np.mean(accuracy), np.mean(losses)

def test(Init, cnn_model, test_loader):
    #TODO: load the best model and complete the rest of the function for testing
    fname = Init['fname'];
    info_name = './info/info' + fname +'.txt';
    info_file = open(info_name, 'a');

    mean_iou_scores, accuracy, losses = \
        val(Init, epoch = 'Test', cnn_model = cnn_model, val_loader = test_loader)

    print(f"Loss at Test is {losses}", file = info_file)
    print(f"IoU at Test is {mean_iou_scores}", file = info_file)
    print(f"Pixel at Test is {accuracy}", file = info_file)

    info_file.close();
    pass




def main(file_name = 'config.yaml'):

    config = load_config("./", file_name)
    if not os.path.exists('./img/'):
        os.mkdir('./img/');
    if not os.path.exists('./info/'):
        os.mkdir('./info/');
    if not os.path.exists('./model/'):
        os.mkdir('./model/');

    Init=Initialization(config);

    cnn_model, optimizer, train_loader, val_loader, test_loader = Init()

    val(Init, -1 , cnn_model, val_loader)  # show the accuracy before training
    train(Init, cnn_model, optimizer, train_loader, val_loader)


    fname = Init['fname']

    model_name = './model/mlp' + fname + '.pth';

    cnn_model.load_state_dict(torch.load(model_name))
    cnn_model.eval();

    test(Init, cnn_model, test_loader)
    
    # housekeeping
    gc.collect() 
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main();
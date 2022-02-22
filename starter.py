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

def load_config(path, file_name = 'config.yaml'):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open(path + file_name, 'r'), Loader=yaml.SafeLoader)


def Init(config):
    # TODO: Some missing values are represented by '__'. You need to fill these up.

    batch_size = config['batch_size'];
    n_class = config['n_class'];
    learning_rate = config['learning_rate']
    momentum = config['lambda']

    if (config['loss'] == 'CrossEntropy'):
        criterion = nn.CrossEntropyLoss();
        # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
    

    if (config['processor'] == 'cuda'):    
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); 
        # determine which device to use (gpu or cpu)
    else:
        device = torch.device("cpu");


    train_dataset = TASDataset('tas500v1.1') 
    val_dataset = TASDataset('tas500v1.1', eval=True, mode='val')
    test_dataset = TASDataset('tas500v1.1', eval=True, mode='test')


    train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size= batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle=False)

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight.data)
            torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases   

    
    if (config['model'] == 'baseline'):
        cnn_model = FCN(n_class=n_class)
        cnn_model.apply(init_weights)
        
    optimizer = optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=momentum)
    # choose an optimizer

    cnn_model = cnn_model.to(device) #transfer the model to the device

    return cnn_model, device, optimizer, criterion, train_loader, val_loader, test_loader


def train(config, cnn_model, optimizer, train_loader, val_loader):

    epochs = config['epochs']
    if (config['loss'] == 'CrossEntropy'):
        criterion = nn.CrossEntropyLoss();
        

    if (config['processor'] == 'cuda'):    
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); 
        # determine which device to use (gpu or cpu)
    else:
        device = torch.device("cpu");

    fname = '_' + config['model'] + '_' + config['transform'] + '_' + config['loss'] \
        + '_' + 'lr' + str(config['learning_rate'])
    
    best_iou_score = 0.0
    
    for epoch in range(epochs):
        ts = time.time()
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
        
        
        current_miou_score = val(config, epoch, cnn_model, val_loader)
        
        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            file_name = './model/mlp' + fname + '.pth';
            torch.save(cnn_model.state_dict(), file_name);
            
    

def val(config, epoch, cnn_model, val_loader):

    n_class = config['n_class']
    if (config['processor'] == 'cuda'):    
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); 
        # determine which device to use (gpu or cpu)
    else:
        device = torch.device("cpu");

    if (config['loss'] == 'CrossEntropy'):
        criterion = nn.CrossEntropyLoss();
        # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
    

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

    return np.mean(mean_iou_scores)

def test():
    #TODO: load the best model and complete the rest of the function for testing
    pass




def main(file_name = 'config.yaml'):
    config = load_config("./", file_name)
    cnn_model, device, optimizer, criterion, train_loader, val_loader, test_loader = Init(config)
    
    val(config, 0 , cnn_model, val_loader)  # show the accuracy before training
    train(config, cnn_model, optimizer, train_loader, val_loader)
    test()
    
    # housekeeping
    gc.collect() 
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main();
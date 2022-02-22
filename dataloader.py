from torch.utils.data import Dataset
from torchvision import transforms

import yaml
import random
import os
import numpy as np
import glob
import PIL
from tqdm import tqdm 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from basic_fcn import *
from utils import *

def load_config(path, file_name = 'config.yaml'):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open(path + file_name, 'r'), Loader=yaml.SafeLoader)

def get_config_info(config):
    batch_size = config['batch_size'];
    n_class = config['n_class'];
    learning_rate = config['learning_rate']
    momentum = config['lambda']
    epochs = config['epochs']

    if (config['loss'] == 'CrossEntropy'):
        criterion = nn.CrossEntropyLoss();
        # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html    

    if (config['processor'] == 'cuda'):    
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); 
        # determine which device to use (gpu or cpu)
    else:
        device = torch.device("cpu");
    
    fname = '_' + config['model'] + '_' + config['transform'] + '_' + config['loss'] \
        + '_' + 'lr' + str(config['learning_rate'])
    
    tname = '\n'+ config['model'] + ' ' + config['transform'] + '\n' + \
        config['loss'] + ' ' + 'learning rate : ' + str(config['learning_rate'])

    return batch_size, n_class, device, fname, tname, \
        criterion, epochs, learning_rate, momentum

def Init(config):
    # TODO: Some missing values are represented by '__'. You need to fill these up.

    batch_size, n_class, device, __, __, __, __, learning_rate, momentum = \
        get_config_info(config);

    train_dataset = TASDataset('tas500v1.1',config) 
    val_dataset = TASDataset('tas500v1.1', config, eval=True, mode='val')
    test_dataset = TASDataset('tas500v1.1', config, eval=True, mode='test')


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

    return cnn_model, optimizer, train_loader, val_loader, test_loader

def rgb2int(arr):
    """
    Convert (N,...M,3)-array of dtype uint8 to a (N,...,M)-array of dtype int32
    """
    return arr[...,0]*(256**2)+arr[...,1]*256+arr[...,2]

def rgb2vals(color, color2ind):
   
    int_colors = rgb2int(color)
    int_keys = rgb2int(np.array(list(color2ind.keys()), dtype='uint8'))
    int_array = np.r_[int_colors.ravel(), int_keys]
    uniq, index = np.unique(int_array, return_inverse=True)
    color_labels = index[:int_colors.size]
    key_labels = index[-len(color2ind):]

    colormap = np.empty_like(int_keys, dtype='int32')
    colormap[key_labels] = list(color2ind.values())
    out = colormap[color_labels].reshape(color.shape[:2])

    return out


class TASDataset(Dataset):
    def __init__(self, data_folder, config, eval=False, mode=None):
        self.data_folder = data_folder
        self.eval = eval
        self.mode = mode
        self.config=config

        # You can use any valid transformations here

        # The following transformation normalizes each channel using the mean and std provided
        self.transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                              ])
        # we will use the following width and height to resize
        self.width = 768
        self.height = 384

        self.color2class = {
                #terrain
                (192,192,192): 0, (105,105,105): 0, (160, 82, 45):0, (244,164, 96): 0, \
                #vegatation
                ( 60,179,113): 1, (34,139, 34): 1, ( 154,205, 50): 1, ( 0,128,  0): 1, (0,100,  0):1, ( 0,250,154):1, (139, 69, 19): 1,\
                #construction
                (1, 51, 73):2, ( 190,153,153): 2, ( 0,132,111): 2,\
                #vehicle
                (0,  0,142):3, ( 0, 60,100):3, \
                #sky
                (135,206,250):4,\
                #object
                ( 128,  0,128): 5, (153,153,153):5, (255,255,  0 ):5, \
                #human
                (220, 20, 60):6, \
                #animal
                ( 255,182,193):7,\
                #void
                (220,220,220):8, \
                #undefined
                (0,  0,  0):9
        }

        self.input_folder = os.path.join(self.data_folder, 'train')
        self.label_folder = os.path.join(self.data_folder, 'train_labels')

        if self.eval:
            self.input_folder = os.path.join(self.data_folder, 'val')
            self.label_folder = os.path.join(self.data_folder, 'val_labels')
        
        image_names = os.listdir(self.input_folder)
        
        invalid_labels = ['1537962190852671077.png','1539600515553691119.png', '1539600738459369245.png','1539600829359771415.png','1567611260762673589.png']
            
        image_names = list(set(image_names).difference(set(invalid_labels)))
            
        self.paths = [(os.path.join(self.input_folder, i), os.path.join(self.label_folder, i)) for i in image_names]
        
        if self.mode == 'val': # use first 50 images for validation
            self.paths = self.paths[:50]
            
        elif self.mode == 'test': # use last 50 images for test
            self.paths = self.paths[50:]

    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):        
            
        image = np.asarray(PIL.Image.open(self.paths[idx][0]).resize((self.width, self.height)))
        mask_image = np.asarray(PIL.Image.open(self.paths[idx][1]).resize((self.width, self.height), PIL.Image.NEAREST))
        mask =  rgb2vals(mask_image, self.color2class)

        if self.transform:
            image = self.transform(image).float()

        return image, mask
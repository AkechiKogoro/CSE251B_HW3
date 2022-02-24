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


class Initialization():

    def __init__(self, config):
        # TODO: Some missing values are represented by '__'. You need to fill these up.
        self.config=config

        if (self.config['processor'] == 'cuda'):    
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); 
            # determine which device to use (gpu or cpu)
        else:
            self.device = torch.device("cpu");

    def __call__(self):
        train_dataset = TASDataset('tas500v1.1',self.config) 
        val_dataset = TASDataset('tas500v1.1', self.config, eval=True, mode='val')
        test_dataset = TASDataset('tas500v1.1', self.config, eval=True, mode='test')

        batch_size, n_class, learning_rate, momentum = self.config['batch_size'],\
            self.config['n_class'], self.config['lr'], self.config['gamma'];

        

        train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size= batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle=False)

        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases   

        self.CE_weight = None;
        if (self.config['loss'] == 'weightedCrossEntropy'):
            self.CE_weight = aug_weight(train_loader, n_class);
            #self.CE_weight = self.CE_weight.to(device);
        
        
        
        if (self.config['model'] == 'baseline'):
            cnn_model = FCN(n_class=n_class)
            cnn_model.apply(init_weights)
            cnn_model.to(self.device)
            
        optimizer = optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=momentum)
        # choose an optimizer

        cnn_model = cnn_model.to(self.device) #transfer the model to the device

        return cnn_model, optimizer, train_loader, val_loader, test_loader

    def __getitem__(self, idx):

        if (idx in {'n_class', 'batch_size', 'lr', 'epochs', 'early_stop_epoch', 'gamma'}):
            return self.config[idx]

        elif (idx == 'processor') :
            return self.device;

        elif (idx == 'fname'):
            fname = '_' + self.config['model'] + '_' + self.config['transform'] + '_' + self.config['loss'] \
        + '_' + 'lr' + str(self.config['lr']);
            return fname;

        elif (idx == 'tname'):
            tname = '\n'+ self.config['model'] + ' ' + self.config['transform'] + '\n' + \
                self.config['loss'] + ' ' + 'learning rate : ' + str(self.config['lr']);
            return tname
        
        elif (idx == 'loss'):
            if (self.config['loss'] == 'CrossEntropy'):
                criterion = nn.CrossEntropyLoss().to(self.device);
            elif (self. config['loss'] == 'weightedCrossEntropy'):
                criterion = nn.CrossEntropyLoss(weight = self.CE_weight).to(self.device)
            return criterion;
        
        else:
            return None;

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

        if (self.config['transform'] == 'NoTransform'):
            return image, mask
        elif (self.config['transform'] == 'Flip'):
            random.seed();
            p = random.randint(0,1);
            if (p == 0):
                return flip(image, mask)
            else:
                return image, mask
        elif (self.config['transform'] == 'Rotate'):
            return rotate(self.config, image, mask)
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
import random
import copy

def iou(pred, target, n_classes = 10):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for undefined class ("9")
    for cls in range(n_classes-1):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = torch.sum(pred_inds & target_inds) #complete this
        union = torch.sum(pred_inds | target_inds) #complete this

        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append( float(intersection / union) )  #complete this

    return np.array(ious)

def pixel_acc(pred, target, n_classes=10):
    #TODO complete this function, make sure you don't calculate the accuracy for undefined class ("9")
    same = torch.sum(pred == target) - torch.sum((pred==target) & (target == n_classes -1 ));


    total = pred.numel() - torch.sum ( target == n_classes-1);

    return float(same/total)

def flip(image, mask):
    new_mask = copy.deepcopy(np.fliplr(mask))
    return TF.hflip(image), new_mask

def rotate(config, image, mask, degree = 30):
    n_class = config['n_class']
    random.seed();
    d= float(random.randint(-degree, degree));

    new_image = TF.rotate(image, d, fill = 0)

    new_mask = torch.tensor(mask)
    new_mask = torch.unsqueeze(new_mask, 0)
    new_mask = TF.rotate(new_mask, d, fill = n_class - 1)
    new_mask = torch.squeeze(new_mask, 0)
    new_mask = new_mask.numpy();

    return new_image, new_mask

def aug_weight(train_loader, n_class, power=0.5):
    fre = torch.zeros(n_class)
    for __ , (__, label) in enumerate(train_loader):
        for cls in range(n_class):
            fre[cls] += torch.sum(label == cls);
    
    total_number = torch.sum(fre);

    weight = torch.pow( total_number/(fre+1), power );
    return weight
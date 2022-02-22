import numpy as np
import torch
import torch.optim as optim



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

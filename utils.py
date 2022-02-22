import numpy as np


def iou(pred, target, n_classes = 10):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for undefined class ("9")
  for cls in range(n_classes-1):  # last class is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = __ #complete this
    union = __ #complete this
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      __ #complete this

  return np.array(ious)

def pixel_acc(pred, target):
    #TODO complete this function, make sure you don't calculate the accuracy for undefined class ("9")
import torch
import torch.nn as nn


def get_criterion(criterion_name):
    
    if criterion_name == 'mse':
        return nn.MSELoss()
    
    if criterion_name == 'iou':
        return IoULoss()
        
    else:
        raise ValueError('Unknown criterion: {}'.format(criterion_name))
    
    
    
def op_sum(x):
    return x.sum(-1).sum(-1)
    
class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        self.EPSILON = 1e-6
            
    def forward(self, y_pred, y_true):
        inter = op_sum(y_true * y_pred)
        union = op_sum(y_true ** 2) + op_sum(y_pred ** 2) - op_sum(y_true * y_pred)
        iou = (inter + self.EPSILON) / (union + self.EPSILON)
        iou = torch.mean(iou)
        return 1-iou
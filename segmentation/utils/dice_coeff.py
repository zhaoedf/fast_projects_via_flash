


# def Dice_coeff(preds,target):
#     eps = 1e-6
#     b = preds.size(0)
#     p = preds.view(b,-1)
#     t = target.view(b,-1)
    
#     inter = (p*t).sum(1) + eps
#     union = p.sum(1) + t.sum(1) + eps
#     coeff = (2*inter /union)
    
#     return coeff


import torch
import torch.nn as nn

class Dice_coeff(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, pred, target):
        pred=pred[:,1,...]
        b = pred.size(0)
        p = pred.view(b,-1)
        t = target.view(b,-1)
        
        # print(pred.shape, target.shape)
        
        inter = (p*t).sum(1) + self.eps
        union = p.sum(1) + t.sum(1) + self.eps
        coeff = (2*inter /union)
        # print('$'*100, coeff.shape,coeff, target.max())
        return coeff
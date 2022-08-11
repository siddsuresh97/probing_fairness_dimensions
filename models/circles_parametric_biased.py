import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import entropy_wrapper

class UnfairClassifierParametric(nn.Module):
   
    def __init__(self, *args):
        super(UnfairClassifierParametric, self).__init__()
        self.alpha = 3
        
    def forward(self, x):
        """
        Inputs:
        - x : input data sample [size, shade]
        
        Outputs:
        - out: unnormalized output
        - prob_out: probability output
        """
        xflat = x.view(x.shape[0], -1)
        assert xflat.shape[1] == 2

        pbig = x[:, 1]*(torch.sigmoid(self.alpha*(x[:, 0] - 0.5))) + (1-x[:, 1])*(0.5)
        prob_out = torch.hstack((pbig, 1-pbig))

        # FIXME: this is a band-aid to put on wrapper
        entropy_prob_out = entropy_wrapper(prob_out)

        # return prob_out, out
        return entropy_prob_out, None
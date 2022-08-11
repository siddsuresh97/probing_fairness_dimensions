import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import entropy_wrapper

class CNN(nn.Module):
   
    def __init__(self, y_dim):
        """
        Initialize classifier

        Inputs:
        - y_dim : number of classes
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, y_dim)
        self.y_dim = y_dim
        
    def forward(self, x):
        """
        Perform classification using the CNN classifier
        
        Inputs:
        - x : input data sample
        
        Outputs:
        - out: unnormalized output
        - prob_out: probability output
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        out = self.fc2(x)
        prob_out = F.softmax(out)

        # FIXME: this is a band-aid to put on wrapper
        entropy_prob_out = entropy_wrapper(prob_out)

        # return prob_out, out
        return entropy_prob_out, None
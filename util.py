"""
Miscellaneous helper methods

Adapted from https://github.com/siplab-gt/generative-causal-explanations
"""

import numpy as np
import os
import torch
from GCE import GenerativeCausalExplainer

def load_mnist_idx(data_type):
       data_dir = 'data/mnist/'
       fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
       loaded = np.fromfile(file=fd,dtype=np.uint8)
       trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)
       fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
       loaded = np.fromfile(file=fd,dtype=np.uint8)
       trY = loaded[8:].reshape((60000)).astype(np.float)
       fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
       loaded = np.fromfile(file=fd,dtype=np.uint8)
       teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)
       fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
       loaded = np.fromfile(file=fd,dtype=np.uint8)
       teY = loaded[8:].reshape((10000)).astype(np.float)
       trY = np.asarray(trY)
       teY = np.asarray(teY)
       if data_type == "train":
           X = trX[0:50000,:,:,:]
           y = trY[0:50000].astype(np.int)
       elif data_type == "test":
           X = teX
           y = teY.astype(np.int)
       elif data_type == "val":
           X = trX[50000:60000,:,:,:]
           y = trY[50000:60000].astype(np.int)
       idxUse = np.arange(0,y.shape[0])
       seed = np.random.randint(2**32-1)
       np.random.seed(seed)
       np.random.shuffle(X)
       np.random.seed(seed)
       np.random.shuffle(y)
       np.random.seed(seed)
       np.random.shuffle(idxUse)

       return X/255.,y,idxUse
   
def load_mnist_classSelect(data_type,class_use,newClass):
    
    # X is numpy (nsamp,nrows,ncols,nchans)
    # Y is numpy (nsamp,)
    # idx is numpy (nsamp,)
    X, Y, idx = load_mnist_idx(data_type)
    class_idx_total = np.zeros((0,0))
    Y_use = Y
    
    count_y = 0
    for k in class_use:
        class_idx = np.where(Y[:]==k)[0]
        Y_use[class_idx] = newClass[count_y]
        class_idx_total = np.append(class_idx_total,class_idx)
        count_y = count_y +1
        
    class_idx_total = np.sort(class_idx_total).astype(int)

    X = X[class_idx_total,:,:,:]
    Y = Y_use[class_idx_total]
    return X,Y,idx

def load_circles_parametric(data_type,class_use,newClass):

    data_dir = './data/circle_shade_radius_stimuli'
    x = []
    y = []
    for img in os.listdir(data_dir):
        temp = img.split('_')
        try:
            shade = float(temp[1]) / 2 + 1
        except:
            print(shade, type(shade), img)
        try:
            radii = float(temp[-1].split('.png')[0])
        except:
            print(radii, type(radii), img) 
        x.append([radii, shade])
        if radii>0.5:
            y.append(1)
        else:
            y.append(0)

    idxUse = np.arange(0,y.shape[0])
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    np.random.seed(seed)
    np.random.shuffle(idxUse)

    X, Y, idx = load_mnist_idx(data_type)
    class_idx_total = np.zeros((0,0))
    Y_use = Y
    
    count_y = 0
    for k in class_use:
        class_idx = np.where(Y[:]==k)[0]
        Y_use[class_idx] = newClass[count_y]
        class_idx_total = np.append(class_idx_total,class_idx)
        count_y = count_y +1
        
    class_idx_total = np.sort(class_idx_total).astype(int)

    X = X[class_idx_total,:,:,:]
    Y = Y_use[class_idx_total]
    return X,Y,idx

def get_data_loader(dataset_type):

    if dataset_type == 'mnist':
        return load_mnist_classSelect
    elif dataset_type == 'circles_parametric':
        return load_circles_parametric


def get_classifier(classifier_type):

    if classifier_type == 'cnn':
        from models.CNN_classifier import CNN
        return CNN
    elif classifier_type == 'circles_parametric_biased':
        from models.circles_parametric_biased import UnfairClassifierParametric


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def get_new_gce(gce_type, K, L, c_dim, x_dim, classifier, device):

    if gce_type == 'CVAE':
        from models.CVAE import Decoder, Encoder
    elif gce_type == 'linear':
        from models.linear import Decoder, Encoder

    encoder = Encoder(K+L, c_dim, x_dim).to(device)
    decoder = Decoder(K+L, c_dim, x_dim).to(device)
    encoder.apply(weights_init_normal)
    decoder.apply(weights_init_normal)
    gce = GenerativeCausalExplainer(classifier, decoder, encoder, device)
    return gce

def entropy_wrapper(prob_out):
    """
    Inputs:
        prob_out: nsamp x nclass Tensor
    Outputs:
        entropy_prob_out: nsamp x 2 Tensor, evaluated as entropy of prob_out, converted into probability vector
    """
    normalized_entropy = torch.sum(-prob_out * torch.log(prob_out), 1).div(np.log(prob_out.shape[1]))
    normalized_entropy = normalized_entropy[:, None]
    entropy_prob_out = torch.hstack((normalized_entropy, 1-normalized_entropy))

    return entropy_prob_out

        

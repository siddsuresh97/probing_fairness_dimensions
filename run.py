"""
Main script for running GCE on circle classifiers

Adapted from https://github.com/siplab-gt/generative-causal-explanations
"""

import torch
import os
import numpy as np
import util
from GCE import GenerativeCausalExplainer
import scipy.io as sio
import plotting

# TODO: implement args

# --- global parameters ---
dataset_type = 'mnist'
randseed = 0
retrain_gce = True # train GCE from scratch
save_gce = True # save/overwrite pretrained GCE


# --- dataset-specific parameters ---

if dataset_type  == 'circles_parametric':
    data_classes = [0, 1]
    classifier_type = 'linear'
    classifier_path = './pretrained_classifiers/circle_parameteric'
    gce_path = './gce_models/circle_parametric_gce'
    gce_type = 'linear'

    K = 1 # number of causal factors
    L = 1 # number of non-causal factors

elif dataset_type == 'circles_raw':
    data_classes = [0, 1]
    classifier_type = 'linear'
    classifier_path = './pretrained_classifiers/circle_raw'
    gce_path = './gce_models/circle_raw_gce'
    gce_type = 'CVAE'

    K = 1 # number of causal factors
    L = 1 # number of non-causal factors

elif dataset_type == 'mnist':
    data_classes = [3, 8]
    classifier_type = 'cnn'
    classifier_path = './pretrained_classifiers/mnist_38_classifier'
    gce_path = './gce_models/mnist_38_gce'
    gce_type = 'CVAE'

    K = 1 # number of causal factors
    L = 7 # number of non-causal factors
    train_steps = 8000
    Nalpha = 25 # number of samples for information gain outer loop
    Nbeta = 100 # number of samples for information gain inner loop
    lam = 0.05 # data fidelity hyperparameter
    batch_size = 64
    lr = 5e-4


# --- initialize --- 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if randseed is not None:
    np.random.seed(randseed)
    torch.manual_seed(randseed)
ylabels = range(0,len(data_classes))


# --- load data --- 
# TODO: make this compatible with non-image data
load_data = util.get_data_loader(dataset_type)
X, Y, tridx = load_data('train', data_classes, ylabels)
vaX, vaY, vaidx = load_data('val', data_classes, ylabels)
ntrain, nrow, ncol, c_dim = X.shape
x_dim = nrow*ncol


# --- load classifier ---
# TODO: make compatible with non-CNN
classifier_object = util.get_classifier(classifier_type)

print('Initializing classifier...')
classifier = classifier_object(len(data_classes)).to(device)
print('Initialized!')

checkpoint = torch.load('%s/model.pt' % classifier_path, map_location=device)
classifier.load_state_dict(checkpoint['model_state_dict_classifier'])


# --- train/load GCE ---

if retrain_gce:
    gce = util.get_new_gce(gce_type, K, L, c_dim, x_dim, classifier, device)

    traininfo = gce.train(X, K, L,
                          steps=train_steps,
                          Nalpha=Nalpha,
                          Nbeta=Nbeta,
                          lam=lam,
                          batch_size=batch_size,
                          lr=lr)
    if save_gce:
        if not os.path.exists(gce_path):
            os.makedirs(gce_path)
        torch.save(gce, os.path.join(gce_path,'model.pt'))
        sio.savemat(os.path.join(gce_path, 'training-info.mat'), {
            'data_classes' : data_classes, 'classifier_path' : classifier_path,
            'K' : K, 'L' : L, 'train_step' : train_steps, 'Nalpha' : Nalpha,
            'Nbeta' : Nbeta, 'lam' : lam, 'batch_size' : batch_size, 'lr' : lr,
            'randseed' : randseed, 'traininfo' : traininfo})

else: # load pretrained model
    gce = torch.load(os.path.join(gce_path,'model.pt'), map_location=device)


# --- compute final information flow ---
I = gce.informationFlow()
Is = gce.informationFlow_singledim(range(0,K+L))
print('Information flow of K=%d causal factors on classifier output:' % K)
print(Is[:K])
print('Information flow of L=%d noncausal factors on classifier output:' % L)
print(Is[K:])


# --- generate explanation and create figure ---
sample_ind = np.concatenate((np.where(vaY == 0)[0][:4],
                             np.where(vaY == 1)[0][:4]))
x = torch.from_numpy(vaX[sample_ind])
zs_sweep = [-3., -2., -1., 0., 1., 2., 3.]
Xhats, yhats = gce.explain(x, zs_sweep)
plotting.plotExplanation(1.-Xhats, yhats, save_path='figs/demo')


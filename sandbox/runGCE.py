"""
Script to run Generative Causal Explainer on blackbox classifier
"""

import numpy as np
import scipy.io as sio
import os
import torch
#import util
#import plotting
from GCE import GenerativeCausalExplainer
import util


# --- parameters ---
# dataset
data_classes = [3, 8] # possible data classes

# vae
K = 1 # number of causal factors
L = 7 # number of non-causal factors
train_steps = 8000
Nalpha = 25 # number of outer loop samples for mutual information
Nbeta = 100 # number of inner loop samples for mutual information
lam = 0.05 # data fidelity hyperparameter
batch_size = 64
lr = 5e-4

# other
randseed = 0
gce_path = './my_gce'
retrain_gce = False # train explanatory VAE from scratch
save_gce = False # save/overwrite pretrained explanatory VAE at gce_path


# --- initialize ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if randseed is not None:
    np.random.seed(randseed)
    torch.manual_seed(randseed)
ylabels = range(0,len(data_classes))


# --- load data ---
#from load_mnist import load_mnist_classSelect
#X, Y, tridx = load_mnist_classSelect('train', data_classes, ylabels)
#vaX, vaY, vaidx = load_mnist_classSelect('val', data_classes, ylabels)

X, Y, tridx = util.load_data('train', data_classes, ylabels)
vaX, vaY, vaidx = util.load_data('val', data_classes, ylabels)

ntrain, nrow, ncol, c_dim = X.shape
x_dim = nrow*ncol


# --- load classifier ---
#from models.CNN_classifier import CNN
#classifier = CNN(len(data_classes)).to(device)
#checkpoint = torch.load('%s/model.pt' % classifier_path, map_location=device)
#classifier.load_state_dict(checkpoint['model_state_dict_classifier'])
classifier = util.load_classifier()

# --- train/load GCE ---
from models.CVAE import Decoder, Encoder
if retrain_gce:
    encoder = Encoder(K+L, c_dim, x_dim).to(device)
    decoder = Decoder(K+L, c_dim, x_dim).to(device)
    encoder.apply(util.weights_init_normal)
    decoder.apply(util.weights_init_normal)
    gce = GenerativeCausalExplainer(classifier, decoder, encoder, device)
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
            'data_classes' : data_classes,
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
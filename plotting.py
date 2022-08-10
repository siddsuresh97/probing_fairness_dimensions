import numpy as np
import matplotlib.pyplot as plt
#import gif

"""
surface plot

INPUTS:
    - ax : axis to draw figure on
    - x : numpy array corresponding to ROWS of Z (displayed on x-axis)
          x[0] corresponds to Z[0,:] and x[end] corresponds to Z[end,:]
    - y : numpy array corresponding to COLUMNS of Z (displayed on y-axis)
          y[0] corresponds to Z[:,0] and y[end] corresponds to Z[:,end]
    - Z : image to plot
    - clim : color limits for image; default: [min(Z), max(Z)]
"""
def plotsurface(ax, x, y, Z, clim=None):
    x = x.flatten()
    y = y.flatten()
    deltax = x[1]-x[0]
    deltay = y[1]-y[0]
    extent = (np.min(x)+deltax/2,
              np.max(x)-deltax/2,
              np.min(y)+deltay/2,
              np.max(y)-deltay/2)
    if clim == None:
        clim = [np.min(Z), np.max(Z)]
    im = ax.imshow(np.transpose(Z),
                   origin='lower',
                   extent=extent,
                   vmin=clim[0],
                   vmax=clim[1])
    return im


"""
plotExplanation - plot explanation created by GCE.explain().

Rows in output figure correspond to samples (first dimension of Xhats);
columns correspond to latent values in sweep.

:param Xhats: result from GCE.explain()
:param yhats: result from GCE.explain()
:param save_path: if provided, will export to {<save_path>_latentdimX.svg}
"""
def plotExplanation(Xhats, yhats, save_path=None):
    cols = [[0.047,0.482,0.863],[1.000,0.761,0.039],[0.561,0.788,0.227]]
    border_size = 3
    (nsamp,z_dim,nz_sweep,nrows,ncols,nchans) = Xhats.shape
    for latent_dim in range(z_dim):
        fig, axs = plt.subplots(nsamp, nz_sweep)
        for isamp in range(nsamp):
            for iz in range(nz_sweep):
                img = Xhats[isamp,latent_dim,iz,:,:,0].squeeze()
                yhat = int(yhats[isamp,latent_dim,iz])
                img_bordered = np.tile(np.expand_dims(np.array(cols[yhat]),(0,1)),
                    (nrows+2*border_size,ncols+2*border_size,1))
                img_bordered[border_size:-border_size,border_size:-border_size,:] = \
                    np.tile(np.expand_dims(img,2),(1,1,3))
                axs[isamp,iz].imshow(img_bordered, interpolation='nearest')
                axs[isamp,iz].axis('off')
        axs[0,round(nz_sweep/2)-1].set_title('Sweep latent dimension %d' % (latent_dim+1))
        if save_path is not None:
            plt.savefig('./%s_latentdim%d.svg' % (save_path,latent_dim+1), bbox_inches=0)
    
# #############################
# ########## IMPORTS ##########
# #############################

# Standard stuff
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-white')
from matplotlib import cm

# ML stuff
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Concatenate
import tensorflow as tf
from utils import join_models

# IFN Architectures
from Architectures.dnn import DNN
from Architectures.ifn import IFN, GaussianAnsatz

# Energy-flow package for CMS Open Data loader
import energyflow as ef
from energyflow.archs import PFN, EFN
from energyflow.utils import remap_pids



# ###############################
# ########## LOAD DATA ##########
# ###############################

def load_data(cache_dir, pt_lower, pt_upper, eta, quality, pad, x_dim = 3, momentum_scale = 250, n = 100000, amount = 1, max_particle_select = None, frac = 1.0, return_pfcs = True):

    # Load data
    specs = [f'{pt_lower} <= gen_jet_pts <= {pt_upper}', f'abs_jet_eta < {eta}', f'quality >= {quality}']
    sim = ef.mod.load(*specs, cache_dir = cache_dir, dataset='sim', amount= amount)

    # Gen_pt for Y
    Y1 = sim.jets_f[:,sim.gen_jet_pt]
    Y = np.zeros((Y1.shape[0], 1), dtype = np.float32 )
    Y[:,0] = Y1 / momentum_scale

    # Sim_pt for X
    X = np.zeros((Y1.shape[0],3), dtype = np.float32)
    X[:,0] = sim.jets_f[:,sim.jet_pt] / momentum_scale
    X[:,1] = sim.jets_f[:,sim.jet_eta]
    X[:,2] = sim.jets_f[:,sim.jet_phi]



    # CMS JEC's
    C = sim.jets_f[:,sim.jec]

    # PFC's
    pfcs = sim.particles

    # Shuffle and trim
    shuffle_indices = np.random.choice(np.arange(pfcs.shape[0]), size = int(pfcs.shape[0] * frac), replace=False)
    pfcs = pfcs[shuffle_indices]
    Y = Y[shuffle_indices]
    X = X[shuffle_indices]
    C = C[shuffle_indices]

    pfcs = pfcs[:n]
    Y = Y[:n]
    X = X[:n]
    C = C[:n]

    # PFC's
    dataset = np.zeros( (pfcs.shape[0], pad, x_dim), dtype = np.float32 )
    particle_counts = []
    if return_pfcs:
        for (i, jet) in enumerate(pfcs):
            size = min(jet.shape[0], pad)
            indices = (-jet[:,0]).argsort()
            dataset[i, :size, 0] = jet[indices[:size],0] / momentum_scale
            dataset[i, :size, 1] = jet[indices[:size],1]
            dataset[i, :size, 2] = jet[indices[:size],2]
            if x_dim == 4:
                dataset[i, :size, 3] = jet[indices[:size],4] # PID
            particle_counts.append(jet.shape[0])
        if x_dim == 4:
            remap_pids(dataset, pid_i = 3, error_on_unknown = False)

        for x in dataset:
            mask = x[:,0] > 0
            yphi_avg = np.average(x[mask,1:3], weights = x[mask,0], axis = 0)
            x[mask,1:3] -= yphi_avg  

    particle_counts = np.array(particle_counts)

    # Trim and shuffle
    if max_particle_select is not None:
        dataset = dataset[particle_counts < max_particle_select]
        Y = Y[particle_counts < max_particle_select]
        X = X[particle_counts < max_particle_select]
        C = C[particle_counts < max_particle_select]
        particle_counts = particle_counts[particle_counts < max_particle_select]

    shuffle_indices = np.random.choice(np.arange(dataset.shape[0]), size = int(dataset.shape[0] * frac), replace=False)

    print("X: ", X.shape, X.dtype)
    print("Y: ", Y.shape, Y.dtype)
    print("PFCs: ", dataset.shape, dataset.dtype)

    if not return_pfcs:
        return X, Y, C, particle_counts
   
    print("Max # of particles: %d" % max(particle_counts))
    return X, dataset, Y, C, particle_counts

def plot_mesh(model, pt_lower, pt_upper, momentum_scale, x_dim = 3):

    # Make meshgrid
    lower = pt_lower * 0.9 / momentum_scale
    upper = pt_upper * 1.1 / momentum_scale
    x = np.linspace(lower, upper, 40)
    y = np.linspace(lower, upper, 40)
    X, Y = np.meshgrid(x, y)

    # Evaluate Model
    x_ = np.ravel(X)
    x = np.zeros((x_.shape[0], x_dim)) # Zero out phi ans eta
    x[:,0] = x_
    Z = np.reshape(model.predict([x, np.ravel(Y)]), Y.shape)

    fig, ax = plt.subplots()

    # Plot
    contours = plt.contour(X, Y, Z, 8, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.contourf(X * momentum_scale , Y * momentum_scale , Z, 50, cmap = 'RdGy_r', origin='lower',  alpha=0.5)
    cbar = plt.colorbar()
    cbar.set_label("T(x,y)")

    # Identity
    plt.plot( [pt_lower * 0.9, pt_upper * 1.1] , [pt_lower * 0.9, pt_upper * 1.1], 'k--', alpha=0.75, zorder=0, label = 'Identity')
    

    # Plot embellishments
    plt.ylim(pt_lower * 0.9, pt_upper * 1.1)
    plt.xlim(pt_lower * 0.9, pt_upper * 1.1)
    plt.xlabel(r"Sim $p_T$ [GeV]")
    plt.ylabel(r"Gen $p_T$ [GeV]")
    plt.title("Maximum Likelihood Task")
    plt.legend()
    plt.grid()
    ax.tick_params(direction='in', length = 6, width = 2)


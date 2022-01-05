# #############################
# ########## IMPORTS ##########
# #############################

# Standard stuff
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-white')
from matplotlib import cm

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ML stuff
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Concatenate, LeakyReLU
import tensorflow as tf
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print("GPUs Available: ", tf.config.list_physical_devices("GPU"))

# IFN Architectures
from Architectures.dnn import DNN
from Architectures.ifn import IFN, gIFN
from Architectures.ifn import mine_loss, regulated_mine_loss, joint, marginal, MI
from Architectures.utils import build_gIFN_DNN, build_gIFN_EFN, build_gIFN_PFN


# Extra utils
from JEC.JEC_utils import load_data
from JEC.JEC_utils import plot_mesh
from JEC.config import param_dict, dataset_dict
from utils import plot_MI

# #################################
# ########## PARAMETERS ###########
# #################################

y_dim = 1
x_dim = 4
loadfile = None
savefile = param_dict["dict"] + "PFN_PID.hdf5"
retrain = param_dict["re-train"]

# Network Parameters
epochs = param_dict["epochs"]
pre_train_epochs = param_dict["pre_train_epochs"]
batch_size = param_dict["batch_size"]
pre_train_batch_size =  param_dict["pre_train_batch_size"]
Phi_sizes, F_sizes = param_dict["Phi_sizes"], param_dict["F_sizes"]

# Learning Parameters
learning_rate = param_dict["learning_rate"]
clipnorm = param_dict["clipnorm"]
l2_reg = param_dict["l2_reg"]
d_l1_reg = param_dict["d_l1_reg"]
d_multiplier = param_dict["d_multiplier"]

# Dataset Parameters
cache_dir = dataset_dict["cache_dir"]
momentum_scale = dataset_dict["momentum_scale"]
n = dataset_dict["n"]
pad = dataset_dict["pad"]
pt_lower, pt_upper = dataset_dict["pt_lower"], dataset_dict["pt_upper"]
eta = dataset_dict["eta"]
quality = dataset_dict["quality"]

# #############################
# ########## DATASET ##########
# #############################

X, PFCs, Y, C, N = load_data(cache_dir, pt_lower, pt_upper, eta, quality, pad, x_dim = x_dim, momentum_scale = momentum_scale, n = n, max_particle_select = 150, amount = dataset_dict["amount"])
X_test, PFCs_test, Y_test, C_test, N_test = load_data(cache_dir, pt_lower, pt_upper, eta, quality, pad, x_dim = x_dim, momentum_scale = momentum_scale, n = 50)
print(X.shape, PFCs.shape, Y.shape)


# ############################
# ########## MODELS ##########
# ############################

MI_histories = []
retrain_points = []
for train_count in range(retrain + 1):

    print("TRAINING %d" % (train_count))

    # Pretain
    if loadfile is None:
        ifn = build_gIFN_PFN(x_dim, y_dim, Phi_sizes, F_sizes, LeakyReLU(), l2_reg = l2_reg, d_l1_reg = d_l1_reg, d_multiplier = d_multiplier)
        print("PRE-TRAINING")
        ifn.pre_train([PFCs[:],Y[:]], epochs = pre_train_epochs, batch_size= pre_train_batch_size, verbose = True)
        ifn.save_weights(savefile)
        loadfile = savefile

    # Build Model 
    if param_dict["use_distributed_gpu"]:    
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            ifn = build_gIFN_PFN(x_dim, y_dim, Phi_sizes, F_sizes, LeakyReLU(), l2_reg = l2_reg, d_l1_reg = d_l1_reg, d_multiplier = d_multiplier)
            opt = tf.keras.optimizers.Adam(clipnorm = clipnorm, lr = learning_rate)
            ifn.compile(loss=mine_loss, optimizer=opt, metrics = [MI, joint, marginal])

    else:
        ifn = build_gIFN_PFN(x_dim, y_dim, Phi_sizes, F_sizes, LeakyReLU(), l2_reg = l2_reg, d_l1_reg = d_l1_reg, d_multiplier = d_multiplier)
        opt = tf.keras.optimizers.Adam(clipnorm = clipnorm, lr = learning_rate)
        ifn.compile(loss=mine_loss, optimizer=opt, metrics = [MI, joint, marginal])


    # Load a previous model, or pretrain
    if loadfile is not None:
        ifn.built = True
        ifn.load_weights(loadfile)

    # Fit
    history = ifn.fit([PFCs, Y],
            batch_size= batch_size,
            epochs = epochs,
            shuffle=True, verbose= 2)
    
    # Retrain checkpoints
    if train_count == retrain:
        ifn.save_weights(savefile)
    else:
        name, ext = os.path.splitext(savefile)
        checkpoint_savefile = "{name}_ckpt{checkpoint}{ext}".format(name = name, checkpoint = train_count, ext = ext) 
        loadfile = checkpoint_savefile
        ifn.save_weights(checkpoint_savefile)
        retrain_points = retrain_points + [(epochs) * (train_count + 1)]

    MI_history = history.history["MI"]
    MI_histories = MI_histories + MI_history

    # Remake hyperparameters for next training
    learning_rate = learning_rate / 10.0
    d_l1_reg = d_l1_reg * 10.0
    d_multiplier = d_multiplier / 2.0


plot_MI(epochs * (retrain + 1), MI_histories, os.path.splitext(savefile)[0] + '.png', retrain_points = retrain_points,  label = "PFN PID", title = "")


# #####################################
# ########## PLOTS AND TESTS ##########
# #####################################

ifn = build_gIFN_PFN(x_dim, y_dim, Phi_sizes, F_sizes, LeakyReLU(), l2_reg = l2_reg, d_l1_reg = d_l1_reg, d_multiplier = d_multiplier)
opt = tf.keras.optimizers.Adam(clipnorm = clipnorm, lr = learning_rate)
ifn.compile(loss=mine_loss, optimizer=opt, metrics = [MI, joint, marginal])
ifn.built = True
ifn.load_weights(loadfile)

# Predict values for test set
Y_pred = ifn.maximum_likelihood(PFCs_test)
covariance = ifn.covariance(PFCs_test)
sigmas = np.sqrt(np.abs(covariance[:,0,0]))

for i,j,k,l in zip(X_test, Y_pred, covariance, Y_test):
    print("infer y = %.3f +- %.3f (%.3f), true y = %.3f" % (j, np.sqrt(k[0,0]), (np.sqrt(k[0,0]) / j), l))

# # Mesh plot
# plot_mesh(ifn, pt_lower, pt_upper, momentum_scale, )

# # Errorbar plots
# plt.errorbar(X_test[:,0] * momentum_scale , Y_pred * momentum_scale, yerr=sigmas * momentum_scale, fmt='o', color='blue',
#              ecolor='k', elinewidth=1, capsize=2, label= r'$y_{ML}(x)$')

# plt.scatter(X_test[:,0] * momentum_scale , Y_test[:,0] * momentum_scale ,  color = 'red', label= r'True Gen $p_T$')
# plt.scatter(X_test[:,0] * momentum_scale , np.multiply(X_test[:,0] * momentum_scale , C_test[:] ) ,  color = 'green', label= r'SIM $\times$ CMS-JEC')
        
# plt.savefig("JEC/Plots/PFN.png")

# # Losses
# jec_loss = np.mean(np.square(np.multiply(X_test[:,0] * momentum_scale , C_test[:] )  - (Y_test * momentum_scale)))
# ml_loss = np.mean(np.square( (Y_pred * momentum_scale)  - (Y_test * momentum_scale)))

# print(jec_loss)
# print(ml_loss)# 
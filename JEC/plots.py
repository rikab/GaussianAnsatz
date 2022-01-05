# #############################
# ########## IMPORTS ##########
# #############################

# Standard stuff
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from matplotlib import pyplot as plt
plt.style.use('seaborn-white')
from matplotlib import cm
from scipy.stats import norm


import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ML stuff
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Concatenate, LeakyReLU
import tensorflow as tf

# IFN Architectures
from Architectures.dnn import DNN
from Architectures.ifn import IFN, gIFN
from Architectures.ifn import mine_loss, regulated_mine_loss, joint, marginal, MI
from Architectures.utils import build_gIFN_DNN, build_gIFN_EFN, build_gIFN_PFN, determine_constant


# Extra utils
from JEC.JEC_utils import load_data
from JEC.JEC_utils import plot_mesh
from utils import plot_MI

# #################################
# ########## PARAMETERS ###########
# #################################

y_dim = 1
x_dim = 3
loadfile = None
dnn_loadfile = "JEC/Models/DNN.hdf5"
efn_loadfile = "JEC/Models/EFN.hdf5"
pfn_loadfile = "JEC/Models/PFN.hdf5"
pfn_pid_loadfile = "JEC/Models/PFN_PID.hdf5"
cache_dir = "data"

colors = ['red', 'yellow', 'green', 'blue', "purple"]
labels = ['DNN', "EFN", "PFN", "PFN-PID", "CMS"]

# Dataset Parameters
momentum_scale = 1000
n = 250000
pad = 150
pad_EFN = 150
pad_PFN = 150
pt_lower, pt_upper = 695, 705
eta = 2.4
quality = 2
epochs = 150
d_multiplier = 0.0

# #############################
# ########## DATASET ##########
# #############################

X_test, PFCs_test, Y_test, C_test, N_test = load_data(cache_dir, pt_lower, pt_upper, eta, quality, pad, x_dim = 4, momentum_scale = momentum_scale, n = n, max_particle_select = 150)
test = (X_test, PFCs_test, Y_test, C_test)


plt.hist(N_test, bins=25, histtype = 'step', color = "red", label = "# of Particles", density=True)
plt.xlabel(r"$N$")
plt.ylabel("Density")
plt.title(r"Particle Count")
plt.grid()
plt.legend()
plt.savefig("JEC/Plots/particle_count.pdf")
plt.close()

# ############################
# ########## MODELS ##########
# ############################
# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# with strategy.scope():
opt = tf.keras.optimizers.Adam()
DNN = build_gIFN_DNN(x_dim, y_dim, [100,100,128], opt = opt)
EFN = build_gIFN_EFN(x_dim, y_dim, (100, 100, 128, ), (100, 100, 100, ), LeakyReLU(), pad_EFN, loadfile = efn_loadfile, d_multiplier = d_multiplier, opt = opt)
PFN = build_gIFN_PFN(x_dim, y_dim, (100, 100, 128, ), (100, 100, 100, ), LeakyReLU(),loadfile = pfn_loadfile, d_multiplier = d_multiplier, opt = opt)
PFN_pid = build_gIFN_PFN(x_dim + 1, y_dim, (100, 100, 128, ), (100, 100, 100, ), LeakyReLU(), loadfile = pfn_pid_loadfile, d_multiplier = d_multiplier, opt = opt)

# DNN loading is weird:
DNN.pre_train([X_test[:100],Y_test[:100]], epochs = 1, batch_size= 2, verbose = False)
DNN.load_weights(dnn_loadfile)


def dnn_predict(test, model, c = 0):

    yhat, sigma, T, MI = DNN.eval(test[0], test[2], loss = joint, c = c)
    return yhat * momentum_scale, sigma * momentum_scale, T, MI

def efn_predict(test, model, c = 0):
    yhat, sigma, T, MI = EFN.eval(test[1][:,:,:3], test[2], loss = joint, c = c)
    return yhat * momentum_scale, sigma * momentum_scale, T, MI

def pfn_predict(test, model, c = 0):
    print(test[1].shape)    
    yhat, sigma, T, MI = model.eval(test[1][:,:,:3], test[2], loss = joint, c = c)
    return yhat * momentum_scale, sigma * momentum_scale, T, MI

def pfn_pid_predict(test, model, c = 0):

    yhat, sigma, T, MI = model.eval(test[1], test[2], loss = joint, c = c)
    return yhat * momentum_scale, sigma * momentum_scale, T, MI

def cms_predict(test, model, c = 0):
    x = test[0]
    pt = x[:,0]
    yhat = momentum_scale * np.multiply(pt, test[3])
    jer = np.sqrt( 30/yhat**2 + 0.81/yhat + 0.04**2   )
    return yhat, np.multiply(yhat, jer), np.zeros_like(yhat), 0.0




# #####################################
# ########## PLOTS AND TESTS ##########
# #####################################

# Mutual Information plots
names = ["DNN", "EFN", "PFN", "PFN_pid"]

# for (i, name) in enumerate(names):
#     MI = np.load("JEC/Models/%s.npy" % name)
#     max_MI = np.amax(MI)
#     plt.plot(range(epochs), MI, color = colors[i], label = labels[i])
#     plt.ylim(0.8, 1.1 * max_MI)
#     plt.axhline(max_MI, linestyle = '--', color = colors[i], alpha = 0.5)

# plt.xlabel("Epochs")
# plt.ylabel("I(X;Y)")
# plt.title("Learned Mutual Information")
# plt.grid()
# plt.legend()
# plt.savefig("JEC/Plots/MI.pdf")
# plt.close()


# Make predictions
predictions = np.array((dnn_predict(test, DNN), efn_predict(test, EFN), pfn_predict(test, PFN), pfn_pid_predict(test, PFN_pid), cms_predict(test, None)))
print(predictions.shape)

# P_t distribution histograms
for i in range(4):
    mean, std = np.mean(predictions[i,0]), np.std(predictions[i,0])
    print(predictions[i,0].shape)
    plt.hist(predictions[i,0], bins=25, range=[550,850], histtype = 'step', color = colors[i], label = r'$\hat{y}_{%s}$; %0.1f$\pm$%0.1f GeV' % (labels[i], mean, std), density=True)
    

    # Gaussian Fit
    temp = predictions[i,0][predictions[i,0] < 850]
    temp = predictions[i,0][predictions[i,0] > 550]
    x = np.linspace(550, 850, 100)
    fit = norm.pdf(x, *(norm.fit(temp)))
    plt.plot(x,fit, color = colors[i] ,label = r"%s Fit; %0.1f $\pm$ %0.1f" % (labels[i], *norm.fit(temp) )  )  

# mean, std = momentum_scale * np.mean(X_test[:,0]), momentum_scale * np.std(X_test[:,0])
# plt.hist(momentum_scale * X_test[:,0], bins=25, range=[550,850], histtype = 'step', color = "black", label = r'%s; %0.1f$\pm$%0.1f GeV' % (r'Measured $p_T$', mean, std), density=True)

plt.xlabel(r"$p_T$ [GeV]")
plt.ylabel("Density")
plt.title(r"Distributions for Gen $p_T \in [695, 705]$ GeV")
plt.grid()
plt.legend()
plt.savefig("JEC/Plots/pt_hist.pdf")
plt.close()

# P_t uncertainty distribution histograms
for i in range(predictions.shape[0]):
    mean, std = np.mean(predictions[i,1]), np.std(predictions[i,1])
    print(predictions[i,1].shape)
    plt.hist(predictions[i,1], bins=50, range=[20,45], histtype = 'stepfilled', alpha = 0.25, color = colors[i], label = r'$\sigma_{%s}$; %0.1f$\pm$%0.1f GeV' % (labels[i], mean, std), density=True)
    plt.hist(predictions[i,1], bins=50, range=[20,45], histtype = 'step', color = colors[i], density=True)

plt.xlabel(r"$\sigma_{p_T}$ [GeV]")
plt.ylabel("Density")
plt.title(r"Distributions for Gen $p_T \in [695, 705]$ GeV")
plt.grid()
plt.legend(loc = "upper left")
plt.savefig("JEC/Plots/uncertainty_hist.pdf")
plt.close()


# Particle Number Uncertainty
for i in range(predictions.shape[0] - 1):
    stds = predictions[i,1]
    lower, upper = N_test[stds < 35], N_test[stds > 40]
    print(predictions[i,1].shape)
    plt.hist(lower, bins=25, histtype = 'step', color = colors[i], label = r'%s; $\sigma < 35$ GeV' % (labels[i]), density=True)
    plt.hist(upper, bins=25, histtype = 'step', color = colors[i], label = r'%s; $\sigma > 40$ GeV' % (labels[i]), density=True)



plt.xlabel(r"$N$")
plt.ylabel("Density")
plt.title(r"Distributions for Gen $p_T \in [695, 705]$")
plt.grid()
plt.legend()
plt.savefig("JEC/Plots/n_hist.pdf")
plt.close()




# ##############################
# ########## PT PLOTS ##########
# ##############################

sum_events = 0
consts = np.zeros((5,))

def const():
    X, PFCs, Y, C, N = load_data(500, 1000, eta, quality, pad, x_dim = 4, momentum_scale = momentum_scale, n = 1000000, max_particle_select = 150, frac = 0.1)

    consts[0] = marginal(None, DNN.predict([X, DNN.shuffle(Y)]))
    consts[1] = marginal(None, EFN.predict([PFCs[:,:,:3], EFN.shuffle(Y)]))
    consts[2] = marginal(None, PFN.predict([PFCs[:,:,:3], PFN.shuffle(Y)]))
    consts[3] = marginal(None, PFN_pid.predict([PFCs[:,:,:4], PFN.shuffle(Y)]))
const()
print(consts)

def get_values(pt_low, pt_high):

    X_test, PFCs_test, Y_test, C_test, N_test = load_data(pt_low, pt_high, eta, quality, pad, x_dim = 4, momentum_scale = momentum_scale, n = 1000000, max_particle_select = 150, frac = 0.1)
    test = (X_test, PFCs_test, Y_test, C_test)
    num_events = X_test.shape[0]

    pt = (pt_high + pt_low) / 2

    predictions = np.array((dnn_predict(test, DNN, consts[0]), efn_predict(test, EFN, consts[1]), pfn_predict(test, PFN, consts[2]), pfn_pid_predict(test, PFN_pid, consts[3]), cms_predict(test, None, consts[4])))
    means, mean_uncertainties = [], []
    resolutions, resolution_uncertainties = [], []
    MIs = []
    for i in range(5):

        mean, mean_uncertainty = [*norm.fit(predictions[i,0])]
        resolution, resolution_uncertainty = [*norm.fit(predictions[i,1])]
        means.append(mean / pt)
        mean_uncertainties.append(mean_uncertainty / pt)
        resolutions.append(resolution / pt)
        resolution_uncertainties.append(resolution_uncertainty / pt)

        MIs.append(predictions[i,3])

    return means, mean_uncertainties, resolutions, resolution_uncertainties, MIs, num_events


width = 10
num_entries = int((1000 - 500) / width) + 1
pts = np.linspace(500, 1000, num_entries)
print(pts)

means = np.zeros((5, num_entries))
mean_uncertainties = np.zeros((5, num_entries))
resolutions = np.zeros((5, num_entries))
resolution_uncertainties = np.zeros((5, num_entries))
MIs = np.zeros((5, num_entries))
num_events = np.zeros((num_entries,))

for i,pt in enumerate(pts):

    means[:,i], mean_uncertainties[:,i], resolutions[:,i], resolution_uncertainties[:,i], MIs[:,i], num_events[i] = get_values(pt - width / 2, pt + width /2)
    sum_events += num_events[i]

fig, axs = plt.subplots(3, sharex=True,)

for i in range(5):

    axs[0].errorbar(pts, means[i,:], xerr = 5, yerr = mean_uncertainties[i,:], color = colors[i])
    axs[1].errorbar(pts, resolutions[i,:], xerr = 5, yerr = resolution_uncertainties[i,:], color = colors[i])
    if i < 4:
        axs[2].plot(pts, MIs[i,:] , color = colors[i], label = labels[i] )
    print(np.log(num_events[:] / sum_events))

axs[0].set_ylabel(r"$\hat{JEC}$")
axs[1].set_ylabel(r"$\hat{JER}$")
axs[2].set_ylabel(r"$I(X;Y)$")

axs[2].set(xlabel = r"Gen $p_T$ [GeV]")
axs[2].legend()

plt.savefig("JEC/Plots/triple_plot.pdf")

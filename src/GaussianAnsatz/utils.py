# #############################
# ########## IMPORTS ##########
# #############################

# Standard stuff
import sys
import os


from energyflow.utils import remap_pids
from energyflow.archs import PFN, EFN
import energyflow as ef

from GaussianAnsatz.archs import mine_loss, joint, marginal, MI
from GaussianAnsatz.archs import GaussianAnsatz
from GaussianAnsatz.dnn import DNN


import tensorflow as tf
from keras.layers import Dense, Dropout, Input, Concatenate
from keras.models import Model
import keras

import numpy as np
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ML stuff

# IFN Architectures

# Energy-flow package for CMS Open Data loader


# ######################################
# ########## HELPER FUNCTIONS ##########
# ######################################


def build_gaussianAnsatz_DNN(x_dim, y_dim, layers, opt=None, l2_reg=0.0, d_l1_reg=0.0, d_multiplier=1.0, loadfile=None):
    """Helper function to build a basic gIFN DNN in one line

    Args:
        x_dim (int): X-dimension
        y_dim (int): Y-dimension
        layers (int array): Hidden layer sizes. All 4 networks use the same size
        opt (Keras optimizer, optional): If provided, compiles the network. Defaults to None.
        l2_reg (float, optional): L2 regularization to apply to all weights in all 4 networks. Defaults to 0.0.
        d_l1_reg (float, optional): L1 regularization to apply to the D-Network output. Defaults to 0.0.
        loadfile (string, optional): If provided, loads in weights from a file. Defaults to None.

    Returns:
        gIFN: [description]
    """

    model_A = DNN(x_dim, layers, 1, l2_regs=l2_reg)
    model_B = DNN(x_dim, layers, y_dim, l2_regs=l2_reg)
    model_C = DNN([x_dim, y_dim], layers, [y_dim, y_dim], symmetrize=True, l2_regs=l2_reg)
    model_D = DNN(x_dim, layers, y_dim, l2_regs=l2_reg)

    ifn = GaussianAnsatz(model_A, model_B, model_C, model_D, d_multiplier=d_multiplier, d_l1_reg=d_l1_reg, y_dim=y_dim)

    # Compile
    if opt is not None:
        ifn.compile(loss=mine_loss, optimizer=opt, metrics=[MI, joint, marginal])

    # Load a previous model, or pretrain
    if loadfile is not None:
        ifn.built = True
        ifn.load_weights(loadfile)

    return ifn


def build_gaussianAnsatz_EFN(x_dim, y_dim, Phi_layers, F_layers, acts, pad, opt=None, l2_reg=0.0, d_l1_reg=0.0, d_multiplier=1.0, loadfile=None):
    """Helper function to build a basic gIFN DNN in one line

    Args:
        x_dim (int): X-dimension
        y_dim (int): Y-dimension
        Phi_layers (int array): Hidden Phi layer sizes. All 4 networks use the same size
        F_layers (int array): Hidden F layer sizes. All 4 networks use the same size
        opt (Keras optimizer, optional): If provided, compiles the network. Defaults to None.
        l2_reg (float, optional): L2 regularization to apply to all weights in all 4 networks. Defaults to 0.0.
        d_l1_reg (float, optional): L1 regularization to apply to the D-Network output. Defaults to 0.0.
        loadfile (string, optional): If provided, loads in weights from a file. Defaults to None.

    Returns:
        gIFN: [description]
    """

    model_A = EFN(input_dim=x_dim-1, Phi_sizes=Phi_layers, F_sizes=F_layers, Phi_acts=acts, F_acts=acts,
                  output_act='linear', output_dim=1,  Phi_l2_regs=l2_reg, F_l2_regs=l2_reg, name_layers=False, ).model
    model_B = EFN(input_dim=x_dim-1, Phi_sizes=Phi_layers, F_sizes=F_layers, Phi_acts=acts, F_acts=acts,
                  output_act='linear', output_dim=y_dim,  Phi_l2_regs=l2_reg, F_l2_regs=l2_reg, name_layers=False, ).model
    model_D = EFN(input_dim=x_dim-1, Phi_sizes=Phi_layers, F_sizes=F_layers, Phi_acts=acts, F_acts=acts,
                  output_act='linear', output_dim=y_dim, Phi_l2_regs=l2_reg, F_l2_regs=l2_reg, name_layers=False,).model
    model_C = EFN(input_dim=x_dim-1, Phi_sizes=Phi_layers, F_sizes=F_layers, Phi_acts=acts, F_acts=acts,  output_act='linear',
                  output_dim=y_dim*y_dim, num_global_features=y_dim,  Phi_l2_regs=l2_reg, F_l2_regs=l2_reg, name_layers=False,).model

    # EFN Converter
    model_A = efn_input_converter(model_A, shape=(pad, x_dim))
    model_B = efn_input_converter(model_B, shape=(pad, x_dim))
    model_C = efn_input_converter(model_C, shape=(pad, x_dim), num_global_features=y_dim)
    model_D = efn_input_converter(model_D, shape=(pad, x_dim))

    ifn = GaussianAnsatz(model_A, model_B, model_C, model_D, d_multiplier=d_multiplier, y_dim=y_dim, d_l1_reg=d_l1_reg)

    # Compile
    if opt is not None:
        ifn.compile(loss=mine_loss, optimizer=opt, metrics=[MI, joint, marginal])

    # Load a previous model, or pretrain
    if loadfile is not None:
        ifn.built = True
        ifn.load_weights(loadfile)

    return ifn


def build_gaussianAnsatz_PFN(x_dim, y_dim, Phi_layers, F_layers, acts, opt=None, l2_reg=0.0, d_l1_reg=0.0, d_multiplier=1.0, loadfile=None):
    """Helper function to build a basic gIFN DNN in one line

    Args:
        x_dim (int): X-dimension
        y_dim (int): Y-dimension
        Phi_layers (int array): Hidden Phi layer sizes. All 4 networks use the same size
        F_layers (int array): Hidden F layer sizes. All 4 networks use the same size
        opt (Keras optimizer, optional): If provided, compiles the network. Defaults to None.
        l2_reg (float, optional): L2 regularization to apply to all weights in all 4 networks. Defaults to 0.0.
        d_l1_reg (float, optional): L1 regularization to apply to the D-Network output. Defaults to 0.0.
        loadfile (string, optional): If provided, loads in weights from a file. Defaults to None.

    Returns:
        gIFN: [description]
    """

    model_A = PFN(input_dim=x_dim, Phi_sizes=Phi_layers, F_sizes=F_layers, Phi_acts=acts, F_acts=acts,
                  output_act='linear', output_dim=1,  Phi_l2_regs=l2_reg, F_l2_regs=l2_reg, name_layers=False, ).model
    model_B = PFN(input_dim=x_dim, Phi_sizes=Phi_layers, F_sizes=F_layers, Phi_acts=acts, F_acts=acts,
                  output_act='linear', output_dim=y_dim,  Phi_l2_regs=l2_reg, F_l2_regs=l2_reg, name_layers=False, ).model
    model_D = PFN(input_dim=x_dim, Phi_sizes=Phi_layers, F_sizes=F_layers, Phi_acts=acts, F_acts=acts,
                  output_act='linear', output_dim=y_dim, Phi_l2_regs=l2_reg, F_l2_regs=l2_reg, name_layers=False,).model
    model_C = PFN(input_dim=x_dim, Phi_sizes=Phi_layers, F_sizes=F_layers, Phi_acts=acts, F_acts=acts,  output_act='linear',
                  output_dim=y_dim*y_dim, num_global_features=y_dim,  Phi_l2_regs=l2_reg, F_l2_regs=l2_reg, name_layers=False,).model

    ifn = GaussianAnsatz(model_A, model_B, model_C, model_D, d_multiplier=d_multiplier, y_dim=y_dim, d_l1_reg=d_l1_reg)

    # Compile
    if opt is not None:
        ifn.compile(loss=mine_loss, optimizer=opt, metrics=[MI, joint, marginal])

    # Load a previous model, or pretrain
    if loadfile is not None:
        ifn.built = True
        ifn.load_weights(loadfile)

    return ifn


def determine_constant(model, x1, y1, x2, y2):
    """Given a model trained using the DV Representation, the model will learn an arbitrary constant +c. This can constant can be decuded by looking at two *disjoint* datasets, x1 and x2.


    Args:
        model (IFN): [description]
        x1 (np.array): First partition
        x2 (np.array): Second partition

    Returns:
        c (float): The learned arbitrary constant  
    """

    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    p1 = x1.shape[0] / x.shape[0]
    p2 = x2.shape[0] / x.shape[0]

    i = model.eval_MI(x, y, mine_loss)
    i1 = model.eval_MI(x1, y1, mine_loss)
    i2 = model.eval_MI(x2, y2, mine_loss)

    return i - (i1 + np.log(p1)) - (i2 + np.log(p2))


def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


def pt(x, y):
    return np.sqrt(np.power(x, 2) + np.power(y, 2))


def phi(x, y):
    return np.arctan2(y, x)


def eta(E, z):
    return 0.5 * np.log((E + z) / (E - z))


def theta(pt, z):
    return np.arctan2(pt, z)


def cartesian_to_collider(x, norm=False, sort=False):

    y = np.zeros((x.shape[0], x.shape[1], 3))
    y[:, :, 0] = pt(x[:, :, 1], x[:, :, 2])
    if norm:
        for i in range(y.shape[0]):
            y[i, :, 0] = np.divide(y[i, :, 0], np.sum(y[i, :, 0]))
    mask = y[:, :, 0] > 0
    y[mask, 1] = phi(x[mask, 1], x[mask, 2])
    y[mask, 2] = eta(x[mask, 0], x[mask, 3])
    if sort:
        for i in range(y.shape[0]):
            indices = (-y[i, :, 0]).argsort()
            y[i] = y[i, indices]
    return y


def cartesian_to_ee_collider(x, norm=False, sort=False):

    y = np.zeros((x.shape[0], x.shape[1], 3))
    y[:, :, 0] = x[:, :, 0]
    if norm:
        for i in range(y.shape[0]):
            y[i, :, 0] = np.divide(y[i, :, 0], np.sum(y[i, :, 0]))
    mask = y[:, :, 0] > 0
    y[mask, 2] = phi(x[mask, 1], x[mask, 2])
    y[mask, 1] = theta(pt(x[mask, 1], x[mask, 2]), x[mask, 3])
    if sort:
        for i in range(y.shape[0]):
            indices = (-y[i, :, 0]).argsort()
            y[i] = y[i, indices]
    return y


def spherical_to_cartesian(X):
    Y = np.zeros((X.shape[0], X.shape[1], 4))
    for i in range(Y.shape[0]):
        radius = X[i, :, 0]
        theta = X[i, :, 1]
        phi = X[i, :, 2]
        Y[i, :, 0] = radius
        Y[i, :, 1] = radius * np.sin(theta) * np.cos(phi)
        Y[i, :, 2] = radius * np.sin(theta) * np.sin(phi)
        Y[i, :, 3] = radius * np.cos(theta)
    return Y


def sort(x):

    y = x.copy()
    if sort:
        for i in range(y.shape[0]):
            indices = (-y[i, :, 0]).argsort()
            y[i] = y[i, indices]
    return y


# Make MI plot
def plot_MI(epochs, history, filename=None, retrain_points=[], label="I(X;Y)", title="I(X;Y)", color="red"):

    MI = history
    if filename is not None:
        np.save(os.path.splitext(filename)[0] + '.npy', MI)
    max_MI = np.amax(MI)
    plt.plot(range(epochs), MI, color="red", label=label)
    plt.ylim(-0.1 * max_MI, 1.1 * max_MI)
    plt.axhline(max_MI, linestyle='--', color=color, alpha=0.5)
    plt.xlabel("Epochs")
    plt.ylabel("I(X;Y)")
    plt.title(title)
    # plt.grid()
    # plt.legend()

    for point in retrain_points:
        plt.axvline(x=point, color="grey", linestyle="--")


def join_models(model_X, model_Y, model_Z):
    """ Concanenate model_X and model_Y, then feed to model_Z

    Args:
        model_X (Model): [description]
        model_Y (Model): [description]
        model_Z (Model): [description]

    Returns:
        Model: A new Model Z( X, Y )
    """

    input_x = model_X.input
    input_y = model_Y.input
    output = model_Z(Concatenate()([model_X(input_x), model_Y(input_y)]))

    return Model([input_x, input_y], output)


def efn_input_converter(model_efn, shape=None, num_global_features=0):

    if num_global_features == 0:
        input_layer = Input(shape=shape)
        output = model_efn([input_layer[:, :, 0], input_layer[:, :, 1:]])
        return Model(input_layer, output)
    else:
        input_layer_1 = Input(shape=shape)
        input_layer_2 = Input(shape=(num_global_features,))
        output = model_efn([input_layer_1[:, :, 0], input_layer_1[:, :, 1:], input_layer_2])
        return Model([input_layer_1, input_layer_2], output)

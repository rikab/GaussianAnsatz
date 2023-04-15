# #############################
# ########## IMPORTS ##########
# #############################

# Standard stuff
import sys
import os
from utils import efn_input_converter
from energyflow.utils import remap_pids
from energyflow.archs import PFN, EFN
import energyflow as ef
from GaussianAnsatz.archs import mine_loss, joint, marginal, MI
from GaussianAnsatz.archs import GaussianAnsatz
from Architectures.dnn import DNN
import tensorflow as tf
from keras.layers import Dense, Dropout, Input, Concatenate
from keras.models import Model
import keras
from matplotlib import cm
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-white')

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

# standard library imports
from __future__ import absolute_import, division, print_function

# standard numerical library imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from keras.models import Model
from keras.layers import Dense, Dropout, Input, Concatenate

from itertools import repeat
import sys, os


def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
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
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


def pt(x, y):
    return np.sqrt(np.power(x,2) + np.power(y, 2))

def phi(x, y):
    return np.arctan2(y, x)

def eta(E, z):
    return 0.5 * np.log( (E + z) / (E - z))

def theta(pt, z):
    return np.arctan2(pt, z)

def cartesian_to_collider(x, norm = False, sort = False):

    y = np.zeros((x.shape[0], x.shape[1], 3))
    y[:,:,0] = pt(x[:,:,1],x[:,:,2] )
    if norm:
        for i in range(y.shape[0]):
            y[i,:,0] = np.divide(y[i,:,0], np.sum(y[i,:,0]))
    mask = y[:,:,0] > 0
    y[mask,1] = phi(x[mask,1],x[mask,2] )
    y[mask,2] = eta(x[mask,0],x[mask,3] )
    if sort:
        for i in range(y.shape[0]):
            indices = (-y[i,:,0]).argsort()
            y[i] = y[i,indices]
    return y



def cartesian_to_ee_collider(x, norm = False, sort = False):

    y = np.zeros((x.shape[0], x.shape[1], 3))
    y[:,:,0] = x[:,:,0]
    if norm:
        for i in range(y.shape[0]):
            y[i,:,0] = np.divide(y[i,:,0], np.sum(y[i,:,0]))
    mask = y[:,:,0] > 0
    y[mask,2] = phi(x[mask,1],x[mask,2] )
    y[mask,1] = theta(pt(x[mask,1],x[mask,2] ),x[mask,3] )
    if sort:
        for i in range(y.shape[0]):
            indices = (-y[i,:,0]).argsort()
            y[i] = y[i,indices]
    return y


def spherical_to_cartesian(X):
    Y = np.zeros((X.shape[0], X.shape[1], 4))
    for i in range(Y.shape[0]):
        radius = X[i,:,0]
        theta = X[i,:,1]
        phi = X[i,:,2]
        Y[i,:,0] = radius
        Y[i,:,1] = radius * np.sin(theta) * np.cos(phi)
        Y[i,:,2] = radius * np.sin(theta) * np.sin(phi)
        Y[i,:,3] = radius * np.cos(theta)
    return Y

def sort(x):

    y = x.copy()
    if sort:
        for i in range(y.shape[0]):
            indices = (-y[i,:,0]).argsort()
            y[i] = y[i,indices]
    return y


# Make MI plot
def plot_MI(epochs, history, filename, retrain_points = [], label = "I(X;Y)", title = "I(X;Y)"):

    MI = history
    np.save( os.path.splitext(filename)[0] + '.npy', MI)
    max_MI = np.amax(MI)
    plt.plot(range(epochs), MI, color = "red", label = label)
    plt.ylim(-0.1 * max_MI, 1.1 * max_MI)
    plt.axhline(max_MI, linestyle = '--', color = "grey", alpha = 0.5)
    plt.xlabel("Epochs")
    plt.ylabel("I(X;Y)")
    plt.title(title)
    plt.grid()
    plt.legend()

    for point in retrain_points:
        plt.axvline(x = point, color = "grey", linestyle = "--")

    plt.savefig(filename)


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

def efn_input_converter(model_efn, shape = None, num_global_features = 0):

    if num_global_features == 0:
        input_layer = Input(shape = shape)
        output = model_efn([input_layer[:,:, 0], input_layer[:,:,1:]])
        return Model(input_layer, output)
    else: 
        input_layer_1 = Input(shape = shape)
        input_layer_2 = Input(shape = (num_global_features,))
        output = model_efn([input_layer_1[:,:, 0], input_layer_1[:,:,1:], input_layer_2])
        return Model([input_layer_1, input_layer_2], output)



# return argument if iterable else make repeat generator
def iter_or_rep(arg):
    if isinstance(arg, (tuple, list)):
        if len(arg) == 1:
            return repeat(arg[0])
        else:
            return arg
    elif isinstance(arg, repeat):
        return arg
    else:
        return repeat(arg)




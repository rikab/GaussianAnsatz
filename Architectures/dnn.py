import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Concatenate
from keras import regularizers
import keras.backend as K
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

import numpy as np
import scipy.special
import sklearn.feature_selection

from matplotlib import pyplot as plt
from matplotlib import cm

from utils import iter_or_rep


# Function to construct 
def construct_dense_layers(sizes, acts = 'relu', k_inits = 'he_uniform', dropouts = 0.0, l2_regs = 0.0, names = None):

    # repeat options if singletons
    acts, k_inits, names = iter_or_rep(acts), iter_or_rep(k_inits), iter_or_rep(names)
    dropouts, l2_regs = iter_or_rep(dropouts), iter_or_rep(l2_regs)

    # iterate over layers:
    layers = []
    for size, act, k_init, dropout, l2_reg, name in zip(sizes, acts, k_inits, dropouts, l2_regs, names):
        layers.append(Dense(size, 
                  activation=act,
                  kernel_initializer=k_init,
                  bias_initializer=k_init,
                  kernel_regularizer=regularizers.l2(l2_reg), 
                  bias_regularizer=regularizers.l2(l2_reg),
                  name = name))
        if dropout > 0:
            dr_name = None if name is None else '{}_dropout'.format(name)
            layers.append(Dropout(dropout, name = dr_name))

    return layers


# Symmetrize (TODO: Expand to D > 2)
def symmetrize(x):
    return (x + tf.keras.layers.Permute((2,1)) (x) ) / 2.0



class DNN(keras.Model):


    def __init__(self, input_sizes, hidden_sizes, output_shape, acts = 'relu', output_act = 'linear', k_inits = 'he_uniform', dropouts = 0, l2_regs=0, names = None, symmetrize = True, input_merge = "concatenate"):

        super(DNN, self).__init__()

        # Hyperparameters
        self.input_sizes = input_sizes
        self.hidden_sizes = hidden_sizes
        self._output_shape = output_shape # self.output_shape is protected
        self.input_merge = input_merge

        self.acts = iter_or_rep(acts)
        self.output_act = output_act
        self.symmetrize = symmetrize

        self.k_inits = iter_or_rep(k_inits)
        self.dropouts = iter_or_rep(dropouts)
        self.names = iter_or_rep(names)
        self.l2_regs = iter_or_rep(l2_regs)

        if (self.symmetrize and not np.max(self._output_shape) == np.min(self._output_shape)):
            raise Exception("The output shape must be square to use symmetrize. The output shape was: {}".format(self._output_shape))   

        # Build model
        self.construct_model()


    def construct_model(self):

        # Construct hidden layers
        self.hidden_layers = construct_dense_layers(self.hidden_sizes, 
                                                    acts = self.acts, 
                                                    k_inits = self.k_inits,
                                                    dropouts = self.dropouts,
                                                    l2_regs = self.l2_regs,
                                                     names = self.names)

        # Construct outputs (TODO: SYMMETRIZE FOR D > 2)
        self.output_layers = []
        if isinstance(self._output_shape, (tuple, list)):
            self.output_layers.append(Dense(np.prod(self._output_shape), activation=  self.output_act,kernel_initializer = next(self.k_inits), bias_initializer=next(self.k_inits),  kernel_regularizer=regularizers.l2(next(self.l2_regs)),  bias_regularizer=regularizers.l2(next(self.l2_regs))))
            if len(self._output_shape) > 1:
                self.output_layers.append(tf.keras.layers.Reshape(self._output_shape, input_shape=(np.prod(self._output_shape),)))
                # if self.symmetrize:
                #     self.output_layers.append(symmetrize)
        else:
            self.output_layers.append(Dense(self._output_shape, activation= self.output_act, kernel_initializer = next(self.k_inits), bias_initializer=next(self.k_inits), kernel_regularizer=regularizers.l2(next(self.l2_regs)), bias_regularizer=regularizers.l2(next(self.l2_regs))))

        # Combine 
        self._layers = self.hidden_layers + self.output_layers

        # Add concatenate layer to input
        if isinstance(self.input_sizes, (tuple, list)):
            if len(self.input_sizes) > 1:
               self._layers =  [Concatenate()] + self._layers
       
    # Neural network function (input given output)
    def feed_forward(self, inputs):
                
        output_x = self.network_x(inputs[0])
        output_y = self.network_y(inputs[1])

        merged = keras.layers.concatenate([output_x, output_y])
        output = self.network_combined(merged)

        if symmetrize:
            output = (output + tf.keras.layers.Permute((2,1)) (output) ) / 2.0


        return output


    # Run instance of network to get output + derivatives
    def call(self, inputs, training = False):

        x = inputs
        for l in self._layers:
            x = l(x)

        return x

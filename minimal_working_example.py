import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Input
import keras.backend as K
import tensorflow as tf

import numpy as np
import scipy.special
import sklearn.feature_selection

from matplotlib import pyplot as plt
plt.style.use('seaborn-white')
from matplotlib import cm

from utils import iter_or_rep
from Architectures.dnn import DNN
from Architectures.ifn import IFN, gIFN


# #################################
# ########## PARAMETERS ###########
# #################################

x_dim = 1
y_dim = 1

epochs = 50
pre_train_epochs = 1
batch_size = 512
pre_train_batch_size = 128


# #############################
# ########## DATASET ##########
# #############################

N = 25000

# Gaussian Noise Dataset
X = np.random.uniform(low = -5, high = 5, size = (N, x_dim))
Y = X + np.random.normal(scale = 1, size = (N, x_dim))

# ############################
# ########## MODELS ##########
# ############################

# Need to initialize T(x,y) = a(x) + (y-b(x))d(x) + 1/2 (y-b(x))^T c(x,y) (y-b(x))
model_A = DNN(x_dim, [32, 32, 32, 32], 1)
model_B = DNN(x_dim, [32, 32, 32, 32], y_dim)
model_C = DNN([x_dim, y_dim], [32, 32, 32, 32], [y_dim, y_dim], symmetrize=False)
model_D = DNN(x_dim, [32, 32, 32, 32], y_dim, l2_regs= 0.1)

ifn = gIFN(model_A, model_B, model_C, model_D, d_multiplier= 0.0)

# Loss Function
def f_loss(out_joint, out_marginal):
    return -(tf.reduce_mean(out_joint, axis=0) - tf.reduce_mean(tf.math.exp(out_marginal - 1), axis = 0))
# MI Metric (Does not include l2 Reg losses)
def MI(out_joint, out_marginal):
    return (tf.reduce_mean(out_joint, axis=0) - tf.reduce_mean(tf.math.exp(out_marginal - 1), axis = 0))

# Compile and fit
ifn.compile(loss=f_loss, optimizer="adam", metrics = [MI])
ifn.pre_train([X,Y], epochs = pre_train_epochs, batch_size= pre_train_batch_size, verbose = True)
ifn.fit([X, Y],
            batch_size= batch_size,
            epochs = epochs)



# #####################################
# ########## PLOTS AND TESTS ##########
# #####################################

x_test = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).reshape((11, 1))
y_pred = ifn.maximum_likelihood(x_test)
covariance = ifn.covariance(x_test)
sigmas = np.sqrt(np.abs(covariance[:,0,0]))

for i,j,k in zip(x_test, y_pred, covariance):
    print("For x = %.3f, infer y = %.3f +- %.3f" % (i, j, np.sqrt(k[0,0])))


# Plotting meshes
x = np.linspace(-5, 5, 40)
y = np.linspace(-5, 5, 40)

X, Y = np.meshgrid(x, y)
Z = np.reshape(ifn.predict([np.ravel(X), np.ravel(Y)]), X.shape)


contours = plt.contour(X, Y, Z, 8, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.contourf(X, Y, Z, 50, cmap = 'RdGy_r', origin='lower',  alpha=0.5)
plt.xlabel("Measured X")
plt.ylabel("Inferred Y")
plt.title("Maximum Likelihood Task")
cbar = plt.colorbar()
cbar.set_label("T(x,y)")


plt.errorbar(x_test, y_pred, yerr=sigmas, fmt='o', color='blue',
             ecolor='skyblue', elinewidth=3, capsize=0)

plt.xlim(-5, 5)
plt.ylim(-5, 5)


plt.savefig("test.png")
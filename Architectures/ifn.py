import keras
from keras.models import Model
import keras.backend as K
import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm




class IFN(keras.Model):

    # Initialize
    def __init__(self):
        super(IFN, self).__init__()

    def __init__(self, network, is_DV = True):

        super(IFN, self).__init__()

        # Networks
        self.network = network
        self.is_DV = is_DV

    # Neural network function (input given output)

    def feed_forward(self, inputs):
                
        output = self.network(inputs)
        return output


    # Run instance of network to get output + derivatives

    def call(self, inputs, training = True):

        return self.feed_forward(inputs)


    # Maximum Likelihood Task
    def maximize(self, input, axis, epochs, samples = None):
        
        # Generate initial random seed
        if samples is None:
            if axis == 0:
                dim = self.x_shape[1]
            elif axis == 1:
                dim = self.y_shape[1]
            N = input.shape(0)
            samples = tf.Variable(tf.random.normal(mean= self.means[axis], stddev = self.vars[axis], shape = (N, dim)))
        else:
            samples = tf.Variable(samples)

        # Train Step
        for epoch in range(epochs):

            with tf.GradientTape() as tape:
                tape.watch(samples)
                out = self([input, samples], training = True) if axis == 0 else self([samples, input], training = True)
                out = out * -1

            gradients = tape.gradient(out, samples)
            self.optimizer.apply_gradients(zip([gradients], [samples]))

        return samples


    def custom_loss(self, x, y):
        return 0

    def shuffle(self, y):
        return tf.random.shuffle(y)


    def train_step(self, data):
  
        x, y = data[0]
        y_shuffle = self.shuffle(y)

        with tf.GradientTape() as tape:
            out_joint = self([x,y], training=True)  # Forward pass
            out_marginal = self([x,y_shuffle], training=True)  # Forward pass
        
            loss = self.compiled_loss(out_joint, out_marginal, regularization_losses=self.losses)
            loss += self.custom_loss(x, y)


        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights and metrics
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(out_joint, out_marginal)

        return {m.name: m.result() for m in self.metrics}


    
    def pre_train(self, train_dataset, epochs, batch_size = 64, verbose = False):
        
        x, y = train_dataset[0], train_dataset[1]


      
        # Instantiate a loss function.
        def loss(target_y, predicted_y):
            return tf.reduce_mean(tf.square(target_y - predicted_y))

        L = 1
        prev_MI_1 = 1
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            for batch in range(int(x.shape[0] / batch_size) - 5):

                current_index = batch_size * batch
                next_index = batch_size * (batch + 1)

                x_batch = x[current_index:next_index]
                y_batch = y[current_index:next_index]
                y_shuffle = tf.random.shuffle(y_batch)

                prev_MI_2 = prev_MI_1
                prev_MI_1 = L

                with tf.GradientTape() as tape:   

                    output_joint = self([x_batch, y_batch], training = True)
                    output_marginal = self([x_batch, y_shuffle], training = True)
                    # loss_value =  tf.reduce_mean(tf.square(output_joint))+ tf.reduce_mean(tf.square(output_marginal)) 
                    loss_value =  tf.reduce_mean(tf.square(output_joint+output_marginal))
                    loss_value += sum(self.losses)
                    # loss_value = tf.reduce_mean(tf.math.exp(output_marginal), axis = 0)

                grads = tape.gradient(loss_value, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                L = (tf.reduce_mean(output_joint, axis=0) + 1 - tf.reduce_mean(tf.math.exp(output_marginal), axis = 0))
                L1 = tf.reduce_mean(output_joint, axis=0) + 1
                L2 = tf.reduce_mean(tf.math.exp(output_marginal), axis = 0)
                if verbose: 
                    print("Epoch %d, Batch %d: Loss = %.3f, Joint = %.3f, Marginal = %0.3f, MI = %.3f" % (epoch, batch, loss_value, np.mean(output_joint), np.mean(output_marginal), L))

                # if np.abs(L) < 0.01 and np.abs(prev_MI_1) < 0.01 and np.abs(prev_MI_2) < 0.01:
                #     return




class GaussianAnsatz(IFN):

    # Initialize
    def __init__(self, network_A, network_B, network_C, network_D = None, d_multiplier = 1.0, d_l2_reg = 0.0, d_l1_reg = 0.0, x_dim = None, y_dim = None, is_efn = False):

        super(GaussianAnsatz, self).__init__(network_A)

        # Networks
        self.network_A = network_A
        self.network_B = network_B
        self.network_C = network_C
        self.network_D = network_D

        self.d_multiplier = d_multiplier
        self.d_l1_reg = d_l1_reg
        self.d_l2_reg = d_l2_reg

        self.y_dim = y_dim

    # D Network loss

    def custom_loss(self, x, y):
        return self.d_l2_reg * tf.reduce_sum(tf.math.square(self.network_D(x))) + self.d_l1_reg * tf.reduce_sum(tf.math.abs(self.network_D(x)))


    # Neural network function (input given output)

    def feed_forward(self, inputs):
                
        output_A = self.network_A(inputs[0])
        output_B = self.network_B(inputs[0])
        output_C = self.network_C([inputs[0], inputs[1]])
        output_C_symmetrized = output_C
        if (self.y_dim is not None):
            output_C_symmetrized = tf.keras.layers.Reshape( (self.y_dim, self.y_dim) )(output_C_symmetrized)
        output_D = self.network_D(inputs[0])

        difference = inputs[1] - output_B
        matmul = (tf.matmul(output_C_symmetrized , difference[...,None]))[...,0]

        output = output_A + self.d_multiplier*tf.keras.layers.Dot(axes = 1)([difference, output_D])  + 0.5 * tf.keras.layers.Dot(axes = 1)([difference, matmul ])

        return output



    def ensemble_maximum(self, y_ensemble):

        b = self.network_B.predict(y_ensemble)
        c = self.network_C.predict([b, y_ensemble])
        cov = tf.linalg.inv(tf.reduce_sum(c, 0))

        weighted_b = tf.reduce_sum(tf.matmul(c, b[...,None])[...,0], 0)
        x = tf.matmul(cov, weighted_b[...,None])[...,0]
        
        return x.numpy(), -1*cov.numpy()

    def pre_train(self, train_dataset, epochs, batch_size = 64, verbose = False):
        
        x, y = train_dataset[0], train_dataset[1]
        indices = np.random.permutation(x.shape[0])
        x = x[indices]
        y = y[indices]

        # Instantiate an optimizer.
        optimizer = tf.keras.optimizers.Adam()
        # Instantiate a loss function.
        def loss(target_y, predicted_y):
            return tf.reduce_mean(tf.square(target_y - predicted_y))

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            for batch in range(int(x.shape[0] / batch_size) - 5):

                current_index = batch_size * batch
                next_index = batch_size * (batch + 1)

                x_batch = x[current_index:next_index]
                y_batch = y[current_index:next_index]
                y_shuffle = tf.random.shuffle(y_batch)

                with tf.GradientTape() as tape:   

                    output_joint = self([x_batch, y_batch], training = True)
                    output_marginal = self([x_batch, y_shuffle], training = True)
                    b = self.network_B(x_batch)
                    a = loss(self.network_A(x_batch), 0)
                    b = loss(b, y_batch) 
                    c = self.network_C([x_batch, y_batch])
                    loss_value =  b
                    loss_value += a
                    loss_value +=  tf.reduce_mean(tf.square(output_joint))
                    loss_value += loss(c, - np.std(y_batch) *np.identity(y.shape[1]))
                    loss_value += sum(self.losses)
                    # loss_value = tf.reduce_mean(tf.math.exp(output_marginal), axis = 0)

                grads = tape.gradient(loss_value, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))
                L = (tf.reduce_mean(output_joint, axis=0)  -  tf.math.log(tf.reduce_mean(tf.boolean_mask(tf.math.exp(output_marginal), tf.math.is_finite(tf.math.exp(output_marginal))  ), axis = 0)))
                L1 = tf.reduce_mean(output_joint, axis=0) 
                L2 =  tf.math.log(tf.reduce_mean(tf.boolean_mask(tf.math.exp(output_marginal), tf.math.is_finite(tf.math.exp(output_marginal))  ), axis = 0))
                if verbose: 
                    print("Pre-Train Epoch %d, Batch %d: Loss = %.3f, a = %.3f b = %.3f, c = %.3f, Joint = %.3f, Marginal = %0.3f, MI = %.3f" % (epoch, batch, loss_value, np.mean(a), np.mean(b), np.mean(np.diagonal(c)), np.mean(output_joint), np.mean(output_marginal), L))


    def get_network_A(self):
        return self.network_A

    def get_network_B(self):
        return self.network_B

    def get_network_C(self):
        return self.network_C

    def maximum_likelihood(self, x):
        return self.network_B(x)

    def covariance(self, x):
        b = self.maximum_likelihood(x)
        # b = np.squeeze(b)
        output_c = self.network_C([x, b])
        tf.print(output_c)
        return -1 * tf.linalg.inv(tf.keras.layers.Reshape( (self.y_dim, self.y_dim) )(output_c)).numpy()
        
    def uncertainty(self, x):
        covariance = self.covariance(x)
        return np.sqrt(np.abs(covariance[:,0,0]))

    def eval_MI(self, x, y = None, loss = None, c = 0):

        if y is not None:
            T_joint = self([x, y]) - c
            T_marginal = self([x,self.shuffle(y)]) - c
            if loss == None:
                MI = -1*  self.compiled_loss(T_joint, T_marginal, regularization_losses= None)
            else:
                MI = loss(T_joint, T_marginal)

            return  MI.numpy()


    def eval(self, x, y = None, loss = None, c = 0):

        yhat = self.maximum_likelihood(x)
        sigma = self.uncertainty(x)  

        if y is not None:
            T_joint = self([x, y]) - c
            T_marginal = self([x,self.shuffle(y)]) - c
            MI = self.eval_MI(x,y,loss, c)

            return yhat, sigma, T_joint.numpy(), MI

        return yhat, sigma

# ###########################
# ######### LOSSES ##########
# ###########################

def f_loss(out_joint, out_marginal):
    return -(tf.reduce_mean(out_joint, axis=0) - tf.reduce_mean(tf.math.exp(out_marginal - 1), axis = 0))

def regulated_f_loss(out_joint, out_marginal):
    joint = tf.reduce_mean(out_joint, axis=0)
    marginal - tf.reduce_mean(tf.math.exp(out_marginal - 1), axis = 0)   

def mine_loss(out_joint, out_marginal):
    return -(tf.reduce_mean(out_joint, axis=0) - tf.math.log(tf.reduce_mean(tf.boolean_mask(tf.math.exp(out_marginal), tf.math.is_finite(tf.math.exp(out_marginal))  ), axis = 0)))

def regulated_mine_loss(out_joint, out_marginal):
    joint = tf.reduce_mean(out_joint, axis=0)
    marginal = tf.math.log(tf.reduce_mean(tf.boolean_mask(tf.math.exp(out_marginal), tf.math.is_finite(tf.math.exp(out_marginal))  ), axis = 0))
    print(joint, marginal)
    return tf.cond(tf.greater(marginal, tf.math.abs(joint * 1e6)),  lambda: -(joint - tf.reduce_mean(out_marginal, axis=0)), lambda: -(joint - marginal),)

def joint(out_joint, out_marginal):
    return tf.reduce_mean(out_joint, axis=0)

def marginal(out_joint, out_marginal):
    return tf.math.log(tf.reduce_mean(tf.boolean_mask(tf.math.exp(out_marginal), tf.math.is_finite(tf.math.exp(out_marginal))  ), axis = 0))

def MI(out_joint, out_marginal):
    return (tf.reduce_mean(out_joint, axis=0) - tf.math.log(tf.reduce_mean(tf.math.exp(out_marginal), axis = 0)))



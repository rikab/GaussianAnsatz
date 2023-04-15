from GaussianAnsatz.archs import IFN, GaussianAnsatz
import tensorflow as tf

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ################################
# ########## FACTORIZER ##########
# ################################


class Factorizer(IFN):

    def __init__(self, N, x_networks, y_networks,):

        super(IFN, self).__init__(None)

        # Inputs
        self.N = N
        self.x_networks = x_networks
        self.y_networks = y_networks

        # If only 1 network architecture is specified, create N copies
        if not isinstance(x_networks, (list, tuple)):
            self.x_networks = []
            for i in range(N):
                self.x_networks.append(tf.keras.models.clone_model(x_networks))
        if not isinstance(y_networks, (list, tuple)):
            self.y_networks = []
            for i in range(N):
                self.y_networks.append(tf.keras.models.clone_model(y_networks))

        # Weights
        self.f_weights = tf.Variable(initial_value=tf.ones(shape=(self.N,)) / self.N, trainable=True)

    def feed_forward(self, inputs):

        # Unpack
        x = inputs[0]
        y = inputs[1]

        # Concatenation trick
        output = self.x_networks[0](x) * self.y_networks[0](y)
        for i in range(1, self.N):
            output = tf.concat([output, self.x_networks[i](x) * self.y_networks[i](y)], axis=1)

        return tf.math.log(tf.reduce_sum(self.f_weights[None, :] * output, axis=1))

    # Overwrite train step to include projectors

    def train_step(self, data):
        metrics = super().train_step(data)
        # self.f_weights = self.projector(self.f_weights)
        return metrics

    # Simplex projector for weights
    def projector(self, x):

        x = tf.reshape(x, (1, -1,))

        m, n = x.shape
        cnt_m = tf.range(m)
        cnt_n = tf.range(n)
        u = tf.reverse(tf.sort(x, axis=1), axis=(1,))

        v = (tf.cumsum(u, axis=1) - 1) / tf.cast(cnt_n + 1, tf.float32)
        w = tf.gather(v, tf.reduce_sum(tf.cast(u > v, tf.int32), axis=1), batch_dims=1)

        return tf.reshape(tf.nn.relu(x - tf.reshape(w, (m, 1))), (-1, ))

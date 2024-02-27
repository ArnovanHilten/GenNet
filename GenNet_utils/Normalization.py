import tensorflow as tf
import tensorflow.keras.backend as K
import scipy.sparse as sp


class PerVariantNormalization(tf.keras.layers.Layer):
    def __init__(self, momentum=0.99, epsilon=1e-6, **kwargs):
        super(PerVariantNormalization, self).__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        # Initialize mean and variance for each feature (genetic variant)
        self.mean = self.add_weight(name='mean',
                                    shape=(input_shape[-1],),
                                    initializer='zeros',
                                    trainable=False)
        self.variance = self.add_weight(name='variance',
                                        shape=(input_shape[-1],),
                                        initializer='ones',
                                        trainable=False)

    def call(self, inputs, training=None):
        if training:
            # Compute mean and variance for the batch
            batch_mean, batch_variance = tf.nn.moments(inputs, axes=[0], keepdims=False)

            # Update the running mean and variance
            new_mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            new_variance = self.momentum * self.variance + (1 - self.momentum) * batch_variance

            self.mean.assign(new_mean)
            self.variance.assign(new_variance)

        # Normalize each feature (genetic variant) using the running mean and variance
        x = (inputs - self.mean) / tf.sqrt(self.variance + self.epsilon)
        return x

    def update_mean_and_variance(self, new_mean, new_variance):
        # Update the mean and variance for each genetic variant
        self.mean.assign(new_mean)
        self.variance.assign(new_variance)


class ConnectedNormalization(tf.keras.layers.Layer):
    def __init__(self, connectivity_matrix, momentum=0.99, epsilon=1e-6, **kwargs):
        super(ConnectedNormalization, self).__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon
        # Convert the provided COO matrix to TensorFlow SparseTensor
        self.connectivity_matrix = tf.sparse.SparseTensor(
            indices=np.array([connectivity_matrix.row, connectivity_matrix.col]).T,
            values=connectivity_matrix.data,
            dense_shape=connectivity_matrix.shape
        )

    def build(self, input_shape):
        # Number of neurons (columns of the connectivity matrix)
        num_neurons = self.connectivity_matrix.dense_shape[1]

        # Initialize mean and variance for each neuron
        self.mean = self.add_weight(name='mean',
                                    shape=(num_neurons,),
                                    initializer='zeros',
                                    trainable=False)
        self.variance = self.add_weight(name='variance',
                                        shape=(num_neurons,),
                                        initializer='ones',
                                        trainable=False)

    def call(self, inputs, training=None):
        # Normalize based on connectivity
        # Each neuron's statistics are based on connected inputs

        # Step 1: Calculate weighted sums of inputs for each neuron
        weighted_sums = tf.sparse.sparse_dense_matmul(self.connectivity_matrix, inputs, adjoint_a=True)

        # Step 2: Count connections for each neuron to compute means
        connection_counts = tf.reduce_sum(tf.sparse.to_dense(self.connectivity_matrix), axis=0)
        means = weighted_sums / connection_counts

        # Step 3: Compute variances
        squared_inputs = tf.pow(inputs, 2)
        weighted_squared_sums = tf.sparse.sparse_dense_matmul(self.connectivity_matrix, squared_inputs, adjoint_a=True)
        variances = weighted_squared_sums / connection_counts - tf.pow(means, 2)

        if training:
            # Update the running mean and variance
            self.mean.assign(self.momentum * self.mean + (1 - self.momentum) * means)
            self.variance.assign(self.momentum * self.variance + (1 - self.momentum) * variances)

        # Step 4: Normalize inputs
        normalized_inputs = (inputs - tf.sparse.sparse_dense_matmul(self.connectivity_matrix, tf.expand_dims(self.mean, 1))) / \
                            tf.sqrt(tf.sparse.sparse_dense_matmul(self.connectivity_matrix, tf.expand_dims(self.variance, 1)) + self.epsilon)

        return normalized_inputs

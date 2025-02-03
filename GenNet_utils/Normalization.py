import tensorflow as tf
import tensorflow.keras.backend as K
import scipy.sparse as sp


class PerVariantNormalization(tf.keras.layers.Layer):   # devision error
    def __init__(self, momentum=0.99, epsilon=1e-6, **kwargs):
        super(PerVariantNormalization, self).__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        # Initialize mean and variance for each feature (genetic variant)
        self.mean = self.add_weight(
            name='mean',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=False
        )
        self.variance = self.add_weight(
            name='variance',
            shape=input_shape[-1:],
            initializer='ones',
            trainable=False
        )
        super(PerVariantNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        # Compute batch mean and variance
        batch_mean, batch_variance = tf.nn.moments(inputs, axes=0)

        # Update running mean and variance
        new_mean = self.momentum * self.mean + (1.0 - self.momentum) * batch_mean
        new_variance = self.momentum * self.variance + (1.0 - self.momentum) * batch_variance

        # Create update ops
        mean_update = tf.compat.v1.assign(self.mean, new_mean)
        variance_update = tf.compat.v1.assign(self.variance, new_variance)

        # Add update ops to the layer's updates
        self.add_update([mean_update, variance_update])

        # Use batch statistics during training, running statistics during inference
        mean = tf.keras.backend.in_train_phase(batch_mean, self.mean, training=training)
        variance = tf.keras.backend.in_train_phase(batch_variance, self.variance, training=training)

        # # Normalize inputs
        # std_inv = tf.math.rsqrt(variance + self.epsilon)  # Use the selected variance
        # outputs = (inputs - mean) * std_inv  # Use the selected mean

        outputs = per_variant_normalization(inputs, mean, variance, self.epsilon)

        return outputs

@tf.custom_gradient
def per_variant_normalization(inputs, mean, variance, epsilon):
    # Reshape mean and variance to match inputs
    mean = tf.reshape(mean, [1, -1])
    variance = tf.reshape(variance, [1, -1])
    std = tf.sqrt(variance + epsilon)
    outputs = (inputs - mean) / std

    def grad(dy):
        # Gradient with respect to inputs
        dinputs = dy / std
        # Gradients with respect to mean and variance are None
        return dinputs, None, None, None

    return outputs, grad



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

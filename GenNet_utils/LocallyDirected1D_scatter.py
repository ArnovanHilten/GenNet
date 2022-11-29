# Locally-Directed1D layer
#
# For the article see: https://www.biorxiv.org/content/10.1101/2020.06.19.159152v1
# For an explanation how to use this layer see https://github.com/ArnovanHilten/GenNet
# Locallyconnected1D is used as a basis to write the LocallyDirected layer

"""Locally-Directed1D layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow import concat
from tensorflow.python.keras import backend as Kbb
from tensorflow.keras import backend as Kb
from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export

@keras_export('keras.layers.LocallyConnected1D')
class LocallyDirected1D(Layer):
    """Locally-Directed1D layer for 1D inputs.

    Dense layer with custom connections. The custom connections are defined by the mask input, a sparse (COO) connectivity matrix.

    # The matrix has the shape of (N_nodes_layer_1, N_nodes_layer_2).
    # It is a sparse matrix with zeros for no connections and ones if there is a connections. For example.


    #             output
    #           1 2 3 4 5
    # input 1 | 1 0 0 0 0 |
    # input 2 | 1 1 0 0 0 |
    # input 3 | 0 1 0 0 0 |
    # input 4 | 0 1 0 0 0 |
    # input 5 | 0 0 1 0 0 |
    # input 6 | 0 0 0 1 0 |
    # input 7 | 0 0 0 1 0 |


    # This connects the first two inputs (1,2) to the first neuron in the second layer.
    # Connects input 2,3 and 4 to output neuron 2.
    # Connects input 5 to output neuron 3
    # Connects input 6 and 7 o the 4th neuron in the subsequent layer
    # Connects nothing to the 5th neuron
    #
    # Writtem for Gennet framework: interpretable neural networks for phenotype prediction
    # (https://www.biorxiv.org/content/10.1101/2020.06.19.159152v1.full)


  Arguments:
      mask: sparse matrix with shape (input, output) connectivity matrix,
            True defines connection between (in_i, out_j), should be sparse (False,0) >> True
            should be scipy sparese matrix in COO Format!
      filters: Integer, the dimensionality of the output space
          (i.e. the number of output filters in the convolution).
      padding: Currently only supports `"valid"` (case-insensitive).
          `"same"` may be supported in the future.
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, length, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, length)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.

  Input shape:
      3D tensor with shape: `(batch_size, steps, input_dim)`

  Output shape:
      3D tensor with shape: `(batch_size, new_steps, filters)`
      `steps` value might have changed due to padding or strides.
  """

    def __init__(self,
                 mask,
                 filters,
                 padding='valid',
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(LocallyDirected1D, self).__init__(**kwargs)
        self.filters = filters
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=3)
        self.mask = mask
        self.mask_indices = []
        self.input_filters = None

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            input_dim, input_length = input_shape[1], input_shape[2]
        else:
            input_dim, input_length = input_shape[2], input_shape[1]

        if input_dim is None:
            raise ValueError('Axis 2 of input should be fully-defined. '
                             'Found shape:', input_shape)
            
        self.input_filters = input_dim
        self.output_length = self.mask.shape[1] 

        if self.data_format == 'channels_first':
            self.kernel_shape = (input_dim, input_length,
                                 self.filters, self.output_length)
        else:
            self.kernel_shape = (input_length, input_dim,
                                 self.output_length, self.filters)
        
        self.kernel = self.add_weight(shape=(len(self.mask.data), input_dim*self.filters ),    
        #sum of all nonzero values in mask sum(sum(mask))
                            initializer=self.kernel_initializer,
                            name='kernel',
                            regularizer=self.kernel_regularizer,
                            constraint=self.kernel_constraint)
                        
        self.mask_indices = np.array([self.mask.row, self.mask.col]).T

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.output_length, self.filters * self.input_filters),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None

        if self.data_format == 'channels_first':
            self.input_spec = InputSpec(ndim=3, axes={1: input_dim})
        else:
            self.input_spec = InputSpec(ndim=3, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        
        output_filters = []
   
        for input_filter_n in range(self.input_filters): 
            for i in range(self.filters):
                weight_matrix = tf.scatter_nd(self.mask_indices, self.kernel[:,i], (self.mask.shape[0], self.mask.shape[1]))
                output_mat = tf.matmul(inputs[:,:,input_filter_n], weight_matrix) # 
                output_mat = Kb.reshape(output_mat, [-1, self.output_length, 1])  
                output_filters.append(output_mat)

        
        output = concat(output_filters, axis=-1)   
        
        if self.use_bias:
            output = Kb.bias_add(output, self.bias, data_format=self.data_format)

        output = self.activation(output)
        return output
   

    def get_config(self):
        config = {
            # 'mask':  # replace by two numpy arrays with indices
            # self.mask,
            'filters':
                self.filters,
            'padding':
                self.padding,
            'data_format':
                self.data_format,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
        }
        base_config = super(LocallyDirected1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



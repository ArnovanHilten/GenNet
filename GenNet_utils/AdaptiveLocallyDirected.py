# For the article see https://www.biorxiv.org/content/10.1101/2020.06.19.159152v1
# For an explenation how to use this layer see https://github.com/ArnovanHilten/GenNet
# Locallyconnected1D is used as a basis to write the LocallyDirected layer
# ==============================================================================
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""LocallyDirected1D layer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.layers.AdaptiveLocallyDirected')
class AdaptiveLocallyDirected1D(tf.keras.layers.Layer):
    """AdaptiveLocallyDirected1D layer for 1D inputs.

  The `AdaptiveLocallyDirected1D` layer works as a sparese dense layer
  It takes a connectivity mask as an input, shape (n_inputs, n_outputs). The majority of the non-existend connections are not loaded into memory.
  A different set of filters is applied for each gene.

  Example:
  ```python
      model.add(AdaptiveLocallyDirected1D(mask = SNP_gene_mask, n_filters =1)
      model.add(LocallyDirected1D(32, 3))

  ```

  Arguments:
      mask: sparse matrix with shape (input, output) connectivity matrix,
            True defines connection between (in_i, out_j), should be sparse (False,0) >> True
            should be scipy sparese matrix in COO Format!
      filters: Integer, the dimensionality of the output space
          (i.e. the number of output filters in the convolution).
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
                 filter_size = None,
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
        super(AdaptiveLocallyDirected1D, self).__init__(**kwargs)
        self.mask = mask
        self.filters = filters
        self.filter_size = filter_size
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
        
        

    @tf_utils.shape_type_conversion
    def build(self, input_shape):

       
        if self.filter_size is None:
            self.max_gene_size = np.unique(self.mask.row, return_counts=True)[1].max()
            self.filer_size = self.max_gene_size 
        else:   
            if self.max_gene_size > self.filter_size:
                raise ValueError('filter_size must be greater than max_gene_size')

        if self.data_format == 'channels_first':
            input_dim, input_length = input_shape[1], input_shape[2]
        else:
            input_dim, input_length = input_shape[2], input_shape[1]

        if input_dim is None:
            raise ValueError('Axis 2 of input should be fully-defined. '
                             'Found shape:', input_shape)
        self.output_length = self.mask.shape[1]

        if self.data_format == 'channels_first':
            self.kernel_shape = (input_dim, input_length,
                                 self.filters, self.output_length)
        else:
            self.kernel_shape = (input_length, input_dim,
                                 self.output_length, self.filters)

        self.kernel = self.add_weight(shape=(self.filer_size, self.mask.shape[1]), 
                                      initializer=self.kernel_initializer,
                                      name='kernel', trainable=True,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.output_length, self.filters),
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
        # Reshape input to (input_per_gene, gene_size)
        sequence = []
        for i in range(self.mask.shape[1]):
            sequence.append(inputs[self.mask.row == i])

        reshaped_input = tf.keras.utils.pad_sequences(inputs, padding='post')

        # reshaped_input = tf.reshape(inputs, (-1, self.filer_size))

        # Multiply reshaped input with weight matrix and obtain diagonal elements
        # output = tf.einsum('ij,ji->i', reshaped_input, self.weight_matrix)
        output = tf.reduce_sum(reshaped_input * tf.transpose(self.weight_matrix), axis=1)

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format=self.data_format)

        output = self.activation(output)
        return output

    def get_config(self):  
        config = {
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
        base_config = super(AdaptiveLocallyDirected1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    

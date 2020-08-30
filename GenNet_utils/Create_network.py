import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(1, os.path.dirname(os.getcwd()) + "/GenNet_utils/")
import matplotlib

matplotlib.use('agg')
import tensorflow as tf
import tensorflow.keras as K
import scipy

tf.keras.backend.set_epsilon(0.0000001)
tf_version = tf.__version__  # ToDo use packaging.version
if tf_version <= '1.13.1':
    from GenNet_utils.LocallyDirectedConnected import LocallyDirected1D
elif tf_version >= '2.0':
    from GenNet_utils.LocallyDirectedConnected_tf2 import LocallyDirected1D
else:
    print("unexpected tensorflow version")
    from GenNet_utils.LocallyDirectedConnected_tf2 import LocallyDirected1D


def create_network_from_csv(datapath, l1_value=0.01, regression=False):
    masks = []

    def layer_block(model, mask, i):
        model = LocallyDirected1D(mask=mask, filters=1, input_shape=(mask.shape[0], 1),
                                  name="LocallyDirected_" + str(i))(model)
        model = K.layers.Activation("tanh")(model)
        model = K.layers.BatchNormalization(center=False, scale=False)(model)
        return model

    network_csv = pd.read_csv(datapath + "/topology.csv")
    network_csv = network_csv.filter(like="node", axis=1)
    columns = list(network_csv.columns.values)
    network_csv = network_csv.sort_values(by=columns, ascending=True)

    inputsize = len(network_csv)

    input_layer = K.Input((inputsize,), name='input_layer')
    model = K.layers.Reshape(input_shape=(inputsize,), target_shape=(inputsize, 1))(input_layer)

    for i in range(len(columns) - 1):
        network_csv2 = network_csv.drop_duplicates(columns[i])
        matrix_ones = np.ones(len(network_csv2[[columns[i], columns[i + 1]]]), np.bool)
        matrix_coord = (network_csv2[columns[i]].values, network_csv2[columns[i + 1]].values)
        mask = scipy.sparse.coo_matrix(((matrix_ones), matrix_coord),
                                       shape=(network_csv2[columns[i]].max() + 1,
                                              network_csv2[columns[i + 1]].max() + 1))
        masks.append(mask)
        model = layer_block(model, mask, i)

    model = K.layers.Flatten()(model)

    output_layer = K.layers.Dense(units=1, name="output_layer",
                                  kernel_regularizer=tf.keras.regularizers.l1(l=l1_value))(model)
    if regression:
        output_layer = K.layers.Activation("relu")(output_layer)
    else:
        output_layer = K.layers.Activation("sigmoid")(output_layer)

    model = K.Model(inputs=input_layer, outputs=output_layer)

    print(model.summary())

    return model, masks


def Lasso(inputsize, l1_value):
    inputs = K.Input((inputsize,), name='inputs')
    x1 = K.layers.BatchNormalization(center=False, scale=False, name="inter_out")(inputs)
    x1 = K.layers.Dense(units=1, kernel_regularizer=K.regularizers.l1(l1_value))(x1)
    x1 = K.layers.Activation("sigmoid")(x1)
    model = K.Model(inputs=inputs, outputs=x1)
    return model

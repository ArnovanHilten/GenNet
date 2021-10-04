import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(1, os.path.dirname(os.getcwd()) + "/GenNet_utils/")
import matplotlib

matplotlib.use('agg')
import tensorflow as tf
import tensorflow.keras as K
import scipy
import tables
tf.keras.backend.set_epsilon(0.0000001)
tf_version = tf.__version__  # ToDo use packaging.version
if tf_version <= '1.13.1':
    from GenNet_utils.LocallyDirectedConnected import LocallyDirected1D
elif tf_version >= '2.0':
    from GenNet_utils.LocallyDirectedConnected_tf2 import LocallyDirected1D
else:
    print("unexpected tensorflow version")
    from GenNet_utils.LocallyDirectedConnected_tf2 import LocallyDirected1D

def example_network():
    mask = scipy.sparse.load_npz('./folder/snps_gene.npz')
    masks = [mask]
    
    inputs_ = K.Input((mask.shape[0],), name='inputs_')
    layer_0 = K.layers.Reshape(input_shape=(mask.shape[0],), target_shape=(inputsize, 1))(inputs_)
    layer_1 = LocallyDirected1D(mask=mask, filters=1, input_shape=(inputsize, 1), name="gene_layer")(layer_0)
    layer_1 = K.layers.Flatten()(layer_1)
    layer_1 = K.layers.Activation("tanh")(layer_1)
    layer_1 = K.layers.BatchNormalization(center=False, scale=False, name="inter_out")(layer_1)
    layer_2 = K.layers.Dense(units=1)(layer_1)
    layer_2 = K.layers.Activation("sigmoid")(layer_2)
    model = K.Model(inputs=inputs_, outputs=layer_2)
    return model, masks

def layer_block(model, mask, i):
    model = LocallyDirected1D(mask=mask, filters=1, input_shape=(mask.shape[0], 1),
                              name="LocallyDirected_" + str(i))(model)
    model = K.layers.Activation("tanh")(model)
    model = K.layers.BatchNormalization(center=False, scale=False)(model)
    return model

def create_network_from_csv(datapath, inputsize, genotype_path, l1_value=0.01, regression=False):
    masks = []
    network_csv = pd.read_csv(datapath + "/topology.csv")
    network_csv = network_csv.filter(like="node", axis=1)
    columns = list(network_csv.columns.values)
    network_csv = network_csv.sort_values(by=columns, ascending=True)

    input_layer = K.Input((inputsize,), name='input_layer')
    model = K.layers.Reshape(input_shape=(inputsize,), target_shape=(inputsize, 1))(input_layer)

    for i in range(len(columns) - 1):
        matrix_ones = np.ones(len(network_csv[[columns[i], columns[i + 1]]]), np.bool)
        matrix_coord = (network_csv[columns[i]].values, network_csv[columns[i + 1]].values)
        if i == 0:
            matrixshape = (inputsize, network_csv[columns[i + 1]].max() + 1)
        else:
            matrixshape = (network_csv[columns[i]].max() + 1, network_csv[columns[i + 1]].max() + 1)
        mask = scipy.sparse.coo_matrix(((matrix_ones), matrix_coord), shape = matrixshape)
        masks.append(mask)
        model = layer_block(model, mask, i)

    model = K.layers.Flatten()(model)

    output_layer = K.layers.Dense(units=1, name="output_layer",
                                  kernel_regularizer=tf.keras.regularizers.l1(l=l1_value))(model)
    if regression:
        output_layer = K.layers.Activation("linear")(output_layer)
    else:
        output_layer = K.layers.Activation("sigmoid")(output_layer)

    model = K.Model(inputs=input_layer, outputs=output_layer)

    print(model.summary())

    return model, masks

def create_network_from_npz(datapath, inputsize, genotype_path, l1_value=0.01, regression=False):
    masks = []
    mask_shapes_x = []
    mask_shapes_y = []

    for npz_path in glob.glob(datapath + '/*.npz'):
        mask = scipy.sparse.load_npz(npz_path)
        masks.append(mask)
        mask_shapes_x.append(mask.shape[0])
        mask_shapes_y.append(mask.shape[1])

    for i in range(len(masks)):  # sort all the masks in the correct order
        argsort_x = np.argsort(mask_shapes_x)[::-1]
        argsort_y = np.argsort(mask_shapes_y)[::-1]
        
        mask_shapes_x = np.array(mask_shapes_x)
        mask_shapes_y = np.array(mask_shapes_y)
        assert all(argsort_x == argsort_y) # check that both dimensions have the same order

        masks  = [masks[i] for i in argsort_y] # sort masks
        mask_shapes_x = mask_shapes_x[argsort_x]
        mask_shapes_y = mask_shapes_y[argsort_y]

        for x in range(len(masks)-1): # check that the masks fit eachother
            assert mask_shapes_y[x] == mask_shapes_x[x + 1]

    assert mask_shapes_x[0] == inputsize
    if mask_shapes_y[-1] == 1:     # should we end with a dense layer?
        all_masks_available = True
    else:
        all_masks_available = False

    input_layer = K.Input((inputsize,), name='input_layer')
    model = K.layers.Reshape(input_shape=(inputsize,), target_shape=(inputsize, 1))(input_layer)

    for i in range(len(masks)):
        mask = masks[i]
        model = layer_block(model, mask, i)

    model = K.layers.Flatten()(model)

    if all_masks_available:
        output_layer = LocallyDirected1D(mask=masks[-1], filters=1, input_shape=(mask.shape[0], 1),
                          name="output_layer")(model)
    else:
        output_layer = K.layers.Dense(units=1, name="output_layer",
                                  kernel_regularizer=tf.keras.regularizers.l1(l=l1_value))(model)
    if regression:
        output_layer = K.layers.Activation("linear")(output_layer)
    else:
        output_layer = K.layers.Activation("sigmoid")(output_layer)

    model = K.Model(inputs=input_layer, outputs=output_layer)

    print(model.summary())

    return model, masks


def lasso(inputsize, l1_value):
    masks=[]
    inputs = K.Input((inputsize,), name='inputs')
    x1 = K.layers.BatchNormalization(center=False, scale=False, name="inter_out")(inputs)
    x1 = K.layers.Dense(units=1, kernel_regularizer=K.regularizers.l1(l1_value))(x1)
    x1 = K.layers.Activation("sigmoid")(x1)
    model = K.Model(inputs=inputs, outputs=x1)
    return model, masks
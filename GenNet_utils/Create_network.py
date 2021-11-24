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

   
    
def regression_height(inputsize, num_covariates=2, l1_value=0.001):
    mask = scipy.sparse.load_npz('/home/ahilten/repositories/pheno_height/Input_files/SNP_nearest_gene_mask.npz')
    masks = [mask]
    
    input_cov = K.Input((num_covariates,), name='inputs_cov')
    
    inputs_ = K.Input((mask.shape[0],), name='inputs_')
    layer_0 = K.layers.Reshape(input_shape=(mask.shape[0],), target_shape=(inputsize, 1))(inputs_)
    
    layer_1 = LocallyDirected1D(mask=mask, filters=1, input_shape=(inputsize, 1), name="gene_layer")(layer_0)
    layer_1 = K.layers.Flatten()(layer_1)
    layer_1 = K.layers.Activation("relu")(layer_1)
    layer_1 = K.layers.BatchNormalization()(layer_1)
    
    layer_2 = K.layers.Dense(units=10, kernel_regularizer=tf.keras.regularizers.l1(l=l1_value))(layer_1)
    layer_2 = K.layers.Activation("relu")(layer_2) 
    
    layer_3 = K.layers.concatenate([layer_2, input_cov], axis=1)
    layer_3 = K.layers.BatchNormalization()(layer_3)
    layer_3 = K.layers.Dense(units=10)(layer_3)
    layer_3 = K.layers.Activation("relu")(layer_3) 
        
    layer_4 = K.layers.Dense(units=10)(layer_3)
    layer_4 = K.layers.Activation("relu")(layer_4) 
    
    layer_5 = K.layers.Dense(units=1, bias_initializer= tf.keras.initializers.Constant(168))(layer_4)
    layer_5 = K.layers.Activation("relu")(layer_5)
    
    model = K.Model(inputs=[inputs_, input_cov], outputs=layer_5)
    
    print(model.summary())
    
    return model, masks
    
    
def example_network():
    mask = scipy.sparse.load_npz('./folder/snps_gene.npz')
    masks = [mask]
    
    inputs_ = K.Input((mask.shape[0],), name='inputs_')
    input_cov = K.Input((num_covariates,), name='inputs_cov')
    
    layer_0 = K.layers.Reshape(input_shape=(mask.shape[0],), target_shape=(inputsize, 1))(inputs_)
    
    layer_1 = LocallyDirected1D(mask=mask, filters=1, input_shape=(inputsize, 1), name="gene_layer")(layer_0)
    layer_1 = K.layers.Flatten()(layer_1)
    layer_1 = K.layers.Activation("relu")(layer_1)
    layer_1 = K.layers.BatchNormalization(center=False, scale=False, name="inter_out")(layer_1)
    
    layer_2 = K.layers.Dense(units=1)(layer_1)
    layer_2 = K.layers.Activation("relu")(layer_2)
    model = K.Model(inputs=[inputs_, input_cov], outputs=layer_2)
    print(model.summary())
    
    return model, masks



def layer_block(model, mask, i, regression):
    
    if regression:
        activation_type="relu"
    else:
        activation_type="tanh"
    
    model = LocallyDirected1D(mask=mask, filters=1, input_shape=(mask.shape[0], 1),
                              name="LocallyDirected_" + str(i))(model)
    model = K.layers.Activation(activation_type)(model)
    model = K.layers.BatchNormalization(center=False, scale=False)(model)
    return model


def activation_layer(model, regression):
    if regression:
        model = K.layers.Activation("linear")(model)
    else:
        model = K.layers.Activation("sigmoid")(model)
    return model


def add_covariates(model, num_covariates, regression):
    if num_covariates > 0:
        model = activation_layer(model, regression)
        model = K.layers.concatenate([model, input_cov], axis=1)
        model = K.layers.BatchNormalization()(model)
        model = K.layers.Dense(units=1)(model)
    return model


def create_network_from_csv(datapath, inputsize, genotype_path, l1_value=0.01, regression=False, num_covariates=0):
    masks = []
    
    network_csv = pd.read_csv(datapath + "/topology.csv")
    network_csv = network_csv.filter(like="node", axis=1)
    columns = list(network_csv.columns.values)
    network_csv = network_csv.sort_values(by=columns, ascending=True)

    input_layer = K.Input((inputsize,), name='input_layer')
    input_cov = K.Input((num_covariates,), name='inputs_cov')
    
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
        model = layer_block(model, mask, i, regression)

    model = K.layers.Flatten()(model)

    model = K.layers.Dense(units=1, name="output_layer",
                                  kernel_regularizer=tf.keras.regularizers.l1(l=l1_value))(model)
    
    model = add_covariates(model, num_covariates, regression)
    
    output_layer = activation_layer(model, regression)
   

    model = K.Model(inputs=[input_layer, input_cov], outputs=output_layer)

    print(model.summary())

    return model, masks


def create_network_from_npz(datapath, inputsize, genotype_path, l1_value=0.01, regression=False, num_covariates=0):
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
    input_cov = K.Input((num_covariates,), name='inputs_cov')
    model = K.layers.Reshape(input_shape=(inputsize,), target_shape=(inputsize, 1))(input_layer)

    for i in range(len(masks)):
        mask = masks[i]
        model = layer_block(model, mask, i, regression)

    model = K.layers.Flatten()(model)

    if all_masks_available:
        model = LocallyDirected1D(mask=masks[-1], filters=1, input_shape=(mask.shape[0], 1),
                          name="output_layer")(model)
    else:
        model = K.layers.Dense(units=1, name="output_layer",
                                  kernel_regularizer=tf.keras.regularizers.l1(l=l1_value))(model)
    
    
    model = add_covariates(model, num_covariates, regression)
    
    output_layer = activation_layer(model, regression)
    model = K.Model(inputs=[input_layer, input_cov], outputs=output_layer)

    print(model.summary())

    return model, masks


def lasso(inputsize, l1_value, num_covariates=0, regression=False):
    masks=[]
    inputs = K.Input((inputsize,), name='inputs')
    input_cov = K.Input((num_covariates,), name='inputs_cov')
    model = K.layers.BatchNormalization(center=False, scale=False, name="inter_out")(inputs)
    model = K.layers.Dense(units=1, kernel_regularizer=K.regularizers.l1(l1_value))(model)
    
    model = add_covariates(model, num_covariates, regression)
    
    output_layer = K.layers.Activation("sigmoid")(model)
    
    model = K.Model(inputs=[inputs, input_cov], outputs=output_layer)
    return model, masks
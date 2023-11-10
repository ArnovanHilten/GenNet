import os
import sys
import glob
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib

matplotlib.use('agg')
import tensorflow as tf
import tensorflow.keras as K
import scipy
import tables
tf.keras.backend.set_epsilon(0.0000001)
tf_version = tf.__version__  # ToDo use packaging.version

if tf_version <= '1.13.1':
    from GenNet_utils.LocallyDirected1D import LocallyDirected1D
elif tf_version >= '2.0':
    from GenNet_utils.LocallyDirected1D import LocallyDirected1D
else:
    print("unexpected tensorflow version")
    from GenNet_utils.LocallyDirected1D import LocallyDirected1D
    
    
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


def regression_properties(datapath):
    ytrain = pd.read_csv(datapath + "subjects.csv")
    mean_ytrain = float(ytrain[ytrain["set"] == 1]["labels"].mean())
    negative_values_ytrain = float(ytrain[ytrain["set"] == 1]["labels"].min()) < 0
    return mean_ytrain, negative_values_ytrain


def layer_block(model, mask, i, regression, L1_act =0.01):
    if regression:
        activation_type="relu"
    else:
        activation_type="tanh"
    
    model = LocallyDirected1D(mask=mask, filters=1, input_shape=(mask.shape[0], 1),
                              name="LocallyDirected_" + str(i), activity_regularizer=K.regularizers.l1(L1_act))(model)
    model = K.layers.Activation(activation_type)(model)
    model = K.layers.BatchNormalization(center=False, scale=False)(model)
    return model


def activation_layer(model, regression, negative_values_ytrain):
   
    if regression: 
        if negative_values_ytrain:
            model = K.layers.Activation("linear")(model)
            print('using a linear activation function')
        else:
            model = K.layers.Activation("relu")(model)
            print('using a relu activation function')
    else:
        model = K.layers.Activation("sigmoid")(model)
        
    return model

def one_hot_input(input_layer):
    model = K.layers.LocallyConnected1D(filters=1, strides=1, kernel_size=1, 
                                            name="SNP_layer", implementation=3
                                           )(input_layer)
    model = K.layers.Activation('linear', name="snp_activation")(model) 
    return model
    

def add_covariates(model, input_cov, num_covariates, regression, negative_values_ytrain, mean_ytrain, l1_value, L1_act):
    if num_covariates > 0:
        model = activation_layer(model, regression, negative_values_ytrain)
        model = K.layers.concatenate([model, input_cov], axis=1)
        model = K.layers.BatchNormalization(center=False, scale=False)(model)
        model = K.layers.Dense(units=1, name="output_layer_cov",
                       kernel_regularizer=tf.keras.regularizers.l1(l=l1_value),
                       activity_regularizer=K.regularizers.l1(L1_act),
                       bias_initializer= tf.keras.initializers.Constant(mean_ytrain))(model)
    return model

def create_network_from_npz(datapath,
                            inputsize,
                            genotype_path,
                            l1_value=0.01,
                            L1_act =0.01,
                            regression=False,
                            one_hot = False,
                            num_covariates=0,
                            mask_order = []):
    print("Creating networks from npz masks")
    print("regression", regression)
    print("one_hot", one_hot)
    if regression:
        mean_ytrain, negative_values_ytrain = regression_properties(datapath)
    else:
        mean_ytrain = 0
        negative_values_ytrain = False

    masks = []
    mask_shapes_x = []
    mask_shapes_y = []

    print(mask_order)

    if len(mask_order) > 0:  # if mask_order is defined we use this order
        for mask in mask_order:
            mask = scipy.sparse.load_npz(datapath + '/'+str(mask)+'.npz')
            masks.append(mask)
            mask_shapes_x.append(mask.shape[0])
            mask_shapes_y.append(mask.shape[1])

        for x in range(len(masks) - 1):  # check that the masks fit eachother
            assert mask_shapes_y[x] == mask_shapes_x[x + 1]
    else:
        # if mask order is not defined we can sort the mask by the size
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
            assert all(argsort_x == argsort_y)  # check that both dimensions have the same order

            masks = [masks[i] for i in argsort_y]  # sort masks
            mask_shapes_x = mask_shapes_x[argsort_x]
            mask_shapes_y = mask_shapes_y[argsort_y]

            for x in range(len(masks) - 1):  # check that the masks fit eachother
                assert mask_shapes_y[x] == mask_shapes_x[x + 1]

    assert mask_shapes_x[0] == inputsize
    if mask_shapes_y[-1] == 1:  # should we end with a dense layer?
        all_masks_available = True
    else:
        all_masks_available = False

    input_cov = K.Input((num_covariates,), name='inputs_cov')
    
    if one_hot:
        input_layer = K.Input((inputsize, 3), name='input_layer')
        model = one_hot_input(input_layer)
    else:
        input_layer = K.Input((inputsize,), name='input_layer')
        model = K.layers.Reshape(input_shape=(inputsize,), target_shape=(inputsize, 1))(input_layer)

    for i in range(len(masks)):
        mask = masks[i]
        model = layer_block(model, mask, i, regression, L1_act=L1_act)

    model = K.layers.Flatten()(model)

    if all_masks_available:
        model = LocallyDirected1D(mask=masks[-1], filters=1, input_shape=(mask.shape[0], 1),
                                  name="output_layer")(model)
    else:
        model = K.layers.Dense(units=1, name="output_layer",
                               kernel_regularizer=tf.keras.regularizers.l1(l=l1_value),
                               activity_regularizer=K.regularizers.l1(L1_act),
                               bias_initializer= tf.keras.initializers.Constant(mean_ytrain))(model)

    model = add_covariates(model, input_cov, num_covariates, regression, negative_values_ytrain, mean_ytrain, l1_value, L1_act)

    output_layer = activation_layer(model, regression, negative_values_ytrain)
    model = K.Model(inputs=[input_layer, input_cov], outputs=output_layer)

    print(model.summary())

    return model, masks



def create_network_from_csv(datapath, 
                            inputsize, 
                            genotype_path,
                            l1_value=0.01, 
                            L1_act =0.01, 
                            regression=False,
                            one_hot=False,
                            num_covariates=0):
    
    print("Creating networks from npz masks")
    print("regression", regression)
    if regression:
        mean_ytrain, negative_values_ytrain = regression_properties(datapath)
        print('mean_ytrain',mean_ytrain)
        print('negative_values_ytrain',negative_values_ytrain)
    else:
        mean_ytrain = 0
        negative_values_ytrain = False
        
        
    masks = []
    
    network_csv = pd.read_csv(datapath + "/topology.csv")
    network_csv = network_csv.filter(like="node", axis=1)
    columns = list(network_csv.columns.values)
    network_csv = network_csv.sort_values(by=columns, ascending=True)

    input_cov = K.Input((num_covariates,), name='inputs_cov')
    
    if one_hot:
        input_layer = K.Input((inputsize, 3), name='input_layer')
        model = one_hot_input(input_layer)
    else:
        input_layer = K.Input((inputsize,), name='input_layer')
        model = K.layers.Reshape(input_shape=(inputsize,), target_shape=(inputsize, 1))(input_layer)

    for i in range(len(columns) - 1):
        matrix_ones = np.ones(len(network_csv[[columns[i], columns[i + 1]]]), bool)
        matrix_coord = (network_csv[columns[i]].values, network_csv[columns[i + 1]].values)
        if i == 0:
            matrixshape = (inputsize, network_csv[columns[i + 1]].max() + 1)
        else:
            matrixshape = (network_csv[columns[i]].max() + 1, network_csv[columns[i + 1]].max() + 1)
        mask = scipy.sparse.coo_matrix(((matrix_ones), matrix_coord), shape = matrixshape)
        masks.append(mask)
        model = layer_block(model, mask, i, regression, L1_act=L1_act)

    model = K.layers.Flatten()(model)

    model = K.layers.Dense(units=1, name="output_layer",
                           kernel_regularizer=tf.keras.regularizers.l1(l=l1_value),
                           activity_regularizer=K.regularizers.l1(L1_act),
                           bias_initializer= tf.keras.initializers.Constant(mean_ytrain))(model)
    
    model = add_covariates(model, input_cov, num_covariates, regression, negative_values_ytrain, mean_ytrain, l1_value, L1_act)
    
    output_layer = activation_layer(model, regression, negative_values_ytrain)
   

    model = K.Model(inputs=[input_layer, input_cov], outputs=output_layer)

    print(model.summary())

    return model, masks


def lasso(inputsize, l1_value, num_covariates=0, regression=False):
    masks=[]
    inputs = K.Input((inputsize,), name='inputs')
    input_cov = K.Input((num_covariates,), name='inputs_cov')
    model = K.layers.BatchNormalization(center=False, scale=False, name="inter_out")(inputs)
    model = K.layers.Dense(units=1, kernel_regularizer=K.regularizers.l1(l1_value))(model)
    
    model = add_covariates(model, input_cov, num_covariates, regression, negative_values_ytrain, mean_ytrain, l1_value, L1_act)
    
    output_layer = K.layers.Activation("sigmoid")(model)
    
    model = K.Model(inputs=[inputs, input_cov], outputs=output_layer)
    return model, masks


def sparse_directed_gene_l1(datapath, inputsize, l1_value=0.01, L1_act=0.01, one_hot=False):
    
    
    for npz_path in glob.glob(datapath + '/*.npz'):
        mask = scipy.sparse.load_npz(npz_path)
    masks = [mask]     
    
    
    if one_hot:
        input_layer = K.Input((inputsize, 3), name='input_layer')
        model = one_hot_input(input_layer)
    else:
        input_layer = K.Input((inputsize,), name='input_layer')
        model = K.layers.Reshape(input_shape=(inputsize,), target_shape=(inputsize, 1))(input_layer)
    


    x1_1 = LocallyDirected1D(mask=mask, filters=1, input_shape=(inputsize, 1)
                                    , name="gene_layer", activity_regularizer=K.regularizers.l1(L1_act) )(model)
    x1_out = K.layers.Flatten()(x1_1)
    x1_out = K.layers.Activation("tanh")(x1_out)
    x1_out = K.layers.BatchNormalization(center=False, scale=False, name="inter_out")(x1_out)

    x9 = K.layers.Dense(units=1,kernel_regularizer=K.regularizers.l1(l1_value),  
                        activity_regularizer=K.regularizers.l1(L1_act))(x1_out)
    x9 = K.layers.Activation("sigmoid")(x9)


    model = K.Model(inputs=input_layer, outputs=x9)
    
    print(model.summary())
    
    return model, masks


def gene_network_multiple_filters(datapath, 
                                  inputsize,
                                  genotype_path, 
                                  l1_value=0.01,
                                  L1_act =0.01,
                                  regression=False,
                                  num_covariates=0, 
                                  filters=2,
                                  one_hot=False):
    
    print("Creating networks from npz masks")
    print("regression", regression)
    if regression:
        mean_ytrain, negative_values_ytrain = regression_properties(datapath)
        print('mean_ytrain',mean_ytrain)
        print('negative_values_ytrain',negative_values_ytrain)
    else:
        mean_ytrain = 0
        negative_values_ytrain = False
        
    print("height_multiple_filters with", filters, "filters")
    
    masks = []
    for npz_path in glob.glob(datapath + '/*.npz'):
        mask = scipy.sparse.load_npz(npz_path)
        masks.append(mask)
    if len(masks) == 0:
        print("You need an npz mask to run this network. Convert topology.csv to a mask.npz") 
        exit()
    if len(masks) > 1:
        print("multiple masks found")
    
    input_cov = K.Input((num_covariates,), name='inputs_cov')
    
    if one_hot:
        input_layer = K.Input((inputsize, 3), name='input_layer')
        model = one_hot_input(input_layer)
    else:
        input_layer = K.Input((inputsize,), name='input_layer')
        model = K.layers.Reshape(input_shape=(inputsize,), target_shape=(inputsize, 1))(input_layer)
    
    model = LocallyDirected1D(mask=mask, filters=filters, input_shape=(inputsize, 1), name="gene_layer")(model)
    model = K.layers.Activation("relu")(model)
    model = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_g1")(model)
    
    model = K.layers.LocallyConnected1D(filters=1, strides=1, kernel_size=1, implementation=3)(model)
    model = K.layers.Activation("relu")(model)
    model = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_g2")(model)
    
    model = K.layers.Flatten()(model)
    model = K.layers.Dense(units=1, name="output_layer",
                           kernel_regularizer=tf.keras.regularizers.l1(l=l1_value), 
                           bias_initializer= tf.keras.initializers.Constant(mean_ytrain))(model)
    
    model = add_covariates(model, input_cov, num_covariates, regression, 
                           negative_values_ytrain, mean_ytrain, l1_value, L1_act)
    
    output_layer = activation_layer(model, regression, negative_values_ytrain)
   

    model = K.Model(inputs=[input_layer, input_cov], outputs=output_layer)

    print(model.summary())
    return model, masks


def gene_network_snp_gene_filters(datapath, 
                                  inputsize,
                                  genotype_path, 
                                  l1_value=0.01, 
                                  L1_act =0.01,
                                  regression=False,
                                  num_covariates=0, 
                                  filters=2,
                                  one_hot=False):
    
    print("Creating networks from npz masks")
    print("regression", regression)
    if regression:
        mean_ytrain, negative_values_ytrain = regression_properties(datapath)
        print('mean_ytrain',mean_ytrain)
        print('negative_values_ytrain',negative_values_ytrain)
    else:
        mean_ytrain = 0
        negative_values_ytrain = False
        
    print("height_multiple_filters with", filters, "filters")
    
    masks = []
    for npz_path in glob.glob(datapath + '/*.npz'):
        mask = scipy.sparse.load_npz(npz_path)
        masks.append(mask)
    if len(masks) == 0:
        print("You need an npz mask to run this network. Convert topology.csv to a mask.npz")
        exit()
    
    if len(masks) > 1:
        print("multiple masks found")
    
    input_cov = K.Input((num_covariates,), name='inputs_cov')
    
    if one_hot:
        input_layer = K.Input((inputsize, 3), name='input_layer')
        model = one_hot_input(input_layer)
    else:
        input_layer = K.Input((inputsize,), name='input_layer')
        model = K.layers.Reshape(input_shape=(inputsize,), target_shape=(inputsize, 1))(input_layer)
    
    
    model = LocallyDirected1D(mask=mask, filters=filters, input_shape=(inputsize, 1), name="gene_layer")(model)
    model = K.layers.Activation("relu")(model)
    model = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_g1")(model)
    
    model = K.layers.LocallyConnected1D(filters=filters, strides=1, kernel_size=1, implementation=3)(model)
    model = K.layers.Activation("relu")(model)
    model = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_g2")(model)
    
    model = K.layers.LocallyConnected1D(filters=1, strides=1, kernel_size=1, implementation=3)(model)
    model = K.layers.Activation("relu")(model)
    model = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_g3")(model)
    
    model = K.layers.Flatten()(model)
    model = K.layers.Dense(units=1, name="output_layer",
                           kernel_regularizer=tf.keras.regularizers.l1(l=l1_value), 
                           bias_initializer= tf.keras.initializers.Constant(mean_ytrain))(model)
    
    model = add_covariates(model, input_cov, num_covariates, regression, negative_values_ytrain, mean_ytrain, l1_value, L1_act)
    
    output_layer = activation_layer(model, regression, negative_values_ytrain)
   

    model = K.Model(inputs=[input_layer, input_cov], outputs=output_layer)

    print(model.summary())
    return model, masks



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
    


def remove_batchnorm_model(model, masks):
    original_model = model
    inputs = tf.keras.Input(shape=original_model.input_shape[0][1:])
    x = inputs

    mask_num = 0
    for layer in original_model.layers[1:]: 
        # Skip BatchNormalization layers
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            # Handle LocallyDirected1D layer with custom arguments
            if isinstance(layer, LocallyDirected1D):
                config = layer.get_config()
                new_layer = LocallyDirected1D(filters=config['filters'], 
                                                mask=masks[mask_num],
                                                name=config['name'])
                x = new_layer(x)
                mask_num = mask_num + 1
            else:
                # Add other layers as they are
                x = layer.__class__.from_config(layer.get_config())(x)

    # Create the new model
    new_model = tf.keras.Model(inputs=inputs, outputs=x)

    original_model_layers = [x for x in original_model.layers if not isinstance(x, tf.keras.layers.BatchNormalization)]

    for new_layer, layer in zip(new_model.layers, original_model_layers): 
        new_layer.set_weights(layer.get_weights())
        
    return new_model
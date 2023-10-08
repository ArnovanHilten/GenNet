import statsmodels.api as sm
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
import tables
import tensorflow.keras as K
import sklearn.metrics as skm
import scipy
import sys
import argparse
import time
import tqdm
import glob
import scipy
import scipy.sparse

from collections import defaultdict


def calculate_RLIPP_complete(model, xtrain, ytrain,input_regression_path, child_layer=None, masks=None, verbose = False):
 
    '''returns 
    
    1) new RLIPP: pd_dataframe with columns: [parent_layer_name	parent_layer	child_node	adjR_squar_parent	adjR_squar_child	RLIPP	parent_node	abs(RLIPP)]
    2) old RLIPP: pd_dataframe with columns: ['layer_name', 'layer_num', 'parent_node', 'child_node' 'RLIPP']
    3) child_dic: a dictionary with all the connections between parents and children 
    
    mode: tf model
    child_layer: layer to calculate RLIPP for
    masks: list of sparse coo_matrix masks
    '''
    pd_data_rlipp_old, child_dic =  calculate_RLIPP(model, xtrain, ytrain,input_regression_path, masks=masks)
    pd_data_rlipp_novel =  calculate_RLIPP_novel(model, xtrain, ytrain,input_regression_path,child_dic,  masks=masks)
    return pd_data_rlipp_novel, pd_data_rlipp_old, child_dic# , pd_novel

    
def calculate_RLIPP(model, xtrain, ytrain,input_regression_path, child_layer=None, masks=None ):
    
    '''returns pd_dataframe with columns: ['layer_name', 'layer_num', 'parent_node', 'child_node' 'RLIPP']
    
    mode: tf model
    child_layer: layer to calculate RLIPP for
    masks: list of sparse coo_matrix masks
    '''

    child_dict = {}
    initial = None
    RLIPP_func = None
    pd_current = pd.DataFrame([], columns= ['parent_layer_name','parent_layer', 'child_node', 'RLIPP'])
    
    
    if child_layer is None: 
        layer_names = [x.name for x in model.layers if "activation" in x.name]
        layer_names
        assert len([x.__class__.__name__ for x in model.layers if "LocallyDirected" in x.__class__.__name__]) == len(masks)


    model  = get_node_correlation(model, xtrain, ytrain, layer_names=None)
    model   = get_input_correlation(model, xtrain, ytrain, input_regression_path) # only once per simulation
  

    for layer_num, layer in enumerate(model.layers):

        layer_type = model.layers[layer_num].__class__.__name__ 

        if ("activation" in layer.name) and ("snp" not in layer.name):
            pd_current, child_dict = RLIPP_func(model, layer_num, pd_current, masks, child_dict)
        elif "LocallyDirected1D" in layer_type:  
            print("Directed1D")
            RLIPP_func = get_RLIPP_LD
        elif "LocallyConnected1D" in layer_type :
            print("Connected1D")
            RLIPP_func = get_RLIPP_LC
        elif "Dense" in layer_type :
            print("Dense")
            RLIPP_func = get_RLIPP_dense
        else:
            print(" -- Skipping", layer_type, '--') 
          
    return pd_current, child_dict# , pd_novel
        

def calculate_RLIPP_novel(model, xtrain, ytrain,input_regression_path,child_dic, child_layer=None, masks=None, verbose = False ):
    model  = get_node_correlation_novel(model, xtrain, ytrain,child_dic,verbose, layer_names=None)

    pd_novel = pd.DataFrame([], columns= ['parent_layer_name','parent_layer', 'child_node','adjR_squar_parent', 'adjR_squar_child', 'RLIPP'])
    for layer_num, layer in enumerate(model.layers):
        layer_type = model.layers[layer_num].__class__.__name__
        if ("activation" in layer.name) and ("snp" not in layer.name):
            for node_num, el in enumerate(model.layers[layer_num].fprsq):
                child_prsqrt = model.layers[layer_num].fprsq[node_num]
                parent_prsqrt = model.layers[layer_num].prsqrt[node_num]
                RLIPP_score = float((parent_prsqrt - child_prsqrt) / child_prsqrt)
                if parent_prsqrt < 0:
                    RLIPP_score = 0.
                node_num_parent = node_num # -10 ## not removed the 10
                if verbose:
                    print("node_num ",  node_num_parent)
                pd_novel = pd_novel.append({"parent_layer":layer_num, 
                                    "parent_layer_name":model.layers[layer_num].name,
                                    "child_node":"NA",
                                    "parent_node":node_num_parent,
                                    "adjR_squar_parent": parent_prsqrt, 
                                    "adjR_squar_child": child_prsqrt,         
                                    "RLIPP":RLIPP_score}, ignore_index=True)
    return pd_novel


def get_input_correlation(model, xtrain, ytrain, input_regression_path):
    
    
    if os.path.exists(input_regression_path + '/input_correlation.npy') & False:
        model.layers[0].prsqrt = np.load(input_regression_path + '/input_correlation.npy')
        return model
  
    print("calculate input correlation")
    assert xtrain.ndim == 3, "xtrain needs to be 3 dimensional with (npat, nfeatures, 3 or 1)"
    
    prsqrt_input = np.full((xtrain.shape[1], xtrain.shape[2]), np.NaN)


    for SNP_num in tqdm.tqdm(range(xtrain.shape[1])):
        for allele in range(xtrain.shape[2]):
            try:
                regresult = sm.OLS(ytrain, xtrain[:,SNP_num, allele]).fit(disp=0)
                prsqrt_input[SNP_num, allele] =regresult.rsquared_adj
            except:
                prsqrt_input[SNP_num, allele] = np.NaN
    
    model.layers[0].prsqrt = prsqrt_input
    np.save(input_regression_path + '/input_correlation.npy', prsqrt_input)
                
    return model


def get_node_correlation(model, xtrain, ytrain, layer_names=None):
    for layer_num, layer in enumerate(model.layers):
        if 'activation' in layer.name:
            intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                             outputs=model.get_layer(layer.name).output)
            predictions = intermediate_layer_model.predict(np.squeeze(xtrain))
            predictions = np.reshape(predictions, (xtrain.shape[0], -1))
            prsqrt_node = np.full((predictions.shape[1]), np.NaN)
            
            for node in range(predictions.shape[1]):
                try:
                    if len(np.unique(predictions[:,node])) > 1:
                        regresult = sm.OLS(ytrain, predictions[:,node]).fit(disp=0)
                        prsqrt_node[node] = regresult.rsquared_adj
                    else:
                        prsqrt_node[node] = np.NaN
                except:
                    prsqrt_node[node] = np.NaN
         
        else:
            prsqrt_node = np.full((0), np.NaN)
            
        model.layers[layer_num].prsqrt = prsqrt_node
        
    return model

def get_node_correlation_novel(model, xtrain, ytrain, child_dic, verbose, layer_names=None):
    for layer_num, layer in enumerate(model.layers):
        if 'activation' in layer.name:
                intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                                 outputs=model.get_layer(layer.name).output)
                print(layer_num)
                try: 
                    previous_predictions = predictions
                except: 
                    if verbose:
                        print("1st")
                predictions = intermediate_layer_model.predict(np.squeeze(xtrain))
                pre_resh_shap = predictions.shape
                if verbose:
                    print("prediction = ", predictions.shape)
                predictions = np.reshape(predictions, (xtrain.shape[0], -1))


                prsqrt_node = np.full((predictions.shape[1]), np.NaN)
                prsqrt_node = np.full((predictions.shape[1]), np.NaN)
                fprsq_node  = np.full((predictions.shape[1]), np.NaN)
                for node in range(predictions.shape[1]):
                            if layer_num > 2: 
                                previous_predictions_ = np.reshape(previous_predictions, (xtrain.shape[0], -1))#
                                if verbose:
                                    print(previous_predictions_.shape) # (1000, 100)
                                    print(fprsq_node.shape)
                            try:
                                if len(np.unique(predictions[:,node])) > 1:
                                    regresult = sm.OLS(ytrain, predictions[:,node]).fit(disp=0)
                                    prsqrt_node[node] = regresult.rsquared_adj
                                    if layer_num > 2:  
                                        coef_flatter = (node // pre_resh_shap[1] )*pre_resh_shap[1]
                                        if verbose:
                                            print("coef_flatter", coef_flatter)                                    
                                        regresult_child = sm.OLS(ytrain, previous_predictions_[:, child_dic[layer_num][node-coef_flatter]]).fit(disp=0)
                                        #print(regresult_child.rsquared_adj)
                                        fprsq_node[node] = regresult_child.rsquared_adj
                                        if verbose:    
                                            print(fprsq_node)
                                       

                                    if verbose:
                                        print("layer_num", layer_num)
                                else:
                                    if verbose:
                                        print("else")
                                    prsqrt_node[node] = np.NaN
                                    fprsq_node[node] = np.NaN
                            except:
                                print("excep")
                                prsqrt_node[node] = np.NaN
                                fprsq_node[node] = np.NaN

        else:
            prsqrt_node = np.full((0), np.NaN)
            fprsq_node = np.full((0), np.NaN)

        model.layers[layer_num].prsqrt = prsqrt_node
        if layer_num > 2: 
            if verbose:
                print(fprsq_node)
            model.layers[layer_num].fprsq = fprsq_node
    return model

def get_child_nodes(model, layer_num):
    childeren_prsqrt = None
    
    for i in range(layer_num):
        if model.layers[i].prsqrt.size != 0:
            childeren_prsqrt = model.layers[i].prsqrt   
    return childeren_prsqrt


def get_RLIPP(model, pd_RLIPP, parent_prsqrt, child_prsqrt, node_num_child, node_num_parent, layer_num):
    RLIPP_score = float((parent_prsqrt - child_prsqrt) / child_prsqrt)
    if parent_prsqrt < 0:
        RLIPP_score = 0.
    pd_RLIPP = pd_RLIPP.append({"parent_layer":layer_num, 
                            "parent_layer_name":model.layers[layer_num].name,
                            "child_node":node_num_child,
                            "parent_node":node_num_parent,
                            "RLIPP":RLIPP_score}, ignore_index=True)
    return pd_RLIPP



def get_RLIPP_LC(model, layer_num, pd_RLIPP, masks, child_dict):
    parent_prsqrt = model.layers[layer_num].prsqrt
    childeren_prsqrt = get_child_nodes(model, layer_num)
    child_index = np.arange(len(childeren_prsqrt))
    
    kernelsize = int(len(childeren_prsqrt) / len(parent_prsqrt))
    
    childeren_prsqrt = np.reshape(childeren_prsqrt, (-1, kernelsize))
    
    child_dict[layer_num] = {}  
    child_index = np.reshape(child_index, (-1, kernelsize))
    

    # calculate the RLIPP for the right parent-child
    nodenum_child = 0
    for node_num, child_prsqrts in enumerate(childeren_prsqrt):
        child_dict[layer_num][node_num] = child_index[node_num]
        for child_prsqrt in child_prsqrts:
            
            pd_RLIPP = get_RLIPP(model, pd_RLIPP, parent_prsqrt[node_num], child_prsqrt, 
                                 node_num_child = nodenum_child, node_num_parent = node_num, layer_num = layer_num)  
            nodenum_child = nodenum_child + 1
        
    
    return pd_RLIPP, child_dict


def get_RLIPP_LD(model, layer_num, pd_RLIPP, masks, child_dict):
    parent_prsqrt = model.layers[layer_num].prsqrt
    childeren_prsqrt = get_child_nodes(model, layer_num)
    
    child_dict[layer_num] = {}                                          # to change


    num_mask = -1
    for i in range(layer_num):
        if "LocallyDirected" in model.layers[i].__class__.__name__:
            num_mask +=1

    mask = masks[num_mask]

    nfilters = int(len(parent_prsqrt) / mask.shape[1])
    parent_prsqrt = np.reshape(parent_prsqrt, (mask.shape[1],nfilters))

    nodenum_child = 0
    for cfilter in range(nfilters):                       # per filter
        cparent_prsqrt =  parent_prsqrt[:,cfilter]        # select the parent node
        for node_num in range(mask.shape[1]):            # per gene
            children = mask.row[mask.col==node_num]      # select children
            child_dict[layer_num][node_num] = children
            for child_prsqrt in childeren_prsqrt[children]:  # per child
                pd_RLIPP = get_RLIPP(model, pd_RLIPP, cparent_prsqrt[node_num], child_prsqrt, 
                                     node_num_child = nodenum_child, node_num_parent = node_num, layer_num = layer_num)
                nodenum_child = nodenum_child + 1

    return pd_RLIPP, child_dict



def get_RLIPP_dense(model, layer_num, pd_RLIPP, masks, child_dict):
    parent_prsqrt = model.layers[layer_num].prsqrt
    childeren_prsqrt = get_child_nodes(model, layer_num)
    
    child_dict[layer_num] = {}
    child_dict[layer_num][0] = list(np.arange(len(childeren_prsqrt)))
    
    # calculate the RLIPP for the right parent-child
    for node_num, child_prsqrt in enumerate(childeren_prsqrt):
        pd_RLIPP = get_RLIPP(model, pd_RLIPP, parent_prsqrt, child_prsqrt, 
                             node_num_child = node_num, node_num_parent= 0, layer_num=layer_num)

    return pd_RLIPP, child_dict
    
    
    

def get_RLIPP_dense(model, layer_num, pd_RLIPP, masks, child_dict):
    parent_prsqrt = model.layers[layer_num].prsqrt
    childeren_prsqrt = get_child_nodes(model, layer_num)
    
    child_dict[layer_num] = {}
    child_dict[layer_num][0] = list(np.arange(len(childeren_prsqrt)))
    
    # calculate the RLIPP for the right parent-child
    for node_num, child_prsqrt in enumerate(childeren_prsqrt):
        pd_RLIPP = get_RLIPP(model, pd_RLIPP, parent_prsqrt, child_prsqrt, 
                             node_num_child = node_num, node_num_parent= 0, layer_num=layer_num)

    return pd_RLIPP, child_dict




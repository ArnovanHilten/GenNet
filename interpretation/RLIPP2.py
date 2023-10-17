import statsmodels.api as sm
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
import tensorflow.keras as K
import scipy
import sys
import argparse
import time
import tqdm
import scipy
import scipy.sparse

from collections import defaultdict

class RLIPP:
    def __init__(self, model, xtrain, ytrain, child_layer=None, masks=None, input_regression_path='/'):
        self.model = model
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.child_layer = child_layer
        self.masks = masks
        self.input_regression_path = input_regression_path
        self.layer_names = None
        self.pd_RLIPP = pd.DataFrame([], columns= ['layer_name','layer', 'node', 'RLIPP'])

    def calculate_modelwise(self):
        if self.child_layer is None:
            self.layer_names = [x.name for x in self.model.layers if "activation" in x.name]
            assert len([x.__class__.__name__ for x in self.model.layers if "LocallyDirected" in x.__class__.__name__]) == len(self.masks)
        
        self.model  = self.get_node_correlation()
        self.model  = self.get_input_correlation()

        for layer_num, layer in enumerate(self.model.layers):
            layer_type = self.model.layers[layer_num].__class__.__name__ 

            if ("activation" in layer.name) and ("snp" not in layer.name):
                self.RLIPP(layer_num)
            elif "LocallyDirected1D" in layer_type:
                self.get_RLIPP_LD
            elif "LocallyConnected1D" in layer_type :
                self.get_RLIPP_LC
            elif "Dense" in layer_type :
                self.get_RLIPP_dense
            else:
                print(" -- Skipping", layer_type, '--') 

        return self.pd_RLIPP
    
    def get_input_correlation(self):
        assert self.xtrain.ndim == 3, "xtrain needs to be 3 dimensional with (npat, nfeatures, 3 or 1)"
        prsqrt_input = np.full((self.xtrain.shape[1], self.xtrain.shape[2]), np.NaN)

        for SNP_num in tqdm.tqdm(range(self.xtrain.shape[1])):
            for allele in range(self.xtrain.shape[2]):
                try:
                    regresult = sm.Logit(self.ytrain, self.xtrain[:,SNP_num, allele]).fit(disp=0)
                    prsqrt_input[SNP_num, allele] = regresult.prsquared
                except:
                    prsqrt_input[SNP_num, allele] = np.NaN

        self.model.layers[0].prsqrt = prsqrt_input
        return self.model
 
    
    def get_node_correlation(self):
        
        for layer_num, layer in enumerate(self.model.layers):
            if 'activation' in layer.name:

                intermediate_layer_model = tf.keras.Model(inputs=self.model.input,
                                                 outputs=self.model.get_layer(layer.name).output)
                predictions = intermediate_layer_model.predict(self.xtrain)
                predictions = np.reshape(predictions, (self.xtrain.shape[0], -1))
                prsqrt_node = np.full((predictions.shape[1]), np.NaN)

                for node in range(predictions.shape[1]):
                    if len(np.unique(predictions[:,node])) > 1:
                        regresult = sm.Logit(self.ytrain, predictions[:,node]).fit(disp=0)
                        prsqrt_node[node] = regresult.prsquared
                    else:
                        prsqrt_node[node] = np.NaN
            else:
                prsqrt_node = np.full((0), np.NaN)

            self.layers[layer_num].prsqrt = prsqrt_node
            
            
    def get_RLIPP(self, parent_prsqrt, child_prsqrt, node_num, layer_num):
        RLIPP_score = float((parent_prsqrt - child_prsqrt) / child_prsqrt)

        self.pd_RLIPP.append({"layer":layer_num, 
                                "layer_name":self.model.layers[layer_num].name,
                                "node":node_num,
                                "RLIPP":RLIPP_score}, ignore_index=True)    
         
            
    def get_RLIPP_LC(self, layer_num,):
        parent_prsqrt = self.model.layers[layer_num].prsqrt
        childeren_prsqrt = self.get_child_nodes(self.model, layer_num)

        kernelsize = int(len(childeren_prsqrt) / len(parent_prsqrt))

        childeren_prsqrt = np.reshape(childeren_prsqrt, (-1, kernelsize))

        # calculate the RLIPP for the right parent-child
        for node_num, child_prsqrts in enumerate(childeren_prsqrt):
            for child_prsqrt in child_prsqrts:
                  self.get_RLIPP(parent_prsqrt[node_num], child_prsqrt, node_num, layer_num)


    def get_RLIPP_LD(self, layer_num):
        parent_prsqrt = self.model.layers[layer_num].prsqrt
        childeren_prsqrt = self.get_child_nodes(self.model, layer_num)

        num_mask = -1
        for i in range(layer_num):
            if "LocallyDirected" in self.model.layers[i].__class__.__name__:
                num_mask +=1

        mask = self.masks[num_mask]

        nfilters = int(len(parent_prsqrt) / mask.shape[1])
        parent_prsqrt = np.reshape(parent_prsqrt, (mask.shape[1], nfilters))
        
        for cfilter in range(nfilters):                       # per filter
            cparent_prsqrt =  parent_prsqrt[:,cfilter]        # select the parent node
            for node_num in range(mask.shape[1]):            # per gene
                children = mask.row[mask.col==node_num]      # select children
                for child_prsqrt in childeren_prsqrt[children]:  # per child
                    self.get_RLIPP(cparent_prsqrt[node_num], child_prsqrt, node_num, layer_num)    




    def get_RLIPP_dense(self, layer_num):
        parent_prsqrt = self.model.layers[layer_num].prsqrt
        childeren_prsqrt = self.get_child_nodes(self.model, layer_num)

        # calculate the RLIPP for the right parent-child
        for node_num, child_prsqrt in enumerate(childeren_prsqrt):
            self.get_RLIPP(parent_prsqrt, child_prsqrt, node_num, layer_num)


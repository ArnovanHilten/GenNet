import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as skm
import scipy
import sys
import pandas as pd
import numpy as np
    
# import numba as nb
import itertools

from scipy.sparse import coo_matrix
            
            
def GenNet_pairwise_interactions_topn(w_input, w_later, mask, n):
    '''
    NID fast implementation for topn with:
    
    n: number of interactions per gene
    n cannot be a larger number than the smallest gene size
    '''

    mask_row = np.array(mask.row)
    mask_col = np.array(mask.col)
    mask_data = abs(np.array(w_input))
    
    n_genes = int(mask.shape[1])

    if n=="auto":
       n=10e6
    
    if (min(mask.sum(axis=0)) < n).any():  # Cannot get the top n if genes have less than n
        n = np.min(mask.sum(axis=0))

    num_combinations = int(np.round(np.math.factorial(n) / (np.math.factorial(2) * np.math.factorial((n - 2))) ))

    SNP_coord, strength = get_p_interactions(w_input, w_later, num_combinations, mask_row, mask_col, mask_data, n_genes, n)

    strength, ix = min_combination_rows(strength)  # take the min values of all the combinations
    strength = np.multiply(strength, w_later.flatten())  # multiply those with the w_rest
    
    
    # matrix of the combinations of SNP coord                                                                    
    df1 = pd.DataFrame(SNP_coord.astype(int).astype(str))  
    df1 = pd.DataFrame(df1.values + " ")      
    df1 = pd.DataFrame(df1.values[ix].sum(axis=1))
    SNP_coord = df1.values

    ind = np.argsort(strength, axis=0)[::-1]              # order these by value, descending so inverse
    strength = np.take_along_axis(strength, ind, axis=0)                  # apply ordering strength
    SNP_coord = np.take_along_axis(SNP_coord, ind, axis=0)  

    interaction_ranking = pd.DataFrame(np.array([SNP_coord.flatten(), strength.flatten()]).T, columns=["coord","strength"])
    interaction_ranking = interaction_ranking.sort_values("strength", ascending=False)

    return interaction_ranking  


def get_p_interactions(w_input, w_later,num_combinations, mask_row, mask_col, mask_data, n_genes, n):
    SNP_coord = np.zeros(shape=(n, n_genes)) # all snp numbers, col is gene number
    strength = np.zeros(shape=(n , n_genes)) # 1 value for each SNPcoord
    for gene_id in range(n_genes): # nb.range()  # over all genes
        top_coord = np.argsort(-mask_data[mask_col == gene_id])[:n] # select the highest n values of the gene
        SNP_coord[:, gene_id] = mask_row[mask_col == gene_id][top_coord]    #  select the coordinate of the highest value
        strength[:,gene_id]  = mask_data[mask_col == gene_id][top_coord]    #  select the weights ofthe highest value
    return SNP_coord, strength

def min_combination_rows(comb_array):
    n = comb_array.shape[0]
    ix = np.indices((n,n))[:, ~np.tri(n, k=0, dtype=bool)].T
    minimum_array = comb_array[ix].min(axis=1)
    return minimum_array, ix 

def Get_weight_tsang(model, layer_n, masks):
    '''
    Get the proper w_in and w_out for the NID algorithm.
    
    For multiple layers:
    w_in, w_out = Get_weight_tsang(model, 2, genemask)
    print(w_in.shape, w_out.shape)
    
    print('causal_snps', causal_snps)
    interaction_ranking1 = GenNet_pairwise_interactions_topn(w_in[:,1] ,w_out[:,1], genemask, n=4)
    interaction_ranking2 = GenNet_pairwise_interactions_topn(w_in[:,0] ,w_out[:,0], genemask, n=4)

    interaction_ranking = interaction_ranking1.append(interaction_ranking2)
    interaction_ranking = interaction_ranking.sort_values("strength", ascending =False)
    interaction_ranking.head(20)
    '''
    w_in_layer = model.layers[layer_n]
    w_in = abs(w_in_layer.get_weights()[0])
    print("From", w_in_layer.__class__.__name__, w_in.shape)
    mask_count = 0   
    w_out = 0

    for i in range(len(model.layers)-1, -1, -1):
        if layer_n == i:
            return abs(w_in), abs(w_out)

        layer = model.layers[i]
        layer_name = model.layers[i].name
        layer_type = model.layers[i].__class__.__name__
        print(i, layer_type)

        if "LocallyDirected1D" in layer_type:
            mask = masks[mask_count]
            weights = layer.get_weights()[0]
            w_filter = []

            for j in range(weights.shape[1]):
                mask.values = abs(weights[:,j])
                w_filter.append(np.matmul(mask.todense(), w_out)) 

            w_out = np.concatenate(w_filter, axis=-1) 


            names = ['weights_'+str(i)+ "_"+str(j) for j in range(weights.ndim)]

            mask_count +=1 

        elif "LocallyConnected1D" in layer_type:
            weights = layer.get_weights()[0]
            w_filter = []
            if weights.ndim == 3:
                for j in range(weights.shape[1]):
                    w_filter.append(w_out * weights[:,j])  
                w_out = np.concatenate(w_filter, axis=-1) 
            else:
                print("unexpected shape", weights.shape)

        elif "Dense" in layer_type and "_cov" not in layer_name:
            weights = abs(layer.get_weights()[0])
            if w_out == 0:
                w_out = abs(weights)   
            else:
                w_out = np.matmul(w_out, np.abs(weights))
        else:
            print(" -- Skipping", layer_type, '--')
    
    return w_in, w_out 


import itertools
def GenNet_pairwise_interactions_simplified_mask(w_input, w_later, mask):
    """first simple implementation"""
    mask = mask * 1
    interaction_ranking = []
    list_of_combinations = []
    for gene_id in range(mask.shape[1]):
        neuron_combinations=list(itertools.combinations(mask.row[mask.col == gene_id], 2 ))

    for candidate in neuron_combinations:
        strength = (np.minimum(w_input[candidate[0]], w_input[candidate[1]])*w_later[gene_id]).sum()
        interaction_ranking.append(((candidate[0], candidate[1]), strength))

    interaction_ranking.sort(key=lambda x: x[1], reverse=True)
    return interaction_ranking





    
    
            

    
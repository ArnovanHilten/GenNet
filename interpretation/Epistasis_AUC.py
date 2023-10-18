import pandas as pd
import numpy as np
import glob
import itertools
import sys
import os
import time
import tqdm
import sklearn.metrics as skm


def get_auc_epistasis(true_label, predicted_label,  threshold=100, verbose=True):
       
    
    predicted_label = predicted_label.flatten()[:threshold]
    true_label = true_label.flatten()[:threshold]
    
   
    if (threshold > len(predicted_label) ) | (threshold > len(true_label) ):
        if verbose:
            print("WARNING: Threshold is too high, all labels are considered")
    
      
    if (len(np.unique(predicted_label)) < 2 ) | (len(np.unique(true_label)) < 2):
        if verbose:
            print("Error: all labels are the same")
        return 0
   
    fpr, tpr, thresholds = skm.roc_curve(true_label, predicted_label)
    roc_auc = skm.auc(fpr, tpr)
    
    return roc_auc


def get_GenNet_Gametes_epistasis_AUC(pd_overview, threshold=100, verbose=True):
    """pd_overview: pd dataframe with columns: rfrunpath, num_snps"""
    pd_interact = pd.DataFrame([])
    auc= np.zeros(pd_overview.shape[0])
    
    for i in tqdm.tqdm(range(len(pd_overview))):
        try:
            folder = pd_overview["rfrunpath"].iloc[i] +  "/NID.csv"
            pd_interact  = pd.read_csv(folder, names = ['interaction','strength'], skiprows=1)
            pd_interact[['X_1', 'X_2']] = pd_interact['interaction'].str.rstrip(' ').str.split(" ", expand=True)
            pd_interact["X_1"] = pd.to_numeric(pd_interact["X_1"])
            pd_interact["X_2"] = pd.to_numeric(pd_interact["X_2"])
            pd_interact["groundtruth"] = np.zeros(len(pd_interact), int)

            causal_snps = [pd_overview["num_snps"].iloc[i]-1, pd_overview["num_snps"].iloc[i]-2]
            pd_interact.loc[(pd_interact['X_1'].isin(causal_snps)) & (pd_interact['X_2'].isin(causal_snps)) &
                            (pd_interact['X_2'] != pd_interact['X_1']), "groundtruth"]=1

            pd_interact =  pd_interact.sort_values("strength", ascending=False)
            auc[i] = get_auc_epistasis(pd_interact["groundtruth"].values,pd_interact["strength"].values, 
                                       threshold=threshold, verbose=verbose)
        except:
            print(pd_overview["simu_name"].iloc[i])
            auc[i] = -1    

    pd_overview["NID eAUC"] = auc
    return pd_overview
    
    
def get_LGBM_Gametes_epistasis_AUC(pd_overview, threshold=100, verbose=True):
    """pd_overview: pd dataframe with columns: rfrunpath, num_snps"""
    pd_interact = pd.DataFrame([])
    auc= np.zeros(pd_overview.shape[0])
    
    for i in tqdm.tqdm(range(len(pd_overview))):
        try:
            pd_interact  = pd.read_csv("./Gametes_output_2D/"+pd_overview["simu_name"].iloc[i]+".csv", index_col = 0)
            pd_interact["groundtruth"] = np.zeros(len(pd_interact), int)
            causal_snps = ["M0P0","M0P1"]
            pd_interact.loc[(pd_interact['SNP1'].isin(causal_snps)) &
                            (pd_interact['SNP2'].isin(causal_snps)) & 
                            (pd_interact['SNP2'] != pd_interact['SNP1']), "groundtruth"]=1
            pd_interact =  pd_interact.sort_values("strength", ascending=False)
            auc[i] = get_auc_epistasis(pd_interact["groundtruth"].values, pd_interact["strength"].values, 
                                       threshold=threshold, verbose=verbose)

        except:
            print(pd_overview["simu_name"].iloc[i])
            auc[i] = -1    

    pd_overview["NID eAUC"] = auc
    return pd_overview
    


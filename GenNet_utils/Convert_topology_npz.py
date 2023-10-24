import argparse
import numpy as np
import pandas as pd
import sys
import os
import scipy
import h5py
import tables
from scipy import stats

def main():
    """
    args: 
    snp: the name of the column in the topology.csv dataset with the ID for the SNP column
    gene: the name of the column in the topology.csv dataset with the ID for the gene column
    direc: (Optional) the directory where the topology.csv file is located, if omitted it takes the current directory

    Return: SNP_gene_mask.npz, the .npz file corresponding to the topology.csv
    """
    parser = argparse.ArgumentParser(description="A simple script with command-line arguments")
    parser.add_argument("--snp", help="Your snp", required=True)
    parser.add_argument("--gene", help="Your gene", required=True)
    parser.add_argument("--direc", help="Your Directory", required=False)
    args = parser.parse_args()
    if args.direc:
        try:
            os.chdir(args.direc)
            print(f"Navigated to directory: {os.getcwd()}")
        except FileNotFoundError:
            print(f"Directory '{args.direc}' not found.")
    
    snp_level = args.snp #"layer0_node"
    gene_level = args.gene #"layer1_node"
    topology = pd.read_csv("topology.csv")
    data = np.ones(len(topology), np.bool)
    coord = ( topology[snp_level].values, topology[gene_level].values )
    SNP_gene_matrix = scipy.sparse.coo_matrix(((data),coord),  shape = (topology[snp_level].max()+1, topology[gene_level].max()+1 ))
    scipy.sparse.save_npz('SNP_gene_mask', SNP_gene_matrix)

if __name__ == "__main__":
    main()



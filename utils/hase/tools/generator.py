import sys
import os
import argparse
import numpy as np




parser = argparse.ArgumentParser(description='Script to generate data')

parser.add_argument('-type', type=str,required=True, choices=['phenotype'],
                    help=' Choose what kind of data do you want to generate')


parser.add_argument("-s", type=int, help="number of subject")
parser.add_argument("-ph", type=int, help="number of phenotypes")
parser.add_argument("-o", "--out", type=str, required=True, help="path to save result folder")
parser.add_argument("-save_name", type=str, required=True, help="file name to save")

args = parser.parse_args()
print(args)

def random_phenotype(Ns,Np):
    pass


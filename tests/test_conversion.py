import os
import sys

# os.chdir('../GenNet/')

# print(os.getcwd())
# sys.path.insert(1, os.getcwd())
# # sys.path.insert(1, os.getcwd() + "/GenNet_utils/")

import pytest
import pandas as pd
import shutil
from os.path import dirname, abspath
import argparse

print(sys.path)
from GenNet_utils.Create_plots import plot
# from GenNet_utils.Train_network import train_model
from GenNet_utils.Convert import convert
from GenNet_utils.Topology import topology


class ArgparseSimulatorConvert():
    def __init__(self,
                 mode='/',
                 genotype=['../examples/plink2/'],
                 study_name=['toy_data'],
                 outfolder="processed_data/",
                 step = "all"):
        self.mode = mode
        self.genotype = genotype
        self.study_name = study_name
        self.out = outfolder
        self.step = step
        self.vcf = False
        self.variants = None
        self.tcm = 500000000
        self.n_jobs = 1
        self.comp_level = 1


args = ArgparseSimulatorConvert()

convert(args)

        
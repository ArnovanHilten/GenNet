import os
import sys
import pandas as pd
import shutil
from os.path import dirname, abspath
import os
sys.path.insert(1, os.getcwd())
from GenNet_utils.Interpret import interpret

# TODO: add test without covariates
# TODO add test with covariates for regression + classification
# TODO add test with multiple genotype files.
# test randomnesss after .. epoch shuffles.
# ToDO add test for each file.

class ArgparseSimulator():
    def __init__(self,
                 resultpath = "/trinity/home/avanhilten/repositories/epistasis/prototyping/GenNet_realLife_data/GenNet/results/GenNet_experiment_2_/",
                 type = "NID",
                 layer = None
                 ):

        self.resultpath = resultpath
        self.type = type
        self.layer = layer
        

args = ArgparseSimulator()
print("done")

interpret(args)
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import numpy as np
import pandas as pd
import tables
import tensorflow.keras as K

matplotlib.use('agg')


def check_data(datapath, genotype_path, mode):
    # TODO write checks for multiple genotype files.
    # global groundtruth # why is this a global? # removed did it break something?
    groundtruth = None
    genotype_matrix = False
    network_structure = False
    patient_info = False
    multiple_genotype_matrices = False
    number_of_covariats = False
    classification_problem = "undetermined"

    if os.path.exists(genotype_path + 'genotype.h5'):
        genotype_matrix = True
    elif len(glob.glob(genotype_path + '*.h5')) > 0:
        multiple_genotype_matrices = True
    else:
        print("genotype missing in", genotype_path)

    if os.path.exists(datapath + 'topology.csv'):
        network_structure = True
    elif len(glob.glob(datapath + '*.npz')) > 0:
        network_structure = True
    else:
        print("topology.csv and *.npz are missing")
    if os.path.exists(datapath + 'subjects.csv'):
        patient_info = True
        groundtruth = pd.read_csv(datapath + "/subjects.csv")
        
        number_of_covariats = groundtruth.filter(like="cov_").shape[1]
        print('number of covariates:', number_of_covariats)
        print('Covariate columns found:', list(groundtruth.filter(like="cov_").columns.values))
        
        if {'patient_id', 'labels', 'genotype_row', 'set'}.issubset(groundtruth.columns):
            classification_problem = ((groundtruth["labels"].values == 0) | (groundtruth["labels"].values == 1)).all()
        else:
            print("column names missing need 'patient_id', 'labels', 'genotype_row', 'set', got:",
                  groundtruth.columns.values)
            exit()
    else:
        print("subjects.csv is missing")

    print("mode is", mode)
    
    if (mode == "classification") and classification_problem:
        pass
    elif (mode == "regression") and not classification_problem:
        pass
    else:
        print("The labels and the given mode do not correspond. \n"
              "Classification problems should have binary labels [1, 0]. \n"
              "Regression problems should not be binary. \n"
              "The labels do have the following values", groundtruth["labels"].unique())
        exit()

    if multiple_genotype_matrices & network_structure & patient_info:
        TrainDataGenerator.multi_h5 = True
        return
    if genotype_matrix & network_structure & patient_info:
        return
    else:
        print("Did you forget the last (/) slash?")
        exit()


def get_inputsize(genotype_path):
    single_genotype_path = glob.glob(genotype_path + '*.h5')[0]
    h5file = tables.open_file(single_genotype_path, "r")
    inputsize = h5file.root.data.shape[1]
    h5file.close()
    return inputsize


def get_labels(datapath, set_number):
    groundtruth = pd.read_csv(datapath + "/subjects.csv")
    groundtruth = groundtruth[groundtruth["set"] == set_number]
    ybatch = np.reshape(np.array(groundtruth["labels"].values), (-1, 1))
    return ybatch




class TrainDataGenerator(K.utils.Sequence):

    def __init__(self, datapath, genotype_path, batch_size, trainsize, inputsize, epoch_size, shuffle=True, one_hot=False):
        self.datapath = datapath
        self.batch_size = batch_size
        self.genotype_path = genotype_path
        self.shuffledindexes = np.arange(trainsize)
        self.trainsize = trainsize
        self.multi_h5 = len(glob.glob(self.genotype_path + '*.h5')) > 1
        self.h5filenames = "_UKBB_MRI_QC_T_M"
        self.training_subjects = pd.read_csv(self.datapath + "/subjects.csv")
        self.training_subjects = self.training_subjects[self.training_subjects["set"] == 1]
        self.inputsize = inputsize
        self.epoch_size = epoch_size
        self.left_in_greater_epoch = trainsize
        self.count_after_shuffle = 0
        self.one_hot = one_hot

        if shuffle:
            np.random.shuffle(self.shuffledindexes)

    def __len__(self):
        return int(np.ceil(self.epoch_size / float(self.batch_size)))
        

    def __getitem__(self, idx):
        if self.multi_h5:
            xbatch, ybatch = self.multi_genotype_matrix(idx)
        else:
            xbatch, ybatch = self.single_genotype_matrix(idx)

        return xbatch, ybatch


    def if_one_hot(self, xbatch):       
        xbatch_dim = len(xbatch.shape) 
        if self.one_hot:
            if xbatch_dim == 3:
                pass
            elif xbatch_dim == 2:
                xbatch = K.utils.to_categorical(np.array(xbatch, dtype=np.int8))
            else:
                print("unexpected shape!")   
        return xbatch
    
    
    def single_genotype_matrix(self, idx):
        idx2 = idx + self.count_after_shuffle      
        genotype_hdf = tables.open_file(self.genotype_path + "/genotype.h5", "r")
        batchindexes = self.shuffledindexes[idx2 * self.batch_size:((idx2 + 1) * self.batch_size)]
        ybatch = self.training_subjects["labels"].iloc[batchindexes]
        xcov = self.training_subjects.filter(like="cov_").iloc[batchindexes]
        xcov = xcov.values
        xbatchid = np.array(self.training_subjects["genotype_row"].iloc[batchindexes], dtype=int)
        xbatch = genotype_hdf.root.data[xbatchid, :] 
        xbatch = self.if_one_hot(xbatch)
        ybatch = np.reshape(np.array(ybatch), (-1, 1))
        genotype_hdf.close()
        return [xbatch, xcov], ybatch

    def multi_genotype_matrix(self, idx):
        idx2 = idx + self.count_after_shuffle 
        batchindexes = self.shuffledindexes[idx2 * self.batch_size:((idx2 + 1) * self.batch_size)]
        ybatch = self.training_subjects["labels"].iloc[batchindexes]
        
        xcov = self.training_subjects.filter(like="cov_").iloc[batchindexes]
        xcov = xcov.values
        
        subjects_current_batch = self.training_subjects.iloc[batchindexes]
        subjects_current_batch["batch_index"] = np.arange(len(subjects_current_batch))
        xbatch = np.zeros((len(ybatch), self.inputsize))
        for i in subjects_current_batch["chunk_id"].unique():
            genotype_hdf = tables.open_file(self.genotype_path + "/" + str(i) + self.h5filenames + ".h5", "r")
            subjects_current_chunk = subjects_current_batch[subjects_current_batch["chunk_id"] == i]
            xbatchid = np.array(subjects_current_chunk["genotype_row"].values, dtype=int)
            if len(xbatchid) > 1:
                pass
            else:
                xbatchid = int(xbatchid)
            xbatch[subjects_current_chunk["batch_index"].values, :] = genotype_hdf.root.data[xbatchid, :]
            genotype_hdf.close()
            
        xbatch = self.to_one_hot(xbatch)
        ybatch = np.reshape(np.array(ybatch), (-1, 1))
        return [xbatch, xcov], ybatch

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        left_in_epoch = self.left_in_greater_epoch - self.epoch_size
        print(left_in_epoch, 'left_in_epoch')
        if  left_in_epoch < self.epoch_size: 
            print("Shuffeling epochs")
            np.random.shuffle(self.shuffledindexes)
            self.left_in_greater_epoch = self.trainsize
            self.count_after_shuffle = 0
        else:
            self.left_in_greater_epoch = self.left_in_greater_epoch - self.epoch_size
            self.count_after_shuffle = self.count_after_shuffle + int(np.ceil(self.epoch_size / float(self.batch_size)))
            
          


class EvalGenerator(K.utils.Sequence):

    def __init__(self, datapath, genotype_path, batch_size, setsize, inputsize, evalset="undefined", one_hot=False):
        self.datapath = datapath
        self.batch_size = batch_size
        self.yvalsize = setsize
        self.inputsize = inputsize
        self.genotype_path = genotype_path
        self.h5file = []
        self.h5filenames = "_UKBB_MRI_QC_T_M"
        self.multi_h5 = len(glob.glob(self.genotype_path + '*.h5')) > 1
        self.eval_subjects = pd.read_csv(self.datapath + "/subjects.csv")
        self.one_hot = one_hot
        
        if evalset == "validation":
            self.eval_subjects = self.eval_subjects[self.eval_subjects["set"] == 2]
        elif evalset == "test":
            self.eval_subjects = self.eval_subjects[self.eval_subjects["set"] == 3]
        else:
            print("please add which evalset should be used in the call, validation or test. Currently undefined")

    def __len__(self):
        val_len = int(np.ceil(self.yvalsize / float(self.batch_size)))
        return val_len

    def __getitem__(self, idx):
        if self.multi_h5:
            xbatch, ybatch = self.multi_genotype_matrix(idx)
        else:
            xbatch, ybatch = self.single_genotype_matrix(idx)

        return xbatch, ybatch

    def if_one_hot(self, xbatch):       
        xbatch_dim = len(xbatch.shape) 
        if self.one_hot:
            if xbatch_dim == 3:
                pass
            elif xbatch_dim == 2:
                xbatch = K.utils.to_categorical(np.array(xbatch, dtype=np.int8))
            else:
                print("unexpected shape!")   
        return xbatch

    def single_genotype_matrix(self, idx):
        genotype_hdf = tables.open_file(self.genotype_path + "/genotype.h5", "r")
        ybatch = self.eval_subjects["labels"].iloc[idx * self.batch_size:((idx + 1) * self.batch_size)]
        xcov = self.eval_subjects.filter(like="cov_").iloc[idx * self.batch_size:((idx + 1) * self.batch_size)]
        xcov = xcov.values
        xbatchid = np.array(self.eval_subjects["genotype_row"].iloc[idx * self.batch_size:((idx + 1) * self.batch_size)],
                            dtype=int)
        xbatch = genotype_hdf.root.data[xbatchid, :]  
        xbatch = self.if_one_hot(xbatch)
        ybatch = np.reshape(np.array(ybatch), (-1, 1))
        genotype_hdf.close()
        return [xbatch, xcov], ybatch

    def multi_genotype_matrix(self, idx):      
        subjects_current_batch = self.eval_subjects.iloc[idx * self.batch_size:((idx + 1) * self.batch_size)]
        subjects_current_batch["batch_index"] = np.arange(subjects_current_batch.shape[0])
              
        xbatch = np.zeros((len(subjects_current_batch["labels"]), self.inputsize))

        for i in subjects_current_batch["chunk_id"].unique():
            genotype_hdf = tables.open_file(self.genotype_path + "/" + str(i) + self.h5filenames + ".h5", "r")
            subjects_current_chunk = subjects_current_batch[subjects_current_batch["chunk_id"] == i]
            xbatchid = np.array(subjects_current_chunk["genotype_row"].values, dtype=int)
            xbatch[subjects_current_chunk["batch_index"].values, :] = genotype_hdf.root.data[xbatchid, :]
            genotype_hdf.close()
            
        xbatch = self.to_one_hot(xbatch)
        ybatch = np.reshape(np.array(subjects_current_batch["labels"]), (-1, 1))
        xcov = subjects_current_batch.filter(like="cov_").values                     
        return [xbatch, xcov], ybatch
    

    def get_data(self, sample_pat=0):

        genotype_hdf = tables.open_file(self.genotype_path + "/genotype.h5", "r")
        ybatch = self.eval_subjects["labels"]

        if sample_pat > 0:
            self.eval_subjects = self.eval_subjects.sample(n=sample_pat, random_state=1)
        
        xbatchid = np.array(self.eval_subjects["genotype_row"].values, dtype=int)
            
        xcov = self.eval_subjects.filter(like="cov_")
        xcov = xcov.values
        xbatch = genotype_hdf.root.data[xbatchid,...]  
        xbatch = self.if_one_hot(xbatch)
        ybatch = np.reshape(np.array(ybatch), (-1, 1))
        genotype_hdf.close()
        return [xbatch, xcov], ybatch



import glob
import os
import sys

sys.path.insert(1, os.path.dirname(os.getcwd()) + "/GenNet_utils/")
import matplotlib
import numpy as np
import pandas as pd
import tables
import tensorflow.keras as K

matplotlib.use('agg')


def check_data(datapath, mode):
    # TODO write checks for multiple genotype files.
    # global groundtruth # why is this a global? # removed did it break something?
    groundtruth = None
    genotype_matrix = False
    network_structure = False
    patient_info = False
    multiple_genotype_matrices = False
    classification_problem = "undetermined"

    if os.path.exists(datapath + 'genotype.h5'):
        genotype_matrix = True
    elif len(glob.glob(datapath + '*.h5')) > 0:
        multiple_genotype_matrices = True
    else:
        print("genotype.h5 is missing")

    if os.path.exists(datapath + 'topology.csv'):
        network_structure = True
    elif len(glob.glob(datapath + '*.npz')) > 0:
        network_structure = True
    else:
        print("topology.csv and *.npz are missing")
    if os.path.exists(datapath + 'subjects.csv'):
        patient_info = True
        groundtruth = pd.read_csv(datapath + "/subjects.csv")
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
        print("did you forget the last (/) slash?")
        exit()


def get_labels(datapath, set_number):
    groundtruth = pd.read_csv(datapath + "/subjects.csv")
    groundtruth = groundtruth[groundtruth["set"] == set_number]
    ybatch = np.reshape(np.array(groundtruth["labels"].values), (-1, 1))
    return ybatch


def get_data(datapath, set_number):
    groundtruth = pd.read_csv(datapath + "/subjects.csv")
    h5file = tables.open_file(datapath + "genotype.h5", "r")
    groundtruth = groundtruth[groundtruth["set"] == set_number]
    xbatchid = np.array(groundtruth["genotype_row"].values, dtype=np.int64)
    xbatch = h5file.root.data[xbatchid, :]
    ybatch = np.reshape(np.array(groundtruth["labels"].values), (-1, 1))
    h5file.close()
    return xbatch, ybatch


class TrainDataGenerator(K.utils.Sequence):

    def __init__(self, datapath, batch_size, trainsize, shuffle=True):
        self.datapath = datapath
        self.batch_size = batch_size
        self.ytrainsize = trainsize
        self.shuffledindexes = np.arange(trainsize)
        self.multi_h5 = False
        self.h5file = []
        self.h5filenames = "UKBB_MRI"
        self.training_subjects = pd.read_csv(self.datapath + "/subjects.csv")
        self.training_subjects = self.training_subjects[self.training_subjects["set"] == 1]

        if shuffle:
            np.random.shuffle(self.shuffledindexes)

        if self.multi_h5:
            for i in range(len(glob.glob(datapath + '*.h5'))):
                self.h5file.append(tables.open_file(self.datapath + "/" + str(i) + self.h5filenames + ".h5", "r"))
        else:
            self.h5file.append(tables.open_file(self.datapath + "genotype.h5", "r"))

    def __len__(self):
        return int(np.ceil(self.ytrainsize / float(self.batch_size)))

    def __getitem__(self, idx):
        if self.multi_h5:
            xbatch, ybatch = self.multi_genotype_matrix(idx)
        else:
            xbatch, ybatch = self.single_genotype_matrix(idx)

        return xbatch, ybatch

    def single_genotype_matrix(self, idx):
        batchindexes = self.shuffledindexes[idx * self.batch_size:((idx + 1) * self.batch_size)]
        ybatch = self.training_subjects["labels"].iloc[batchindexes]
        xbatchid = np.array(self.training_subjects["genotype_row"].iloc[batchindexes], dtype=np.int64)
        xbatch = self.h5file[0].root.data[xbatchid, :]
        ybatch = np.reshape(np.array(ybatch), (-1, 1))
        return xbatch, ybatch

    def multi_genotype_matrix(self, idx):
        batchindexes = self.shuffledindexes[idx * self.batch_size:((idx + 1) * self.batch_size)]
        ybatch = self.training_subjects["labels"].iloc[batchindexes]
        subjects_current_batch = self.training_subjects.iloc[batchindexes]
        xbatch = np.zeros((len(ybatch), self.h5file[0].root.data.shape[1]))
        for i in subjects_current_batch["chunk_id"].unique():
            subjects_current_chunk = subjects_current_batch[subjects_current_batch["chunk_id"] == i]
            xbatch[batchindexes, :] = self.h5file[i].root.data[subjects_current_chunk["genotype_row"], :]
        ybatch = np.reshape(np.array(ybatch), (-1, 1))
        return xbatch, ybatch

    def on_epoch_begin(self):
        """Updates indexes after each epoch"""
        np.random.shuffle(self.shuffledindexes)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        np.random.shuffle(self.shuffledindexes)

    def close(self):
        """closes all the hdf5 files"""
        for i in len(self.h5file):
            self.h5file[i].close()


class EvalGenerator(K.utils.Sequence):

    def __init__(self, datapath, batch_size, setsize, evalset="undefined"):
        self.datapath = datapath
        self.batch_size = batch_size
        self.yvalsize = setsize

        self.h5file = []
        self.h5filenames = "UKBB_MRI"
        self.eval_subjects = pd.read_csv(self.datapath + "/subjects.csv")
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

    def single_genotype_matrix(self, idx):
        batchindexes = self.shuffledindexes[idx * self.batch_size:((idx + 1) * self.batch_size)]
        ybatch = self.training_subjects["labels"].iloc[batchindexes]
        xbatchid = np.array(self.training_subjects["genotype_row"].iloc[batchindexes], dtype=np.int64)
        xbatch = self.h5file[0].root.data[xbatchid, :]
        ybatch = np.reshape(np.array(ybatch), (-1, 1))
        return xbatch, ybatch

    def multi_genotype_matrix(self, idx):
        batchindexes = self.shuffledindexes[idx * self.batch_size:((idx + 1) * self.batch_size)]
        ybatch = self.training_subjects["labels"].iloc[batchindexes]
        subjects_current_batch = self.training_subjects.iloc[batchindexes]
        xbatch = np.zeros((len(ybatch), self.h5file[0].root.data.shape[1]))
        for i in subjects_current_batch["chunk_id"].unique():
            subjects_current_chunk = subjects_current_batch[subjects_current_batch["chunk_id"] == i]
            xbatch[batchindexes, :] = self.h5file[i].root.data[subjects_current_chunk["genotype_row"], :]
        ybatch = np.reshape(np.array(ybatch), (-1, 1))
        return xbatch, ybatch



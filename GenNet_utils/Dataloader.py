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
    genotype_matrix = False
    network_structure = False
    patient_info = False
    classification_problem = "undetermined"

    if os.path.exists(datapath + 'genotype.h5'):
        genotype_matrix = True
    else:
        print("genotype.h5 is missing")
    if os.path.exists(datapath + 'topology.csv'):
        network_structure = True
    elif os.path.exists(datapath + '*.npz'):
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
    elif (mode == "regression") and not (classification_problem):
        pass
    else:
        print("The labels and the given mode do not correspond. \n"
              "Classification problems should have binary labels [1, 0]. \n"
              "Regression problems should not be binary. \n"
              "The labels do have the following values", groundtruth["labels"].unique())
        exit()

    if (genotype_matrix & network_structure & patient_info):
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
    # ybatch = (ybatch > 0)*1
    h5file.close()
    return (xbatch, ybatch)


class traindata_generator(K.utils.Sequence):

    def __init__(self, datapath, batch_size, trainsize, shuffle=True):
        self.datapath = datapath
        self.batch_size = batch_size
        self.ytrainsize = trainsize
        self.shuffledindexes = np.arange(trainsize)
        if shuffle:
            np.random.shuffle(self.shuffledindexes)

    def __len__(self):
        return int(np.ceil(self.ytrainsize / float(self.batch_size)))

    def __getitem__(self, idx):
        batchindexes = self.shuffledindexes[idx * self.batch_size:((idx + 1) * self.batch_size)]
        ytrain = pd.read_csv(self.datapath + "/subjects.csv")
        ytrain = ytrain[ytrain["set"] == 1]
        h5file = tables.open_file(self.datapath + "genotype.h5", "r")
        ybatch = ytrain["labels"].iloc[batchindexes]
        xbatchid = np.array(ytrain["genotype_row"].iloc[batchindexes], dtype=np.int64)
        xbatch = h5file.root.data[xbatchid, :]
        # xbatch = 2 - xbatch
        ybatch = np.reshape(np.array(ybatch), (-1, 1))
        # ybatch = (ybatch > 0)*1
        h5file.close()

        return xbatch, ybatch

    def on_epoch_begin(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.shuffledindexes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.shuffledindexes)


class valdata_generator(K.utils.Sequence):

    def __init__(self, datapath, batch_size, valsize):
        self.datapath = datapath
        self.batch_size = batch_size
        self.yvalsize = valsize

    def __len__(self):
        val_len = int(np.ceil(self.yvalsize / float(self.batch_size)))
        return val_len

    def __getitem__(self, idx):
        yval = pd.read_csv(self.datapath + "/subjects.csv")
        yval = yval[yval["set"] == 2]
        h5file = tables.open_file(self.datapath + "genotype.h5", "r")
        ybatch = yval["labels"].iloc[idx * self.batch_size:((idx + 1) * self.batch_size)]
        xbatchid = np.array(yval["genotype_row"].iloc[idx * self.batch_size:((idx + 1) * self.batch_size)],
                            dtype=np.int64)
        xbatch = h5file.root.data[xbatchid, :]
        # xbatch = 2 - xbatch
        ybatch = np.reshape(np.array(ybatch), (-1, 1))
        # ybatch = (ybatch > 0)*1
        h5file.close()
        return xbatch, ybatch


class testdata_generator(K.utils.Sequence):

    def __init__(self, datapath, batch_size, testsize):
        self.datapath = datapath
        self.batch_size = batch_size
        self.ytestsize = testsize

    def __len__(self):
        val_len = int(np.ceil(self.ytestsize / float(self.batch_size)))
        return val_len

    def __getitem__(self, idx):
        yval = pd.read_csv(self.datapath + "/subjects.csv")
        yval = yval[yval["set"] == 3]
        h5file = tables.open_file(self.datapath + "genotype.h5", "r")
        ybatch = yval["labels"].iloc[idx * self.batch_size:((idx + 1) * self.batch_size)]
        xbatchid = np.array(yval["genotype_row"].iloc[idx * self.batch_size:((idx + 1) * self.batch_size)],
                            dtype=np.int64)
        xbatch = h5file.root.data[xbatchid, :]
        # xbatch = 2 - xbatch
        ybatch = np.reshape(np.array(ybatch), (-1, 1))
        # ybatch = (ybatch > 0)*1
        h5file.close()
        return xbatch, ybatch

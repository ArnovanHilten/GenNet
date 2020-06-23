import pandas as pd
import numpy as np
import tensorflow as tf
import os
import tables
import tensorflow.keras as K
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import scipy
tf.keras.backend.set_epsilon(0.0000001)
import sys
from joblib import Parallel, delayed
import scipy
from scipy.sparse import coo_matrix
import glob
import os
from scipy import stats
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score


def explode(df, cols, split_on=','):
    """
    Explode dataframe on the given column, split on given delimeter
    """
    cols_sep = list(set(df.columns) - set(cols))
    df_cols = df[cols_sep]
    explode_len = df[cols[0]].str.split(split_on).map(len)
    repeat_list = []
    for r, e in zip(df_cols.as_matrix(), explode_len):
        repeat_list.extend([list(r)]*e)
    df_repeat = pd.DataFrame(repeat_list, columns=cols_sep)
    df_explode = pd.concat([df[col].str.split(split_on, expand=True).stack().str.strip().reset_index(drop=True)
                            for col in cols], axis=1)
    df_explode.columns = cols
    return pd.concat((df_repeat, df_explode), axis=1)

def sensitivity(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.keras.backend.clip(y_pred,0,1)
    y_true = tf.keras.backend.clip(y_true,0,1)

    y_pred = tf.keras.backend.round(y_pred)


    true_p = K.backend.sum(K.backend.round(y_pred) * y_true)
    pos = tf.keras.backend.sum(y_true)
    sensitivity =  tf.keras.backend.clip((true_p / (pos+ 0.00001)),0,1)
    return sensitivity



def specificity(y_true,y_pred):
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.keras.backend.clip(y_pred,0,1)
    y_true = tf.keras.backend.clip(y_true,0,1)
    
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - K.backend.round(y_pred)
    fp = K.backend.sum(neg_y_true * K.backend.round(y_pred))
    tn = K.backend.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + 0.00001)
    return tf.keras.backend.clip(specificity,0,1)

def binary_accuracy(y_true, y_pred):
    return K.backend.mean(K.backend.equal(y_true, K.backend.round(y_pred)))

def evaluate_performance(y,p):
    print("\n")
    print("Confusion matrix")
    confusion_matrix = skm.confusion_matrix(y, p.round())
    print(confusion_matrix)

    fpr, tpr, thresholds = skm.roc_curve(y, p)
    roc_auc = skm.auc(fpr, tpr)
    print("\n")
    print("Area under the Curve (AUC) = ", roc_auc)

    specificity = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
    print('Specificity = ', specificity)

    sensitivity = confusion_matrix[1, 1] / (confusion_matrix[1, 0] + confusion_matrix[1, 1])
    print('Sensitivity = ', sensitivity)
    print("F_1 score = " + str(skm.f1_score(y, p.round())))
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
            lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(1 - specificity, sensitivity, color='b', marker='o')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import tensorflow as tf
import tensorflow.keras as K

tf.keras.backend.set_epsilon(0.0000001)
import os
import seaborn as sns
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

def use_mixed_precision():
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)


def get_paths(jobid):
    folder = ("GenNet_experiment_" + str(jobid))

    resultpath = os.path.dirname(os.getcwd()) + "/GenNet/results/" + folder + "/"
    if not os.path.exists(resultpath):
        print("Resultspath did not exist but is made now")
        os.mkdir(resultpath)

    return folder, resultpath


def explode(df, cols, split_on=','):
    """
    Explode dataframe on the given column, split on given delimeter
    """
    cols_sep = list(set(df.columns) - set(cols))
    df_cols = df[cols_sep]
    explode_len = df[cols[0]].str.split(split_on).map(len)
    repeat_list = []
    for r, e in zip(df_cols.as_matrix(), explode_len):
        repeat_list.extend([list(r)] * e)
    df_repeat = pd.DataFrame(repeat_list, columns=cols_sep)
    df_explode = pd.concat([df[col].str.split(split_on, expand=True).stack().str.strip().reset_index(drop=True)
                            for col in cols], axis=1)
    df_explode.columns = cols
    return pd.concat((df_repeat, df_explode), axis=1)


def sensitivity(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.keras.backend.clip(y_pred, 0, 1)
    y_true = tf.keras.backend.clip(y_true, 0, 1)

    y_pred = tf.keras.backend.round(y_pred)

    true_p = K.backend.sum(K.backend.round(y_pred) * y_true)
    pos = tf.keras.backend.sum(y_true)
    sensitivity = tf.keras.backend.clip((true_p / (pos + 0.00001)), 0, 1)
    return sensitivity


def specificity(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.keras.backend.clip(y_pred, 0, 1)
    y_true = tf.keras.backend.clip(y_true, 0, 1)

    neg_y_true = 1 - y_true
    neg_y_pred = 1 - K.backend.round(y_pred)
    fp = K.backend.sum(neg_y_true * K.backend.round(y_pred))
    tn = K.backend.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + 0.00001)
    return tf.keras.backend.clip(specificity, 0, 1)


def binary_accuracy(y_true, y_pred):
    return K.backend.mean(K.backend.equal(y_true, K.backend.round(y_pred)))


def evaluate_performance_regression(y, p):
    y = y.flatten()
    p = p.flatten()
    explained_variance = explained_variance_score(y, p)
    mse = mean_squared_error(y, p)
    r2 = r2_score(y, p)
    print("Mean squared error =", mse)
    print("Explained variance =", explained_variance)
    # print("maximum error =", maximum_error)
    print("r2 =", r2)

    plt.figure()
    df = pd.DataFrame([])
    df["truth"] = y
    df["predicted"] = p

    fig = sns.jointplot(x="truth", y="predicted", data=df, alpha=0.5)
    return fig, mse, explained_variance, r2


def evaluate_performance(y, p):
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

    return roc_auc, confusion_matrix


def create_importance_csv(datapath, model, masks):
    network_csv = pd.read_csv(datapath + "/topology.csv")

    coordinate_list = []
    for i, mask in zip(np.arange(len(masks)), masks):
        coordinates = pd.DataFrame([])

        if (i == 0):
            if 'chr' in network_csv.columns:
                coordinates["chr"] = network_csv["chr"]
        coordinates["node_layer_" + str(i)] = mask.row
        coordinates["node_layer_" + str(i + 1)] = mask.col
        coordinates = coordinates.sort_values("node_layer_" + str(i), ascending=True)
        coordinates["weights_" + str(i)] = model.get_layer(name="LocallyDirected_" + str(i)).get_weights()[0]

        coordinate_names = network_csv[["layer" + str(i) + "_node", "layer" + str(i) + "_name"]].drop_duplicates()
        coordinate_names = coordinate_names.rename({"layer" + str(i) + "_node": "node_layer_" + str(i)}, axis=1)
        coordinates = coordinates.merge(coordinate_names, on="node_layer_" + str(i))
        coordinate_list.append(coordinates)

        if i == 0:
            total_list = coordinate_list[i]
        else:
            total_list = total_list.merge(coordinate_list[i], on="node_layer_" + str(i))

    i += 1
    coordinates = pd.DataFrame([])
    coordinates["weights_" + str(i)] = model.get_layer(name="output_layer").get_weights()[0].flatten()
    coordinates["node_layer_" + str(i)] = np.arange(len(coordinates))
    coordinate_names = network_csv[["layer" + str(i) + "_node", "layer" + str(i) + "_name"]].drop_duplicates()
    coordinate_names = coordinate_names.rename({"layer" + str(i) + "_node": "node_layer_" + str(i)}, axis=1)
    coordinates = coordinates.merge(coordinate_names, on="node_layer_" + str(i))
    total_list = total_list.merge(coordinates, on="node_layer_" + str(i))
    total_list["raw_importance"] = total_list.filter(like="weights").prod(axis=1)
    return total_list


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

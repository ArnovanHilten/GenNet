import os

import pandas as pd

from GenNet_utils.Create_plots import plot_layer_weight, manhattan_importance, sunburst_plot
from GenNet_utils.Utility_functions import get_paths


# TODO: add test without covariates
# TODO add test with covariates for regression + classification
# TODO add test with multiple genotype files.

def test_train_standard():
    value = os.system('cd .. && python GenNet.py train  ./examples/example_study/ 1000')
    assert value == 0


def test_train_regression():
    value = os.system('cd .. && python GenNet.py train  ./examples/example_regression/ 1001 -problem_type regression')
    assert value == 0


def test_train(datapath, jobid, wpc, lr_opt, batch_size, epochs, l1_value, problem_type, ):
    test1 = os.system(
        'cd .. && python GenNet.py train {datapath}  {jobid} -problem_type'
        ' {problem_type} -wpc {wpc} -lr {lr} -bs {bs}  -epochs {epochs} -L1 {L1}'.format(
            datapath=datapath, jobid=jobid, problem_type=problem_type, wpc=wpc, lr=lr_opt, bs=batch_size, epochs=epochs,
            L1=l1_value))

    assert test1 == 0

    folder, resultpath = get_paths(jobid=jobid)
    test2 = os.path.exists(resultpath + '/bestweights_job.h5')
    assert test2


def test_convert():
    test1 = os.system(
        "python hase.py - mode converting - g /media/avanhilten/pHDD1TB/dbGaP_BulgarianTrio/GenotypeFiles/matrix/plink/"
        " -o /media/avanhilten/pSSD450/GenNet/hase/"
        " -study_name BulgarianTrio")
    assert test1 == 0


def test_plot(exp_id):
    importance_csv = pd.read_csv(
        "/home/avanhilten/PycharmProjects/GenNet/results/GenNet_experiment_" + str(exp_id) + "/connection_weights.csv",
        index_col=0)
    resultpath = '/home/avanhilten/PycharmProjects/GenNet/results/GenNet_experiment_' + str(exp_id) + '/'

    sunburst_plot(resultpath, importance_csv)
    manhattan_importance(resultpath, importance_csv)
    plot_layer_weight(resultpath, importance_csv, layer=0)
    plot_layer_weight(resultpath, importance_csv, layer=1)
    plot_layer_weight(resultpath, importance_csv, layer=2)

if __name__ == '__main__':
    # test_train_standard()
    # test_train_regression()
    exp_id = 1
    test_plot(exp_id)

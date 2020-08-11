import os
from utils.utils import get_paths


def test_train_standard():
    value = os.system('cd .. && python GenNet.py train  ./processed_data/example_study/ 1')
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


if __name__ == '__main__':
    test_train_standard()

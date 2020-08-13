import sys
import os
import matplotlib
import warnings

warnings.filterwarnings('ignore')
matplotlib.use('agg')
sys.path.insert(1, os.path.dirname(os.getcwd()) + "/utils/")
import tensorflow as tf
import tensorflow.keras as K

tf.keras.backend.set_epsilon(0.0000001)
from utils.Dataloader import *
from utils.utils import *
from utils.Create_network import *
from utils.Create_plots import *


def weighted_binary_crossentropy(y_true, y_pred):
    y_true = K.backend.clip(y_true, 0.0001, 1)
    y_pred = K.backend.clip(y_pred, 0.0001, 1)

    return K.backend.mean(
        -y_true * K.backend.log(y_pred + 0.0001) * weight_positive_class - (1 - y_true) * K.backend.log(
            1 - y_pred + 0.0001) * weight_negative_class)


def train_classification(args):
    datapath = args.path
    jobid = args.ID
    wpc = args.wpc
    lr_opt = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    l1_value = args.L1
    problem_type = args.problem_type
    check_data(datapath, problem_type)

    global weight_positive_class, weight_negative_class
    weight_positive_class = wpc
    weight_negative_class = 1
    optimizer_model = tf.keras.optimizers.Adam(lr=lr_opt)

    train_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 1)
    val_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 2)
    test_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 3)

    folder, resultpath = get_paths(jobid)

    print("weight_possitive_class", weight_positive_class)
    print("weight_possitive_class", weight_negative_class)
    print("jobid =  " + str(jobid))
    print("folder = " + str(folder))
    print("batchsize = " + str(batch_size))
    print("lr = " + str(lr_opt))

    model, masks = create_network_from_csv(datapath=datapath, l1_value=l1_value)
    model.compile(loss=weighted_binary_crossentropy, optimizer=optimizer_model,
                  metrics=["accuracy", sensitivity, specificity])

    with open(resultpath + '/model_architecture.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    earlystop = K.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto',
                                          restore_best_weights=True)
    saveBestModel = K.callbacks.ModelCheckpoint(resultpath + "bestweights_job.h5", monitor='val_loss',
                                                verbose=1, save_best_only=True, mode='auto')

    if os.path.exists(resultpath + '/bestweights_job.h5'):
        print('Model already Trained')
    else:
        history = model.fit_generator(
            generator=traindata_generator(datapath=datapath,
                                          batch_size=batch_size,
                                          trainsize=int(train_size)),
            shuffle=True,
            epochs=epochs,
            verbose=1,
            callbacks=[earlystop, saveBestModel],
            workers=15,
            use_multiprocessing=True,
            validation_data=valdata_generator(datapath=datapath, batch_size=batch_size, valsize=val_size)
        )
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(resultpath + "train_val_loss.png")
        plt.show()

    model.load_weights(resultpath + '/bestweights_job.h5')
    print("Finished")
    print("Analysis over the validation set")
    pval = model.predict_generator(
        valdata_generator(datapath=datapath, batch_size=1, valsize=val_size))
    yval = get_labels(datapath, set_number=2)
    auc_val, confusionmatrix_val = evaluate_performance(yval, pval)
    np.save(resultpath + "/pval.npy", pval)

    print("Analysis over the test set")
    ptest = model.predict_generator(
        testdata_generator(datapath=datapath, batch_size=1, testsize=test_size))
    ytest = get_labels(datapath, set_number=3)
    auc_test, confusionmatrix_test = evaluate_performance(ytest, ptest)
    np.save(resultpath + "/ptest.npy", ptest)

    with open(resultpath + '/Results_' + str(jobid) + '.txt', 'a') as f:
        f.write('\n Jobid = ' + str(jobid))
        f.write('\n Batchsize = ' + str(batch_size))
        f.write('\n Weight positive class = ' + str(weight_positive_class))
        f.write('\n Weight negative class= ' + str(weight_negative_class))
        f.write('\n Learningrate = ' + str(lr_opt))
        f.write('\n Optimizer = ' + str(optimizer_model))
        f.write('\n L1 value = ' + str(l1_value))
        f.write('\n')
        f.write("Validation set")
        f.write('\n Score auc = ' + str(auc_val))
        f.write('\n Confusion matrix:')
        f.write(str(confusionmatrix_val))
        f.write('\n')
        f.write("Test set")
        f.write('\n Score auc = ' + str(auc_test))
        f.write('\n Confusion matrix ')
        f.write(str(confusionmatrix_test))

    importance_csv = create_importance_csv(datapath, model, masks)
    importance_csv.to_csv(resultpath + "connection_weights.csv")


def train_regression(args):
    datapath = args.path
    jobid = args.ID
    lr_opt = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    l1_value = args.L1
    problem_type = args.problem_type
    check_data(datapath, problem_type)

    optimizer_model = tf.keras.optimizers.Adam(lr=lr_opt)

    train_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 1)
    val_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 2)
    test_size = sum(pd.read_csv(datapath + "subjects.csv")["set"] == 3)

    folder, resultpath = get_paths(jobid)

    print("jobid =  " + str(jobid))
    print("folder = " + str(folder))
    print("batchsize = " + str(batch_size))
    print("lr = " + str(lr_opt))

    model, masks = create_network_from_csv(datapath=datapath, l1_value=l1_value, regression=True)
    model.compile(loss="mse", optimizer=optimizer_model,
                  metrics=["mse"])

    with open(resultpath + '/model_architecture.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    earlystop = K.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto',
                                          restore_best_weights=True)
    saveBestModel = K.callbacks.ModelCheckpoint(resultpath + "bestweights_job.h5", monitor='val_loss',
                                                verbose=1, save_best_only=True, mode='auto')

    # %%
    if os.path.exists(resultpath + '/bestweights_job.h5'):
        print('Model already Trained')
    else:
        history = model.fit_generator(
            generator=traindata_generator(datapath=datapath,
                                          batch_size=batch_size,
                                          trainsize=int(train_size)),
            shuffle=True,
            epochs=epochs,
            verbose=1,
            callbacks=[earlystop, saveBestModel],
            workers=15,
            use_multiprocessing=True,
            validation_data=valdata_generator(datapath=datapath, batch_size=batch_size, valsize=val_size)
        )
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(resultpath + "train_val_loss.png")
        plt.show()

    model.load_weights(resultpath + '/bestweights_job.h5')
    print("Finished")
    print("Analysis over the validation set")
    pval = model.predict_generator(
        valdata_generator(datapath=datapath, batch_size=1, valsize=val_size))
    yval = get_labels(datapath, set_number=2)
    fig, mse_val, explained_variance_val, r2_val = evaluate_performance_regression(yval, pval)
    np.save(resultpath + "/pval.npy", pval)
    fig.savefig(resultpath + "/validation_predictions.png", bbox_inches='tight', pad_inches=0)

    print("Analysis over the test set")
    ptest = model.predict_generator(
        testdata_generator(datapath=datapath, batch_size=1, testsize=test_size))
    ytest = get_labels(datapath, set_number=3)
    fig, mse_test, explained_variance_test, r2_test = evaluate_performance_regression(ytest, ptest)
    np.save(resultpath + "/ptest.npy", ptest)
    fig.savefig(resultpath + "/test_predictions.png", bbox_inches='tight', pad_inches=0)

    # %%

    with open(resultpath + '/Results_' + str(jobid) + '.txt', 'a') as f:
        f.write('\n Jobid = ' + str(jobid))
        f.write('\n Batchsize = ' + str(batch_size))
        f.write('\n Learningrate = ' + str(lr_opt))
        f.write('\n Optimizer = ' + str(optimizer_model))
        f.write('\n L1 value = ' + str(l1_value))
        f.write('\n')
        f.write("Validation set")
        f.write('\n Mean squared error = ' + str(mse_val))
        f.write('\n Explained variance = ' + str(explained_variance_val))
        # f.write('\n Maximum error = ' + str(maximum_error_val))
        f.write('\n R2 = ' + str(r2_val))
        f.write("Test set")
        f.write('\n Mean squared error = ' + str(mse_test))
        f.write('\n Explained variance = ' + str(explained_variance_val))
        # f.write('\n Maximum error = ' + str(maximum_error_test))
        f.write('\n R2 = ' + str(r2_test))
    importance_csv = create_importance_csv(datapath, model, masks)
    importance_csv.to_csv(resultpath + "connection_weights.csv")

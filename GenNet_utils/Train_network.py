import os
import sys
import warnings
import shutil
import matplotlib
import datetime
warnings.filterwarnings('ignore')
matplotlib.use('agg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
import tensorflow.keras as K

tf.keras.backend.set_epsilon(0.0000001)
from GenNet_utils.Dataloader import *
from GenNet_utils.Utility_functions import *
from GenNet_utils.Create_network import *
from GenNet_utils.Create_plots import *
from GenNet_utils.Utility_functions import load_train_arguments


def weighted_binary_crossentropy(y_true, y_pred):
    y_true = K.backend.clip(tf.cast(y_true, dtype=tf.float32), 0.0001, 1)
    y_pred = K.backend.clip(tf.cast(y_pred, dtype=tf.float32), 0.0001, 1)

    return K.backend.mean(
        -y_true * K.backend.log(y_pred + 0.0001) * weight_positive_class - (1 - y_true) * K.backend.log(
            1 - y_pred + 0.0001) * weight_negative_class)


def train_model(args):
    args.SlURM_JOB_ID = get_SLURM_id()
    model = None
    masks = None
    
    args.datapath = args.path
    

    if args.genotype_path == "undefined":
        args.genotype_path = args.path

    if args.mixed_precision:
        use_mixed_precision()

    args.multiprocessing = True if args.workers > 1 else False

    check_data(datapath=args.path, genotype_path=args.genotype_path, mode=args.problem_type)

    global weight_positive_class, weight_negative_class

    weight_positive_class = args.wpc
    weight_negative_class = 1

    args.train_size = sum(pd.read_csv(args.path + "subjects.csv")["set"] == 1)
    args.val_size = sum(pd.read_csv(args.path + "subjects.csv")["set"] == 2)
    args.test_size = sum(pd.read_csv(args.path + "subjects.csv")["set"] == 3)
    args.num_covariates = pd.read_csv(args.path + "subjects.csv").filter(like='cov_').shape[1]
    args.val_size_train = args.val_size

    if args.epoch_size is None:
        args.epoch_size = args.train_size
    else:
        args.val_size_train = min(args.epoch_size // 2, args.val_size)
        print("Using each epoch", args.epoch_size, "randomly selected training examples")
        print("Validation set size used during training is also set to half the epoch_size")

    args.inputsize = get_inputsize(args.genotype_path)

    folder, args.resultpath = get_paths(args)

    print("weight_positive_class", weight_positive_class)
    print("weight_negative_class", weight_negative_class)
    print("jobid =  " + str(args.ID))
    print("folder = " + str(folder))
    print("batchsize = " + str(args.batch_size))
    print("lr = " + str(args.learning_rate))

    model, masks = get_network(args)

    csv_logger = K.callbacks.CSVLogger(args.resultpath + 'train_log.csv', append=True)

    early_stop = K.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=args.patience, verbose=1,
                                           mode='auto', restore_best_weights=True)
    save_best_model = K.callbacks.ModelCheckpoint(args.resultpath + "bestweights_job.h5", monitor='val_loss',
                                                  verbose=1, save_best_only=True, mode='auto')

    reduce_lr = K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                              patience=5, min_lr=0.001)

    if os.path.exists(args.resultpath + '/bestweights_job.h5') and not(args.resume):
        print('Model already Trained')
    elif os.path.exists(args.resultpath + '/bestweights_job.h5') and args.resume:
        print("load and save weights before resuming")
        shutil.copyfile(args.resultpath + '/bestweights_job.h5', args.resultpath + '/weights_before_resuming_' 
                        + datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%p")+'.h5') # save old weights
        log_file = pd.read_csv(args.resultpath + "/train_log.csv")
        save_best_model = K.callbacks.ModelCheckpoint(args.args.resultpath + "bestweights_job.h5", monitor='val_loss',
                                                  verbose=1, save_best_only=True, mode='auto', 
                                                      initial_epoch=len(log_file))
            
        print("Resuming training")
        model.load_weights(args.resultpath + '/bestweights_job.h5')
        train_generator = TrainDataGenerator(datapath=args.path,
                                             genotype_path=args.genotype_path,
                                             batch_size=args.batch_size,
                                             trainsize=int(args.train_size),
                                             inputsize=args.inputsize,
                                             epoch_size=args.epoch_size)
        history = model.fit_generator(
            generator=train_generator,
            shuffle=True,
            epochs=args.epochs,
            verbose=1,
            callbacks=[early_stop, save_best_model, csv_logger, reduce_lr],
            workers=args.workers,
            use_multiprocessing=args.multiprocessing,
            validation_data=EvalGenerator(datapath=args.path, genotype_path=args.genotype_path, batch_size=args.batch_size,
                                          setsize=args.val_size_train,
                                          inputsize=args.inputsize, evalset="validation")
        )
    else:
        print("Start training from scratch")
        train_generator = TrainDataGenerator(datapath=args.path,
                                             genotype_path=args.genotype_path,
                                             batch_size=args.batch_size,
                                             trainsize=int(args.train_size),
                                             inputsize=args.inputsize,
                                             epoch_size=args.epoch_size)
        history = model.fit_generator(
            generator=train_generator,
            shuffle=True,
            epochs=args.epochs,
            verbose=1,
            callbacks=[early_stop, save_best_model, csv_logger, reduce_lr],
            workers=args.workers,
            use_multiprocessing=args.multiprocessing,
            validation_data=EvalGenerator(datapath=args.path, genotype_path=args.genotype_path, batch_size=args.batch_size,
                                          setsize=args.val_size_train,
                                          inputsize=args.inputsize, evalset="validation")
        )

    plot_loss_function(args.resultpath)
    model.load_weights(args.resultpath + '/bestweights_job.h5')
    print("Finished")
    save_train_arguments(args)


    if args.regression:
        # Regression-specific post training analysis
        print("Analysis over the validation set")
        pval = model.predict_generator(
            EvalGenerator(datapath=args.datapath, genotype_path=args.genotype_path, batch_size=args.batch_size,
                          setsize=args.val_size, inputsize=args.inputsize, evalset="validation"))
        yval = get_labels(args.datapath, set_number=2)
        fig_val, mse_val, explained_variance_val, r2_val = evaluate_performance_regression(yval, pval)
        np.save(args.resultpath + "/pval.npy", pval)
        fig_val.savefig(args.resultpath + "/validation_predictions.png", bbox_inches='tight', pad_inches=0)

        print("Analysis over the test set")
        ptest = model.predict_generator(
            EvalGenerator(datapath=args.datapath, genotype_path=args.genotype_path, batch_size=args.batch_size,
                          setsize=args.test_size, inputsize=args.inputsize, evalset="test"))
        ytest = get_labels(args.datapath, set_number=3)
        fig_test, mse_test, explained_variance_test, r2_test = evaluate_performance_regression(ytest, ptest)
        np.save(args.resultpath + "/ptest.npy", ptest)
        fig_test.savefig(args.resultpath + "/test_predictions.png", bbox_inches='tight', pad_inches=0)
    else:
        # Classification-specific post training analysis
        print("Analysis over the validation set")
        pval = model.predict_generator(
            EvalGenerator(datapath=args.datapath, genotype_path=args.genotype_path, batch_size=args.batch_size,
                          setsize=args.val_size, evalset="validation", inputsize=args.inputsize))
        yval = get_labels(args.datapath, set_number=2)
        auc_val, confusionmatrix_val = evaluate_performance_classification(yval, pval)
        

        print("Analysis over the test set")
        ptest = model.predict_generator(
            EvalGenerator(datapath=args.datapath, genotype_path=args.genotype_path, batch_size=args.batch_size,
                          setsize=args.test_size, evalset="test", inputsize=args.inputsize))
        ytest = get_labels(args.datapath, set_number=3)
        auc_test, confusionmatrix_test = evaluate_performance_classification(ytest, ptest)
        

    # Saving Results
    data = {'Jobid': args.ID,
            'Datapath': str(args.path),
            'genotype_path': str(args.genotype_path),
            'Batchsize': args.batch_size,
            'Learning rate': args.learning_rate,
            'L1 value': args.L1,
            'L1 act': args.L1_act,
            'patience': args.patience,
            'epoch size': args.epoch_size,
            'epochs': args.epochs,
            'SlURM_JOB_ID': args.SlURM_JOB_ID}
    
    
    if args.regression:
        data.update({
            'MSE validation': mse_val,
            'MSE test': mse_test,
            'Explained variance val': explained_variance_val,
            'Explained variance test': explained_variance_test,
            'R2_validation': r2_val,
            'R2_test': r2_test
        })
    else:
        data.update({
            'Weight positive class': args.wpc,
            'AUC validation': auc_val,
            'AUC test': auc_test,
        })

        data['confusionmatrix_val'] = confusionmatrix_val
        data['confusionmatrix_test'] = confusionmatrix_test

    pd_summary_row = pd.Series(data)
    pd_summary_row.to_csv(args.resultpath + "/pd_summary_results.csv")
    
    with open(args.resultpath + "results_summary.txt", 'w') as f: 
        for key, value in data.items(): 
            f.write('%s:%s\n' % (key, value))

    if os.path.exists(args.datapath + "/topology.csv"):
        importance_csv = create_importance_csv(args.datapath, model, masks)
        importance_csv.to_csv(args.resultpath + "connection_weights.csv")

def get_network(args):
    """needs the following inputs
        args.inputsize
        args.L1
        args.L1_act
        args.datapath
        args.genotype_path
        args.num_covariates
        args.filter
    """
    regression = args.regression if hasattr(args, 'regression') else False

    args.L1 = args.regression if hasattr(args, 'regression') else False

    if args.network_name == "lasso" and not regression:
        print("lasso network")
        model, masks = lasso(inputsize=args.inputsize, l1_value=args.L1, L1_act=args.L1_act if hasattr(args, 'L1_act') else None)

    elif args.network_name == "sparse_directed_gene_l1" and not regression:
        print("sparse_directed_gene_l1 network")
        model, masks = sparse_directed_gene_l1(inputsize=args.inputsize, l1_value=args.L1)


    elif args.network_name == "regression_height" and regression:
        print("regression_height network")
        model, masks = regression_height(inputsize=args.inputsize, l1_value=args.L1)

    elif args.network_name == "gene_network_multiple_filters":
        print("gene_network_multiple_filters network")
        model, masks = gene_network_multiple_filters(datapath=args.datapath, inputsize=args.inputsize, genotype_path=args.genotype_path,
                                                     l1_value=args.L1, L1_act=args.L1_act, 
                                                     regression=regression, num_covariates=args.num_covariates,
                                                     filters=args.filters)

    elif args.network_name == "gene_network_snp_gene_filters":
        print("gene_network_snp_gene_filters network")
        model, masks = gene_network_snp_gene_filters(datapath=args.datapath, inputsize=args.inputsize, genotype_path=args.genotype_path,
                                                     l1_value=args.L1, L1_act=args.L1_act, 
                                                     regression=regression, num_covariates=args.num_covariates,
                                                     filters=args.filters)
    else:
        if os.path.exists(args.datapath + "/topology.csv"):
            model, masks = create_network_from_csv(datapath=args.datapath, inputsize=args.inputsize, genotype_path=args.genotype_path,
                                                   l1_value=args.L1, L1_act=args.L1_act, regression=regression,
                                                   num_covariates=args.num_covariates)
        elif len(glob.glob(args.datapath + "/*.npz")) > 0:
            model, masks = create_network_from_npz(datapath=args.datapath, inputsize=args.inputsize, genotype_path=args.genotype_path,
                                                   l1_value=args.L1, L1_act=args.L1_act, regression=regression,
                                                   num_covariates=args.num_covariates, 
                                                   mask_order=args.mask_order if hasattr(args, 'mask_order') else None)

    optimizer_model = tf.keras.optimizers.Adam(lr=args.learning_rate)

    if regression:
        model.compile(loss="mse", optimizer=optimizer_model, metrics=["mse"])
    else:
        model.compile(loss=weighted_binary_crossentropy, optimizer=optimizer_model,
                      metrics=["accuracy", sensitivity, specificity])

    with open(args.resultpath + '/model_architecture.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    return model, masks


def load_trained_network(args):
    """
    args.resultpath

    get_network needs the following inputs
        args.inputsize
        args.L1
        args.L1_act
        args.datapath
        args.genotype_path
        args.num_covariates
        args.filter
    """

    args = load_train_arguments(args)

    model, mask = get_network(args)
    model.load_weights(args.resultpath + '/bestweights_job.h5')

    return model, mask





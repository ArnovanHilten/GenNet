import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
import tables
import tensorflow.keras as K
import sklearn.metrics as skm
matplotlib.use('agg')
import scipy
import sys
import argparse
import time
import tqdm
import glob
import scipy
import scipy.sparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/GenNet_utils/")
print(os.path.dirname(os.path.abspath(__file__)) + "/GenNet_utils/")
tf.keras.backend.set_epsilon(0.0000001)


from Utility_functions import get_SLURM_id, sensitivity, specificity, evaluate_performance
from GenNet_simulation_utility import make_mask_gene_layer, get_data

tf_version = tf.__version__  # ToDo use packaging.version
if tf_version <= '1.13.1':
    from LocallyDirected1D import LocallyDirected1D
elif tf_version >= '2.0':
    from LocallyDirected1D import LocallyDirected1D
else:
    print("tf version error", tf_version)
    from LocallyDirected1D import LocallyDirected1D
print("start")


def weighted_binary_crossentropy(y_true, y_pred):
    y_true = K.backend.clip(tf.cast(y_true, dtype=tf.float32), 0.0001, 1)
    y_pred = K.backend.clip(tf.cast(y_pred, dtype=tf.float32), 0.0001, 1)
    weight_positive_class = 1

    return K.backend.mean(
        -y_true * K.backend.log(y_pred + 0.0001) * weight_positive_class - (1 - y_true) * K.backend.log(
            1 - y_pred + 0.0001) * weight_positive_class)

def GenNet_one_hot(inputsize, mask, dense_l1, gene_l1, activation_func_snps = "relu" ):
    ''' The function that creates the neural network. 
    The locallyDirected1D layer let's you define all the connections yourself 
    with the help of a mask, the rest is standard Keras
    The created network is plotted at the end of the notebook'''
    

    inputs_ = K.Input((inputsize,3), name='inputs_')
  
    SNP_layer = K.layers.LocallyConnected1D(filters=1, strides=1, kernel_size=1, 
                                            name="SNP_layer", 
                                           )(inputs_)
    SNP_layer = K.layers.Activation(activation_func_snps, name="snp_relu_activation")(SNP_layer) 

    
    Gene_layer = LocallyDirected1D(mask=mask, filters=2, input_shape=(inputsize, 1), name="gene_layer", 
                                   activity_regularizer=K.regularizers.l1(0.001) 
                                  )(SNP_layer)   
    Gene_layer = K.layers.Activation(activation_func_snps, name="gene_activation")(Gene_layer) #gene layer
    Gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out")(Gene_layer)
    
    Gene_comb = K.layers.LocallyConnected1D(filters=1, kernel_size=1, strides=1,
                                           activity_regularizer=K.regularizers.l1(gene_l1),
                                           name = "gene_combine")(Gene_layer)
    Gene_comb = K.layers.Activation(activation_func_snps, name="gene_c_activation")(Gene_comb) #gene layer
    
    Gene_comb = K.layers.BatchNormalization(center=True, scale=True, name="inter_out2")(Gene_comb)
    
    flat_gene = K.layers.Flatten()(Gene_comb)
    Prediction_layer = K.layers.Dense(units=1,    
                                      kernel_regularizer=K.regularizers.l1(dense_l1),
                                     )(flat_gene)
    Prediction_layer = K.layers.Activation("sigmoid", name="end_node_activation")(Prediction_layer)
    model = K.Model(inputs=inputs_, outputs=Prediction_layer)
    return model
    

def GenNet(inputsize, mask, dense_l1, gene_l1, activation_func_snps  = "relu" ):
    ''' The function that creates the neural network. 
    The locallyDirected1D layer let's you define all the connections yourself 
    with the help of a mask, the rest is standard Keras
    The created network is plotted at the end of the notebook'''

    inputs_ = K.Input((inputsize,), name='inputs_')

    Input_layer = K.layers.Reshape(input_shape=(inputsize,), target_shape=(inputsize, 1))(inputs_)
    
    Gene_layer = LocallyDirected1D(mask=mask, filters=2, input_shape=(inputsize, 1), name="gene_layer", 
                                   activity_regularizer=K.regularizers.l1(0.001) # 00000001
                                  )(Input_layer)   
    Gene_layer = K.layers.Activation(activation_func_snps, name="gene_activation")(Gene_layer) #gene layer
    Gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out")(Gene_layer)
    
    Gene_comb = K.layers.LocallyConnected1D(filters=1, kernel_size=1, strides=1,
                                           activity_regularizer=K.regularizers.l1(gene_l1),
                                           )(Gene_layer)
    Gene_comb = K.layers.Activation(activation_func_snps, name="gene_c_activation")(Gene_comb) #gene layer
    
    Gene_comb = K.layers.BatchNormalization(center=True, scale=True, name="inter_out2")(Gene_comb)
    
    flat_gene = K.layers.Flatten()(Gene_comb)
    Prediction_layer = K.layers.Dense(units=1,    
                                      kernel_regularizer=K.regularizers.l1(dense_l1),
                                     )(flat_gene)
    Prediction_layer = K.layers.Activation("sigmoid", name="end_node_activation")(Prediction_layer)
    model = K.Model(inputs=inputs_, outputs=Prediction_layer)
    return model


def main(jobid,
         simu_name, 
         wpc=1, 
         simtype='test',
         lr_opt=0.01,
         batch_size=128, 
         dense_l1= 0.01,
         gene_l1 = 0,
         activation_func_snps = 'relu',
         gene_interaction = 1,
         sim_id = 1,
         one_hot = 1,
         marginal = 0):

    if marginal == 1:
        marginal = True
    elif marginal == 0:
        marginal = False
    else:
        print("wrong marginal")
    
    if gene_interaction == 1:
        gene_interaction = True
    elif gene_interaction == 0:
        gene_interaction = False
    else:
        print("wrong gene_interaction")
    
    if one_hot == 1:
        one_hot = True
    elif one_hot == 0:
        one_hot = False
    else:
        print("wrong one_hot")
    
    
    patience = 50
    if simtype ==  "Gametes":
        patience = 100
        print(simtype)
        basepath = '/data/scratch/avanhilten/epistasis-simulation/Gametes/simulations/'
        resultpath='/data/scratch/avanhilten/epistasis-simulation/GenNet/results/Gametes/'
        datapath =  basepath + "/" + simu_name + "/"
        sim_id = str(sim_id).zfill(2)
        
    elif simtype == "Epigen":
        print(simtype)
        
        if marginal:
            basepath = '/projects/0/emc17610/Arno/epistasis/simulations/GT_Marginal/'
            resultpath='/projects/0/emc17610/Arno/epistasis/results/Epigen_marginal/'
        else:
            basepath = '/projects/0/emc17610/Arno/epistasis/simulations/No_Marginal/'
            resultpath='/projects/0/emc17610/Arno/epistasis/results/Epigen_no_marginal/'
        sim_id = 1
        datapath =  basepath + "/"
    else:
        print('Simtype', simtype)
    
    print("datapath", datapath)
    print('gene_interaction', gene_interaction)
    
    time_start = time.time()
    SlURM_JOB_ID = get_SLURM_id()
    global weight_positive_class
    
    
    epochs = 1000

    weight_possitive_class = 1
    xtrain, ytrain, xval, yval, xtest, ytest, interacting_snps = get_data(datapath, simu_name, sim_id,
                                                        one_hot=one_hot, 
                                                        simtype = simtype)
    inputsize = xtrain.shape[1]

    if lr_opt == 0:
        optimizer_model = tf.keras.optimizers.Adadelta()
        print("Adadelta")
    else:
        optimizer_model = tf.keras.optimizers.Adam( lr=lr_opt)
        print("Adam", lr_opt)

    if wpc == 1:
        weight_positive_class =  (ytrain.shape[0] - ytrain.sum()) / ytrain.sum()    
    else:
        weight_positive_class = wpc
        
    folder = str(simu_name) + "_" + str(sim_id) + "__" + str(jobid)
    print(folder)
    
    rfrun_path = resultpath + "/" + folder + "/"
    if not os.path.exists(rfrun_path):
        print("Runpath did not exist but is made now")
        os.mkdir(rfrun_path)

    print("new num genes")
    num_genes = 5
    if inputsize < 1000:
        num_genes = 5
    else: 
        num_genes = inputsize // 100
     

    genemask, gene_end = make_mask_gene_layer(inputsize, 
                                              numgenes=num_genes, 
                                              gene_interaction=gene_interaction, 
                                              inter_snps = interacting_snps)  # get the mask to know which SNP connect to which gene
    
    if one_hot:
        modelname = 'GenNet_one_hot'
        model = GenNet_one_hot(inputsize=int(inputsize), mask=genemask, dense_l1=dense_l1,
                                       gene_l1=gene_l1, activation_func_snps=activation_func_snps)
        
        
        
        local = model.layers[1].get_weights()[0]
        local_base = np.array(local[:,0,0], copy=True) 
        for i in range(local.shape[1]):
            local[:,i,0] = local_base + i 

        model.layers[1].set_weights([local, model.layers[1].get_weights()[1]])

    else: 
        modelname = 'GenNet'
        model = GenNet(inputsize=int(inputsize), mask=genemask, dense_l1=dense_l1,
                               gene_l1=gene_l1, activation_func_snps=activation_func_snps)
    model.compile(loss= weighted_binary_crossentropy, optimizer=optimizer_model, metrics=["accuracy", sensitivity, specificity])

    print(model.summary())

    csv_logger = K.callbacks.CSVLogger(rfrun_path + 'train_log.csv', append=True)
    
    early_stop = K.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=patience, verbose=1, mode='auto',
                                           restore_best_weights=True)
    save_best_model = K.callbacks.ModelCheckpoint(rfrun_path + "bestweights_job.h5", monitor='val_loss',
                                                  verbose=1, save_best_only=True, mode='auto')
    
    time_train = time.time()
    
    if os.path.exists(rfrun_path + '/bestweights_job.h5'):
        print('Model already Trained')
        return
    else:
        history = model.fit(x = xtrain, y =ytrain, batch_size=batch_size, epochs=epochs, verbose=1,
                            callbacks=[csv_logger, early_stop, save_best_model], shuffle=True, validation_data=(xval, yval))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(rfrun_path + "train_val_loss.png")
        plt.show()
        
    print("Finished training")
    time_trained = time.time()
    
    print('loading weights')
    model.load_weights(rfrun_path + '/bestweights_job.h5')
    print("evaluate over  patients")
    
    
    ptrain = model.predict(xtrain)
    score_auc_train, confusionmatrix_train  = evaluate_performance(ytrain, ptrain)
    print("AUC train", score_auc_train)
    
    print(confusionmatrix_train)
    

    pval = model.predict(xval) 
    score_auc_val, confusionmatrix_val = evaluate_performance(yval, pval)
    print("AUC val", score_auc_val)
    print(confusionmatrix_val)
    np.save(rfrun_path + "/pval.npy", pval)


    ptest = model.predict(xtest) 
    score_auc_test, confusionmatrix_test = evaluate_performance(ytest, ptest) # get the unbiased predictions
    print("AUC test", score_auc_test)
    print(confusionmatrix_test)
    np.save(rfrun_path + "/ptest.npy", ptest)
    
    scipy.sparse.save_npz(rfrun_path + 'genemask.npz', genemask)

    
    time_end = time.time()
    
        
    
    stats_dict = {'jobid':[jobid], 
                  'sim_id':[sim_id],
                  "simu_name":[simu_name],
                'score_auc_train': [score_auc_train],
                'score_auc_val': [score_auc_val],
                'score_auc_test': [score_auc_test],
                'modelname' : [modelname],
                'gene_interaction' : [gene_interaction],
                'lr_opt': [lr_opt],
                'dense_l1': [dense_l1],  
                'gene_l1': [gene_l1], 
                'batch_size': [batch_size],
                'activation_func_snps':[activation_func_snps],
                'simu_name': [simu_name],
                'weight_possitive_class': [weight_possitive_class],
                'train_size':[xtrain.shape[0]],
                'num_snps': [inputsize], 
                'time_start': [time_start],
                'time_train': [time_train],
                'time_trained': [time_trained],
                'time_end': [time_end],  
                'rfrunpath': [rfrun_path], 
                'SlURM_JOB_ID':[SlURM_JOB_ID],
                }
    
    stats=pd.DataFrame(stats_dict)
    stats.to_csv(rfrun_path + '/run_stats.csv', index=False)
    

if __name__ == '__main__':

    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "-j",
        type=int,
    )
    CLI.add_argument(
        "-wpc",
        type=float,
        default=1
    )
    CLI.add_argument(
        "-lr",
        type=float,
        default=0.0005,
    )
    CLI.add_argument(
        "-bs",
        type=int,
        default=32,
    )
    CLI.add_argument(
        "-dense_l1",
        type=float,
        default=0.01,
    )
    CLI.add_argument(
        "-gene_l1",
        type=float,
        default=0.01,
    )
    CLI.add_argument(
        "-simu_name",
        type=str,
        default='unknown'
    )
    CLI.add_argument(
        "-activation_func_snps",
        type=str,
        default="linear"
    )
    CLI.add_argument(
        "-gene_interaction",
        type=int,
        default=1
    )
    CLI.add_argument(
        "-one_hot",
        type=int,
        default=1
    )
    CLI.add_argument(
        "-simtype",
        type=str,
        default="Gametes"
    )
    CLI.add_argument(
        "-sim_id",
        type=int,
        default=1
    )    
    CLI.add_argument(
        "-marginal",
        type=int,
        default=0
    )      
    args = CLI.parse_args()
    print("jobid: "  + str(args.j))

    main(jobid=args.j,
         wpc=args.wpc,  
         lr_opt=args.lr,
         batch_size=args.bs,
         dense_l1=args.dense_l1, 
         gene_l1=args.gene_l1, 
         simu_name=args.simu_name, 
         activation_func_snps=args.activation_func_snps,
         gene_interaction=args.gene_interaction,
         simtype=args.simtype,
         one_hot = args.one_hot,
         sim_id = args.sim_id,
         marginal = args.marginal
         
        )
   



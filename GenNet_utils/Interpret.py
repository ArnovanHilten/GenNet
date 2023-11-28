import sys
import os
import numpy as np
import time
import tqdm
import pandas as pd
import tensorflow as tf
import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shap
import interpretation.DFIM as DFIM
        
from interpretation.weight_importance import make_importance_values_input
from interpretation.NID import Get_weight_tsang, GenNet_pairwise_interactions_topn

from GenNet_utils.Train_network import load_trained_network
from GenNet_utils.Create_network import remove_batchnorm_model, remove_cov
from GenNet_utils.Dataloader import EvalGenerator

def interpret(args):
    if args.type == 'get_weight_scores':
        get_weight_scores(args)
    elif args.type == 'NID':
        get_NID_scores(args)
    elif args.type == 'RLIPP':
        get_RLIPP_scores(args)
    elif args.type == 'DFIM':
        get_DFIM_scores(args)
    elif args.type == 'pathexplain':
        get_pathexplain_scores(args)
    else:
        print("invalid type:", args.type)
        exit()


def get_weight_scores(args):
    model, masks = load_trained_network(args)

    if os.path.exists(args.resultpath + "/weight_importance.npy"):
        print('weight Done')
    else:
        weight_importance = make_importance_values_input(model, masks=masks)
        np.save(args.resultpath + "/weight_importance.npy", weight_importance)


def get_NID_scores(args):
    
    print("Interpreting with NID:")
    model, masks = load_trained_network(args)

    mask = masks[0]

    if args.layer == None:
        if args.onehot == 1:
            interp_layer = 3
        else:
            interp_layer = 2
    else:
        interp_layer = args.layer

    print("Interrpeting layer", interp_layer)
    if os.path.exists(args.resultpath + "/NID.csv"):
        print('RLIPP Done')
        interaction_ranking = pd.read_csv(args.resultpath + "/NID.csv")
    else:
        print("Obtaining the weights")
        w_in, w_out = Get_weight_tsang(model, interp_layer, masks)
        print("Computing interactions")

        pairwise_interactions_dfs = []
        for filter in range(w_in.shape[1]):  # for all the filters
            pairwise_interactions = GenNet_pairwise_interactions_topn(w_in[:,filter] ,w_out[:,filter], mask, n="auto")
            pairwise_interactions_dfs.append(pairwise_interactions)
        
        interaction_ranking = pd.concat(pairwise_interactions_dfs)
        interaction_ranking = interaction_ranking.sort_values("strength", ascending =False)
        interaction_ranking.to_csv(args.resultpath + "/NID.csv")
        print("NID results are saved in", args.resultpath + "/NID.csv")
  
    return interaction_ranking



def get_RLIPP_scores(args):
    print("not implemented yet")



def get_DFIM_scores(args):
    tf.compat.v1.disable_eager_execution()

    print("Interpreting with DFIM:")
    
    num_snps_to_eval = args.num_eval if hasattr(args, 'num_eval') else 100

    model, masks = load_trained_network(args)

    xval, yval= EvalGenerator(datapath=args.path, genotype_path=args.genotype_path, batch_size=64,
                                          setsize=-1, one_hot=args.onehot,
                                          inputsize=-1, evalset="validation").get_data(sample_pat=args.num_sample_pat)
    xtest, ytest = EvalGenerator(datapath=args.path, genotype_path=args.genotype_path, batch_size=64,
                                          setsize=-1, one_hot=args.onehot,
                                          inputsize=-1, evalset="test").get_data(sample_pat= args.num_sample_pat)


    print("Loaded the data")
    
    model = remove_batchnorm_model(model, masks, keep_cov=False)

    print("compile")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy())

    xval = xval[0]
    xtest = xtest[0]

    explainer  = shap.DeepExplainer((model.input, model.output), xval)
    print("Created explainer")

    if os.path.exists( args.resultpath+ "/shap_test.npy"):
        shap_values = np.load(args.resultpath + "/shap_test.npy")
    else:
        max_axis = (0,2) if args.onehot else 0
        shap_values = np.max(explainer.shap_values(xtest)[0], axis=max_axis)
        np.save(args.resultpath + "/shap_test.npy", shap_values)
    
    print("Find most important SNPs..")
        
    
    snp_num_eval = min(num_snps_to_eval, shap_values.shape[0])
    snp_index = np.argsort(shap_values)[::-1][:snp_num_eval]

    part_suffix = ""
    if (args.start_rank == 0) and (args.end_rank == 0):
        part_suffix = ""
    else:
        part_suffix = "_" + str(args.start_rank) + "_" + str(args.end_rank) + "_"
        snp_index = snp_index[args.start_rank:args.end_rank] 

    print("Evaluate the following variants", snp_index)

    print("Start DFIM for the", snp_num_eval, "most important SNPs -> see ", args.resultpath + "/DFIM_loc_not_perturbed_"
          +str(part_suffix)+".npy", "when finished" )

    perturbed_values, max_not_perturbed, loc_not_perturbed= DFIM.DFIM_test_index(explainer, xtest, snp_index)
    np.save(args.resultpath + "/DFIM_not_perturbed_"+str(part_suffix)+".npy", max_not_perturbed)
    np.save(args.resultpath + "/DFIM_loc_not_perturbed_"+str(part_suffix)+".npy", loc_not_perturbed)
    np.save(args.resultpath + "/DFIM_perturbed_"+str(part_suffix)+".npy", perturbed_values)
    time_DFIM = time.time()
    print("results saved to", args.resultpath)
    
    return
    
def get_pathexplain_scores(args):
    from path_explain import PathExplainerTF

    num_snps_to_eval = args.num_eval if hasattr(args, 'num_eval') else 100

    model, masks = load_trained_network(args)

    print("evaluate over", args.num_sample_pat, "exomples")

    xval, yval= EvalGenerator(datapath=args.path, genotype_path=args.genotype_path, batch_size=64,
                                          setsize=-1, one_hot=args.onehot,
                                          inputsize=-1, evalset="validation").get_data(sample_pat=args.num_sample_pat)
    xtest, ytest = EvalGenerator(datapath=args.path, genotype_path=args.genotype_path, batch_size=64,
                                          setsize=-1, one_hot=args.onehot,
                                          inputsize=-1, evalset="test").get_data(sample_pat= args.num_sample_pat)

    model = remove_cov(model, masks)
    
    xval = xval[0]
    xtest = xtest[0]
    print("Shapes",xval.shape, xtest.shape)

    explainer = PathExplainerTF(model)
    n_top_values = min(num_snps_to_eval, xtest.shape[1])

    if os.path.exists( args.resultpath+ "/pathexplain_snp_index.npy"):
        snp_index = np.load(args.resultpath + "/pathexplain_snp_index.npy")
    else:
        attributions = explainer.attributions(xtest.astype(np.float32), xval.astype(np.float32),
                                batch_size=100, num_samples=args.num_sample_pat,
                                use_expectation=True, output_indices=[0] * len(xtest),
                                verbose=True)
        if args.onehot:
            attr_matrix = np.max(abs(attributions), axis=(0,2))
        else:
            attr_matrix = np.max(abs(attributions), axis=0)

        SNP_indices = np.argsort(attr_matrix)[::-1]
        np.save(args.resultpath + "/pathexplain_snp_index", SNP_indices)


    SNP_indices = SNP_indices[:n_top_values]
    
    part_suffix = ""
    if (args.start_rank == 0) and (args.end_rank == 0):
        part_suffix = ""
    else:
        part_suffix = "_" + str(args.start_rank) + "_" + str(args.end_rank) + "_"
        snp_index = snp_index[args.start_rank:args.end_rank] 

    print("Evaluate the following variants", snp_index)

    interactions = []

    if args.onehot:
        print("using one hot")
        for SNP_index in tqdm.tqdm(SNP_indices):
            interaction_onehot = []
            for one_hot_encoding in range(3):
                interaction_ = explainer.interactions(xtest.astype(np.float32), xval.astype(np.float32),
                        batch_size=100, num_samples=100,
                        use_expectation=True, output_indices=[0] * len(xtest),
                        verbose=True, interaction_index=[SNP_index, one_hot_encoding])
                interaction__ = np.max(abs(interaction_), axis=(0,2))
                interaction_onehot.append(interaction__)

            interactions.append(np.max(np.stack(interaction_onehot), axis=0))

        interactions = np.stack(interactions)
    else:
        for SNP_index in tqdm.tqdm(SNP_indices):               

            interaction = explainer.interactions(xtest.astype(np.float32), xval.astype(np.float32),
                        batch_size=100, num_samples=args.num_sample_pat,
                        use_expectation=True, output_indices=[0] * len(xtest),
                        verbose=True, interaction_index=[SNP_index])
            interaction = np.max(abs(interaction), axis=(0))
            interactions.append(interaction)
        interactions = np.stack(interactions)

    time_pathexplain = time.time()
    print("Saving results...")

    np.save(args.resultpath + "/pathexplain_inter" + part_suffix, interactions)
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from interpretation.weight_importance import make_importance_values_input
from interpretation.NID import Get_weight_tsang, GenNet_pairwise_interactions_topn
from interpretation.RLIPP import calculate_RLIPP

from GenNet_utils.Utility_functions import get_SLURM_id, evaluate_performance
from GenNet_utils.Train_network import load_trained_network


def get_weight_scores(args):
    model, masks = load_trained_network(args)

    if os.path.exists(args.resultpath + "/weight_importance.npy"):
        print('weight Done')
    else:
        weight_importance = make_importance_values_input(model, masks=masks)
        np.save(args.resultpath + "/weight_importance.npy", weight_importance)


def get_NID_scores(args):
    model, masks = load_trained_network(args)

    if one_hot == 1:
        one_hot = True
        interp_tsang_layer = 3
    elif one_hot == 0:
        one_hot = False
        interp_tsang_layer = 2
    else:
        print("wich layer to interpret?")



    if os.path.exists(args.resultpath + "/NID.csv"):
        print('RLIPP Done')
        time_NID = 0
    else:
        w_in, w_out = Get_weight_tsang(model, interp_tsang_layer, genemask)
        interaction_ranking1 = GenNet_pairwise_interactions_topn(w_in[:,1] ,w_out[:,1], masks, n=4)
        interaction_ranking2 = GenNet_pairwise_interactions_topn(w_in[:,0] ,w_out[:,0], masks, n=4)

        interaction_ranking = interaction_ranking1.append(interaction_ranking2)
        interaction_ranking = interaction_ranking.sort_values("strength", ascending =False)
        interaction_ranking.to_csv(args.resultpath + "/NID.csv")
    




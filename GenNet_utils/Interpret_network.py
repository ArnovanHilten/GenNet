import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from interpretation.weight_importance import make_importance_values_input
from interpretation.NID import Get_weight_tsang, GenNet_pairwise_interactions_topn
from interpretation.RLIPP import calculate_RLIPP

from GenNet_utils.Utility_functions import get_SLURM_id, evaluate_performance
from GenNet_utils.Train_network import get_classification_network, get_regression_network

def get_network(args):

    if args.regression = True
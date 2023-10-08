import numpy as np
import pandas as pd
import shap
from deeplift.dinuc_shuffle import dinuc_shuffle
from deeplift.visualization import viz_sequence

import tqdm

from shap.explainers._deep.deep_tf import passthrough
shap.explainers._deep.deep_tf.op_handlers['AddV2'] = passthrough
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

from deeplift.dinuc_shuffle import dinuc_shuffle
from deeplift.visualization import viz_sequence
from joblib import Parallel, delayed

# from deeplift.visualization import viz_sequence

class DFIM():
    
    def __init__(self, model, num_shuffles: int, shuffle_type = "dinuc"):
        self.num_shuffles = num_shuffles
        self.model = model
        self.shuffle_type = shuffle_type

    def shuffle_several_times(self, seqs): # input 3 dim array out 2 dim array
        seqs = np.array(seqs)
        assert len(seqs.shape) == 3

        shuffled = []
        for s in seqs:
            din = dinuc_shuffle(s, num_shufs=self.num_shuffles)
            shuffled.append(din)
        shuffle_out = np.squeeze(np.array(shuffled))
        
        return shuffle_out
    
    
    def shuffle_SNPs(self, seqs):
        seqs = np.array(seqs)
        shuffled = []
              
        for i in range(self.num_shuffles):
            if seqs.ndim == 2:
                shuffled.append(seqs[:, np.random.permutation(seqs.shape[1])])
            elif seqs.ndim == 3:
                shuffled.append(seqs[:, np.random.permutation(seqs.shape[1]),:])
        shuffle_out = np.squeeze(np.array(shuffled))
        return shuffled


    def score(self, seqs_to_explain: np.array):
        """seqs_to_explain: 3d input"""
        
        if self.shuffle_type == "dinuc":
            background_function =  self.shuffle_several_times
        elif self.shuffle_type == "random_SNPs":
            background_function =  self.shuffle_SNPs
        
        dinuc_shuff_explainer = shap.DeepExplainer(self.model,background_function )
        raw_shap_explanations = dinuc_shuff_explainer.shap_values(seqs_to_explain, 
                                                                  check_additivity=False)
        return raw_shap_explanations
        
      
    def visualise(self, seqs_to_explain: np.array, raw_shap_explanations: np.array):
    
        '''project the importance at each position onto the base that's actually present'''
        dinuc_shuff_explanation = np.squeeze(np.sum(raw_shap_explanations,
                                                    axis=-1))[:,:,None]*seqs_to_explain
        
        dinuc_shuff_explanations = np.zeros((raw_shap_explanations[0].shape[0],
                                             raw_shap_explanations[0].shape[1],4))
        
        print("one_hot to TCAG")
        dinuc_shuff_explanations[:,:,:3] = dinuc_shuff_explanation

        print(dinuc_shuff_explanations.shape)
        for dinuc_shuff_explanation in dinuc_shuff_explanations:
            viz_sequence.plot_weights(dinuc_shuff_explanation, subticks_frequency=20)
            
        
def mutation_score_parallel(explainer, seqs_to_explain: np.array, njobs:int):
    
    output_DFIM_og = explainer.score(seqs_to_explain)
    output_DFIM_og = np.mean(output_DFIM_og[0], axis=(0, 2))
    
    with Parallel(require = 'sharedmem', n_jobs=njobs) as parallel:
        results = parallel(delayed(mutation_score_per_position)(
            explainer, input_pos, seqs_to_explain, 
            output_DFIM_og) for input_pos in range(seqs_to_explain.shape[1]))
        
    return pd.DataFrame(results)


def mutation_score(explainer, seqs_to_explain: np.array):

    mut_pos = seqs_to_explain.shape[1]
    
    if seqs_to_explain.ndim == 3:
        axis_mean = (0, 2)
    elif seqs_to_explain.ndim == 2:
        axis_mean = 0
    
    output_DFIM_og = explainer.score(seqs_to_explain)
    output_DFIM_og = np.mean(output_DFIM_og[0], axis=axis_mean)
    
    results = []
    for position in tqdm.tqdm(range(mut_pos)):
        results.append(mutation_score_per_position(explainer, position, seqs_to_explain, output_DFIM_og))

    return pd.DataFrame(results)


def mutation_score_bash(resultpath, part_n,  begin_pos, end_pos, explainer, seqs_to_explain: np.array):

    # replace by mutation_score()
    if seqs_to_explain.ndim == 3:
        axis_mean = (0, 2)
    elif seqs_to_explain.ndim == 2:
        axis_mean = 0

    
    output_DFIM_og = explainer.score(seqs_to_explain)
    output_DFIM_og = np.mean(output_DFIM_og[0], axis=axis_mean)
    
    results = []
    for position in tqdm.tqdm(range(begin_pos, end_pos)):
        result_mut = mutation_score_per_position(explainer, position, seqs_to_explain, output_DFIM_og)
        results.append(result_mut)

    results = pd.DataFrame(results)
    results.to_csv(resultpath + "/" + str(part_n)+ ".csv" )
    
    
def mutation_score_per_position(explainer, position, seqs_to_explain, output_DFIM_og):
    if seqs_to_explain.ndim == 2:
        return mutation_score_pos_geno(explainer, position, seqs_to_explain, output_DFIM_og)
    elif seqs_to_explain.ndim == 3:
        return mutation_score_pos_onehot(explainer, position, seqs_to_explain, output_DFIM_og)
    else:
        print("dimensions do not correspond")
        
def mutation_score_pos_onehot(explainer, position, seqs_to_explain, output_DFIM_og):
    nchannels = seqs_to_explain.shape[2]
    npositions = seqs_to_explain.shape[1]

    score_array = np.zeros((npositions, nchannels))

    for channel in range(nchannels):
        # this is necessary
        mutated_seq = seqs_to_explain.copy()

        # change the one hot encoding one by one for all posibilities
        mutated_seq[:,position, :] = np.zeros(3)
        mutated_seq[:,position, channel] = 1

        # create a new instance of DFIM with the model and num_shuffles attributes
#         explainer = DFIM(model, nshuffles)

        # comput the output score for each position
        output_DFIM = explainer.score(mutated_seq)

        # fill score per channel with the difference to the original unmutated
        score_array[:,channel] = np.mean(output_DFIM[0], axis=(0, 2)) - output_DFIM_og

    # calculate max difference in FIS
    max_score_pos = np.max(score_array, axis=1)
    max_score = np.max(max_score_pos)
    interacting_snp = np.argmax(max_score_pos)

    return {'mutated position':position, 'highest FIS pos':interacting_snp ,"FIS value":max_score}


def mutation_score_pos_geno(explainer, position, seqs_to_explain, output_DFIM_og):
    npositions = seqs_to_explain.shape[1]
    score_array = np.zeros((npositions, 3))

    for mut_genotype in range(3):
        mutated_seq = seqs_to_explain.copy()
        mutated_seq[:,position] = mut_genotype
        # comput the output score for each position
        output_DFIM = explainer.score(mutated_seq)

        # fill score per channel with the difference to the original unmutated
        score_array[:,mut_genotype] = np.mean(output_DFIM[0], axis=0) - output_DFIM_og

    # calculate max difference in FIS
    max_score_pos = np.max(score_array, axis=1)
    max_score_pos[position] = 0
    max_score = np.max(max_score_pos)
    interacting_snp = np.argmax(max_score_pos)


    return {'mutated position':position, 'highest FIS pos':interacting_snp ,"FIS value":max_score}
         
        
    
    




    
    

            
        
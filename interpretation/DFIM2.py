import numpy as np
import pandas as pd
import shap

from shap.explainers._deep.deep_tf import passthrough
shap.explainers._deep.deep_tf.op_handlers['AddV2'] = passthrough
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

from deeplift.dinuc_shuffle import dinuc_shuffle
from deeplift.visualization import viz_sequence
from joblib import Parallel, delayed

# from deeplift.visualization import viz_sequence

class DFIM():
    
    def __init__(self, model, nshuffles: int):
        self.num_shuffles = nshuffles
        self.model = model


    def shuffle_several_times(self, seqs): # input 3 dim array out 2 dim array
        seqs = np.array(seqs)
        assert len(seqs.shape) == 3

        shuffled = []

        for s in seqs:
            din = dinuc_shuffle(s, num_shufs=self.num_shuffles)
            shuffled.append(din)

        shuffle_out = np.squeeze(np.array(shuffled))
        return shuffle_out


    def score(self, seqs_to_explain: np.array):
        """seqs_to_explain: 3d input"""
        
        assert seqs_to_explain.ndim == 3
        
        dinuc_shuff_explainer = shap.DeepExplainer(self.model, self.shuffle_several_times)
        self.raw_shap_explanations = dinuc_shuff_explainer.shap_values(seqs_to_explain, check_additivity=False)
        return self.raw_shap_explanations
        
      
   
    def visualise(self, seqs_to_explain: np.array, raw_shap_explanations: np.array):
    
        '''project the importance at each position onto the base that's actually present'''
        dinuc_shuff_explanation = np.squeeze(np.sum(raw_shap_explanations,axis=-1))[:,:,None]*seqs_to_explain
        dinuc_shuff_explanations = np.zeros((raw_shap_explanations[0].shape[0],raw_shap_explanations[0].shape[1],4))
        
        print("one_hot to TCAG")
        dinuc_shuff_explanations[:,:,:3] = dinuc_shuff_explanation

        print(dinuc_shuff_explanations.shape)
        for dinuc_shuff_explanation in dinuc_shuff_explanations:
            viz_sequence.plot_weights(dinuc_shuff_explanation, subticks_frequency=20)
            
            

            


def mutation_score(model, num_shuffles, seqs_to_explain: np.array, njobs:int):
    results = Parallel(n_jobs=njobs)(delayed(mutation_score_pos)(
        model, num_shuffles, input_pos, seqs_to_explain) for input_pos in tqdm.tqdm(range(seqs_to_explain.shape[1])))

    return results



def mutation_score_pos(model, num_shuffles, position, seqs_to_explain):
    nchannels = seqs_to_explain.shape[2]
    npositions = seqs_to_explain.shape[1]

    score_array = np.zeros((npositions,nchannels))

    for channel in range(nchannels):
        # copy just to be sure
        mutated_seq = seqs_to_explain.copy()

        # change the one hot encoding one by one for all posibilities
        mutated_seq[:,position, :] = np.zeros(3)
        mutated_seq[:,position, channel] = 1

        # create a new instance of DFIM with the model and num_shuffles attributes
        expainer = DFIM(model, num_shuffles)

        # comput the output score for each position
        output_DFIM = expainer.score(mutated_seq)

        # fill score per channel with the difference to the original unmutated
        score_array[:,channel] = np.mean(output_DFIM[0], axis=(0, 2)) - output_DFIM_og

    # calculate max difference in FIS
    max_score_pos = np.max(score_array, axis=1)
    max_score = np.max(max_score_pos)
    interacting_snp = np.argmax(max_score_pos)

    return {'max_score': max_score, 'interacting_snp': interacting_snp}




import numpy as np
import tqdm
import shap


def DFIM_test_index(explainer, xtest, index_snps):
    
    # Lists to hold the maximum differences
    perturbed_values = []
    max_not_perturbed = []
    loc_not_perturbed = []
    
    onehot = False if len(xtest.shape) == 2 else True

    for i in tqdm.tqdm(index_snps):  # Assuming second dimension of xtest is features
        variant_shap_values = []

        if onehot == True:
            for SNP_perm in range(3):
                xtest_copy = xtest.copy()
                xtest_copy[:,i, :] = np.zeros(3)
                xtest_copy[:,i, SNP_perm] = 1
        
                shap_values = np.max(explainer.shap_values(xtest_copy)[0], axis=2)
                variant_shap_values.append(shap_values)
        else:
            for SNP_perm in range(3):
                xtest_copy = xtest.copy()
                xtest_copy[:, i] = SNP_perm
                shap_values = explainer.shap_values(xtest_copy)[0]
                variant_shap_values.append(shap_values)

        # Calculate differences between the perturbations for the i-th feature
        differences = []
        for idx1 in range(len(variant_shap_values)):
            for idx2 in range(idx1 + 1, len(variant_shap_values)):
                differences.append(np.abs(variant_shap_values[idx1] - variant_shap_values[idx2]))

        # Stack the differences to get a 3D array of shape (number_of_comparisons, number_of_samples, number_of_features)
        stacked_differences = np.stack(differences)
        stacked_differences = np.mean(stacked_differences, axis=1)
        stacked_differences = np.max(stacked_differences, axis=0)

        # Find the max difference for the perturbed feature (i-th feature)
        perturbed_values.append(stacked_differences[i])

        # Find the max difference for all other features (not i-th feature)
        # We mask the i-th feature to exclude it from the calculation
        stacked_differences[i] = 0
        max_not_perturbed.append(np.max(stacked_differences))
        loc_not_perturbed.append(np.argmax(stacked_differences))

    
    return perturbed_values, max_not_perturbed, loc_not_perturbed



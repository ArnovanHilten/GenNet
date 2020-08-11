
import numpy as np
import os
import sys
from .tools import Timer, timer, timing,save_parameters
import scipy.linalg.blas as FB
import h5py
import gc
import tables


#@timing
def A_covariates(covariates, intercept=True):

	'''
	:param covariates: (n_subjects, n_covariates) - only constant covariates should be included (age, sex, ICV etc)
	:param intercept: default True, add intercept to model
	:return: matrix (n_cavariates, n_covariates), constant part for the rest of the study
	'''

	S,N=covariates.shape
	if intercept:
		I=np.ones(S).reshape(S,1)
		covariates=np.hstack((I,covariates))
	a_cov=np.dot(covariates.T,covariates)
	return a_cov
#@timing
def B4(phenotype,genotype):
	b4=np.tensordot(genotype, phenotype, axes=([1], [0]))
	return b4

def interaction(genotype,factor):
	g=genotype*factor.T
	return g

#@timing
def A_tests(covariates, genotype, intercept=True): #TODO (low) extend for any number of tests in model
	'''
	:param covariates: (n_subjects, n_covariates) - only constant covariates should be included (age, sex, ICV etc)
	:param genotype: (n_tests, n_subjects) - test could be any kind of quantitative covariance
	:return: (1,n_covariates + intercept)
	'''

	if  intercept:
		fst=np.sum(genotype, axis=1).reshape(-1,1)
		sec=np.dot(genotype, covariates)
		tr=np.sum(np.power(genotype,2), axis=1).reshape(-1,1)
		return np.hstack((fst, sec, tr))

	else:
		sec=np.dot(genotype, covariates)
		tr=np.sum(np.power(genotype,2), axis=1).reshape(-1,1)
		return np.hstack(( sec, tr))

#@timing
def B_covariates(covariates, phenotype, intercept=True):

	S,N=covariates.shape

	b_cov = np.dot(covariates.T, phenotype)
	if intercept:
		b1 = np.sum(phenotype, axis=0).reshape(1, phenotype.shape[1])
		B13 = np.append(b1, b_cov, axis=0)
		return B13
	else:
		return b_cov


#@timing
def A_inverse(a_covariates, a_test): #TODO (low) extend for any number of tests in model

	A_inv=[]
	n,m=a_covariates.shape
	k=n+1
	for i in range(a_test.shape[0]): #TODO (low) not in for loop
		inv=np.zeros(k*k).reshape(k,k)
		inv[ 0:k-1,0:k-1 ]=a_covariates
		inv[k-1,:]=a_test[i,:]
		inv[0:k,k-1]=a_test[i,0:k]
		try:
			A_inv.append(np.linalg.inv(inv))
		except:
			A_inv.append(np.zeros(k*k).reshape(k,k)) #TODO (high) test; check influence on results; warning;

	return np.array(A_inv)

#@timing
def C_matrix(phenotype):
	C=np.einsum('ij,ji->i', phenotype.T, phenotype)
	return C

#@timing
#@save_parameters
def HASE(b4, A_inverse, b_cov, C, N_con, DF):

	with Timer() as t:

		B13=b_cov
		B4 = b4

		A1_B_constant = np.tensordot(A_inverse[:, :, 0:(N_con)], B13, axes=([2], [0]))

		A1_B_nonconstant = np.einsum('ijk,il->ijl', A_inverse[:, :, N_con:N_con+1], B4)

		A1_B_full = A1_B_constant + A1_B_nonconstant

		BT_A1B_const = np.einsum('ij,lji->li', B13.T, A1_B_full[:, 0:(N_con), :])

		BT_A1B_nonconst = np.einsum('ijk,ijk->ijk', B4[:, None, :], A1_B_full[:, (N_con):N_con+1, :])

		BT_A1B_full = BT_A1B_const[:, None, :] + BT_A1B_nonconst

		C_BTA1B = BT_A1B_full - C.reshape(1, -1)

		C_BTA1B = np.abs(C_BTA1B)

		a44_C_BTA1B = C_BTA1B * A_inverse[:, (N_con):N_con+1, (N_con):N_con+1]

		a44_C_BTA1B = np.sqrt( (a44_C_BTA1B) )

		t_stat = np.sqrt(DF) * np.divide(A1_B_full[:, (N_con):N_con+1, :], a44_C_BTA1B)

		SE =  a44_C_BTA1B/np.sqrt(DF)


	print("time to compute GWAS for {} phenotypes and {} SNPs .... {} sec".format(b4.shape[1],
																				  A_inverse.shape[0],
																				  t.secs))
	return t_stat, SE










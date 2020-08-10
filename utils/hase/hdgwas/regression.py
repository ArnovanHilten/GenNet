import h5py
import numpy as np
import pandas as pd
import os
import sys
import gc
from hdgwas.tools import study_indexes, Mapper, HaseAnalyser, Timer,merge_genotype
from hdgwas.data import MetaParData
from hdgwas.hdregression import HASE, A_covariates, A_tests, B_covariates, C_matrix, A_inverse,B4
from scipy import stats
import bitarray as ba
from hdgwas.pard import partial_derivatives
import tables




def haseregression(phen,gen,cov, mapper, Analyser, maf,intercept=True, interaction=None):

	g=tuple( [i.folder._data for i in gen ] )

	row_index, ids =  study_indexes(phenotype=phen.folder._data,
											   genotype=g,
											   covariates=cov.folder._data)

	if mapper is not None:
		SNP=[0,0,mapper.n_keys]
	else:
		SNP=[0,0,'unknown']

	covariates=cov.get_next(index=row_index[2])
	a_cov=A_covariates(covariates,intercept=intercept)

	while True:
		gc.collect()
		if mapper is not None:
			if mapper.cluster=='n':
				SNPs_index, keys=mapper.get()
			else:
				ch=mapper.chunk_pop()
				if ch is None:
					SNPs_index=None
					break
				SNPs_index, keys=mapper.get(chunk_number=ch)
			if isinstance(SNPs_index, type(None)):
				break
			Analyser.rsid=keys
		else:
			SNPs_index=None

		with Timer() as t:
			genotype=merge_genotype(gen, SNPs_index, mapper)
		print(('time to read and merge genotype {}s'.format(t.secs)))
		gc.collect()
		if genotype is None:
			print('All genotype processed!')
			break
		SNP[0]+=genotype.shape[0]
		genotype=genotype[:,row_index[0]]

		if mapper is None:
			Analyser.rsid=np.array(list(range(genotype.shape[0])))


		MAF=np.mean(genotype, axis=1)/2
		STD=np.std(genotype, axis=1)

		if maf!=0:

			filter=(MAF>maf) & (MAF<1-maf) & (MAF!=0.5)
			genotype=genotype[filter,:]
			Analyser.MAF=MAF[filter]
			Analyser.rsid=Analyser.rsid[filter]

			if genotype.shape[0]==0:
				print('NO SNPs > MAF')
				continue

		else:
			Analyser.MAF=MAF

		SNP[1]+=genotype.shape[0]

		while True:
			phenotype=phen.get_next(index=row_index[1])

			if isinstance(phenotype, type(None)):
				phen.folder.processed=0
				print('All phenotypes processed!')
				break

			if phen.permutation:
				np.random.shuffle(phenotype)

			b_cov=B_covariates(covariates,phenotype,intercept=intercept)

			C=C_matrix(phenotype)

			if interaction is not None:
				pass


			a_test=A_tests(covariates,genotype,intercept=intercept)
			a_inv=A_inverse(a_cov,a_test)

			N_con=a_inv.shape[1] - 1

			DF=(phenotype.shape[0] - a_inv.shape[1])

			b4=B4(phenotype,genotype)


			t_stat, SE=HASE(b4, a_inv, b_cov, C, N_con, DF)
			print(('Read {}, processed {}, total {}'.format(SNP[0],SNP[1],SNP[2] )))
			Analyser.t_stat=t_stat
			Analyser.SE=SE
			if mapper is not None and mapper.cluster == 'y':
				Analyser.cluster=True
				Analyser.chunk=ch
				Analyser.node=mapper.node[1]
			if phen.permutation:
				Analyser.permutation=True
			Analyser.save_result( phen.folder._data.names[phen.folder._data.start:phen.folder._data.finish])
			t_stat=None
			Analyser.t_stat=None
			del b4
			del C
			del b_cov
			del a_inv
			del a_test
			del t_stat
			gc.collect()

	if Analyser.cluster:
		np.save(os.path.join(Analyser.out,str(Analyser.node)+'_node_RSID.npy'),Analyser.rsid_dic)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAPPER_CHUNK_SIZE, basedir,CONVERTER_SPLIT_SIZE, PYTHON_PATH
os.environ['HASEDIR']=basedir
if PYTHON_PATH is not None:
	for i in PYTHON_PATH: sys.path.insert(0,i)
import h5py
import tables
from hdgwas.tools import Timer,HaseAnalyser, Reference
import argparse
import pandas as pd
import numpy as np
from collections import OrderedDict


if __name__=="__main__":

	os.environ['HASEDIR']=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	parser = argparse.ArgumentParser(description='Script analyse results of HASE')
	parser.add_argument("-r", required=True,help="path to hase results")
	parser.add_argument("-o", "--out", type=str, required=True,help="path to save result folder")
	parser.add_argument("-df", type=float,default=None, help="degree of freedom = ( #subjects in study  - #covariates - 1 )")
	parser.add_argument("-N", type=int,default=None, help="file number to read")
	#TODO (low) add reference panel
	args = parser.parse_args()
	Analyser=HaseAnalyser()
	print(args)

	Analyser.DF=args.df
	Analyser.result_path=args.r
	Analyser.file_number = args.N

	results=OrderedDict()
	results['RSID']=np.array([])
	results['p_value']=np.array([])
	results['t-stat']=np.array([])
	results['phenotype']=np.array([])
	results['SE']=np.array([])
	results['MAF']=np.array([])
	results['BETA'] = np.array([])

	while True:
		Analyser.summary()
		if Analyser.results is None:
			break
		print('Saving data...')
		if not os.path.exists(os.path.join(args.out,'results'+'.csv')):
			df=pd.DataFrame.from_dict(results)
			df.to_csv( os.path.join(args.out,'results'+'.csv'), sep=" ", index=None  )
		df=pd.DataFrame.from_dict(Analyser.results)
		with open(os.path.join(args.out,'results'+'.csv'), 'a') as f:
			df.to_csv(f, sep=" ",header=False,index=None)



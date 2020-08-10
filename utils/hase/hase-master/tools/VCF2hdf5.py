
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PYTHON_PATH
if PYTHON_PATH is not None:
	for i in PYTHON_PATH: sys.path.insert(0,i)
import argparse
import h5py
import pandas as pd
import numpy as np
from hdgwas.tools import Timer
import tables
import glob


def probes_VCF2hdf5(data_path, save_path,study_name, chunk_size=1000000):

	if os.path.isfile(os.path.join(save_path,'probes',study_name+'.h5')):
		os.remove(os.path.join(save_path,'probes',study_name+'.h5'))

	hash_table={'keys':np.array([],dtype=np.int),'allele':np.array([])}

	df=pd.read_csv(data_path,sep='\t',chunksize=chunk_size, header=None,index_col=None)
	for i,chunk in enumerate(df):
		print('add chunk {}'.format(i))
		print(chunk.head())
		chunk.columns=[ "CHR","bp" ,"ID",'allele1','allele2','QUAL','FILTER','INFO'] #TODO (high) parse INFO
		hash_1=chunk.allele1.apply(hash)
		hash_2=chunk.allele2.apply(hash)
		k,indices=np.unique(np.append(hash_1,hash_2),return_index=True)
		s=np.append(chunk.allele1,chunk.allele2)[indices]
		ind=np.invert(np.in1d(k,hash_table['keys']))
		hash_table['keys']=np.append(hash_table['keys'],k[ind])
		hash_table['allele']=np.append(hash_table['allele'],s[ind])
		chunk.allele1=hash_1
		chunk.allele2=hash_2
		chunk.to_hdf(os.path.join(save_path,'probes',study_name+'.h5'),data_columns=["CHR","bp" ,"ID",'allele1','allele2'], key='probes',format='table',append=True,
			 min_itemsize = 25, complib='zlib',complevel=9 )
	pd.DataFrame.from_dict(hash_table).to_csv(os.path.join(save_path,'probes',study_name+'_hash_table.csv.gz'),index=False,compression='gzip', sep='\t')

def ind_VCF2hdf5(data_path, save_path,study_name):

	if os.path.isfile(os.path.join(save_path,'individuals',study_name+'.h5')):
		os.remove(os.path.join(save_path,'individuals',study_name+'.h5'))
	n=[]
	f=open(data_path,'r')
	for i,j in enumerate(f):
		n.append((j[:-1]))
	f.close()
	n=np.array(n)
	chunk=pd.DataFrame.from_dict({"individual":n})
	chunk.to_hdf(os.path.join(save_path,'individuals',study_name+'.h5'), key='individuals',format='table',
				 min_itemsize = 25, complib='zlib',complevel=9 )

def genotype_VCF2hdf5(data_path,id, save_path, study_name):


	df=pd.read_csv(data_path, header=None, index_col=None,sep='\t', dtype=np.float16)
	data=df.as_matrix()
	print(data.shape)
	print('Saving chunk...{}'.format(os.path.join(save_path,'genotype',str(id)+'_'+study_name+'.h5')))
	h5_gen_file = tables.open_file(
		os.path.join(save_path,'genotype',str(id)+'_'+study_name+'.h5'), 'w', title=study_name)

	atom = tables.Float16Atom()
	genotype = h5_gen_file.create_carray(h5_gen_file.root, 'genotype', atom,
										(data.shape),
										title='Genotype',
										filters=tables.Filters(complevel=9, complib='zlib'))
	genotype[:] = data
	h5_gen_file.close()
	os.remove(data_path)


if __name__=="__main__":

	parser = argparse.ArgumentParser(description='Script to convert VCF data')
	parser.add_argument("-study_name", required=True, type=str, help="Study specific name")
	parser.add_argument("-id", type=str, help="subject id")
	parser.add_argument("-data",required=True, type=str, help="path to file")
	parser.add_argument("-out",required=True, type=str, help="path to results save folder")
	parser.add_argument("-flag",required=True,type=str,choices=['individuals','probes','chunk'], help="path to file with SNPs info")


	args = parser.parse_args()

	print(args)
	try:
		print ('Creating directories...')
		os.mkdir(os.path.join(args.out,'genotype') )
		os.mkdir(os.path.join(args.out,'individuals') )
		os.mkdir(os.path.join(args.out,'probes') )
		os.mkdir(os.path.join(args.out,'tmp_files'))
	except:
		print(('Directories "genotype","probes","individuals" are already exist in {}...'.format(args.out)))

	if args.flag=='probes':
		probes_VCF2hdf5(args.data, args.out, args.study_name)
	elif args.flag=='individuals':
		ind_VCF2hdf5(args.data, args.out,args.study_name)
	elif args.flag=='chunk':
		genotype_VCF2hdf5(args.data,args.id, args.out,args.study_name)




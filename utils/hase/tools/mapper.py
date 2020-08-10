import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
if PYTHON_PATH is not None:
	for i in PYTHON_PATH: sys.path.insert(0,i)
import h5py
import pandas as pd
import numpy as np
import argparse
from hdgwas.tools import Reference, Mapper, Timer
from hdgwas.hash import *
import gc

if __name__=='__main__':

	os.environ['HASEDIR']=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

	parser = argparse.ArgumentParser(description='Script to map studies for meta-stage')
	parser.add_argument("-g",required=True, type=str, help="path/paths to genotype data folder")
	parser.add_argument('-study_name',type=str,required=True, default=None, help=' Study names')
	parser.add_argument("-o", "--out", type=str,required=True, help="path to save result folder")
	parser.add_argument('-ref_name', type=str, default='1000Gp1v3_ref', help='Reference panel name')
	parser.add_argument('-mismatch_table',action='store_true',default=False, help='Save table with mismatch IDs')
	parser.add_argument('-flipped_table',action='store_true',default=False, help='Save table with mismatch IDs')
	parser.add_argument('-probe_chunk',type=int,default=10000, help='Probes chunk')
	parser.add_argument('-ref_chunk',type=int,default=10000, help='Reference chunk')
	parser.add_argument('-chunk',type=int,default=2000000, help='Chunk size')
	args = parser.parse_args()
	print(args)

	try:
		print ('Creating directories...')
		os.mkdir(args.out)
	except:
		print(('Directory {} is already exist!'.format(args.out)))

	probes=pd.HDFStore(os.path.join(args.g,'probes', args.study_name+'.h5'),'r')
	probes_n_rows=probes.get_storer('probes').nrows
	chunk_size = np.min([args.chunk,probes_n_rows])

	print(('Merge chunk size {}'.format(chunk_size)))
	match_key=np.array([],dtype=np.int32)
	match_index=np.array([],dtype=np.int32)
	flip_key=np.array([],dtype=np.int32)
	flip_index=np.array([],dtype=np.int32)
	ID=np.array([])

	del_counter_ref={}
	ID2CHR=False
	IDconv=False
	hashing=False
	merge_on={
			'ID':{
				'straight':["ID",'allele1','allele2'],
				  'reverse':["ID",'allele2','allele1']
				 },

			'CHR':{
				 'straight':["CHR",'bp','allele1','allele2'],
				 'reverse':["CHR",'bp','allele2','allele1']

			      }
		  	}

	for p in range(int(np.ceil(probes_n_rows / float(chunk_size)))):
		print('p',p)

		p_start_i = p * chunk_size
		p_stop_i  = min((p + 1) * chunk_size, probes_n_rows)

		a = probes.select('probes', start = p_start_i, stop = p_stop_i)

		if p==0:
			print(a.head())
			if issubclass(type(a.iloc[0]['allele1']), np.str):
				hashing=True
			if "CHR" in a.columns and 'bp' in a.columns:
				ID2CHR=True
				merge=merge_on['CHR']
				print ('Merge on CHR/bp')
			else:
				if ':' in a.ID.iloc[0] and ':' in a.ID.iloc[1]:
					CHR=[]
					bp=[]
					for i in a.ID:
						s=i.split(":")
						CHR.append(s[0])
						bp.append(s[1])
					CHR=np.array(CHR,dtype=np.int8)
					bp=np.array(bp)
					if np.max(CHR)<23 and np.min(CHR)>0:
						a['CHR']=CHR
						a['bp']=bp
						a.CHR = a.CHR.astype(np.int64)
						a.bp = a.bp.astype(np.int64)
						ID2CHR=True
						IDconv=True
						merge=merge_on['CHR']
						print ('Merge on CHR/bp from ID')
						print(a.head())
					else:
						print('No CHR and bp info...')
						merge=merge_on['ID']
						print ('Merge on ID')
				else:
					print('No CHR and bp info...')
					merge=merge_on['ID']
					print ('Merge on ID')

		elif IDconv:
			def f(x):
				s=x.ID.split(':')
				return s[0],s[1]
			CHR_bp=a.apply(f, axis=1 )
			a['CHR'],a['bp']=list(zip(*CHR_bp))
			a.CHR=a.CHR.astype(np.int64)
			a.bp= a.bp.astype(np.int64)
			print(a.head())
		a['counter_prob']=np.arange(p_start_i,p_stop_i,dtype='int32')

		reference=Reference()
		reference.name=args.ref_name
		reference.chunk=args.ref_chunk
		reference.load()
		counter_ref=0
		if hashing:
			print('Hashing...')
			a.allele1=a.allele1.apply(hash)
			a.allele2=a.allele2.apply(hash)
		for r,b in enumerate(reference.dataframe):
			if r==0:
				if np.sum(np.array([ 1 if i in reference.columns else 0 for i in b.columns.tolist()  ]))!=len(reference.columns):
					raise ValueError('Reference table should have {} columns'.format(reference.columns))
			if r==0 and p==0:
				print ('********************************')
				print(('Use {} as a reference panel'.format(args.ref_name)))
				print(b.head())
				print ('********************************')

			print('r',r)
			if p==0:
				ID=np.append(ID,b.ID)

			b['counter_ref']=np.arange(counter_ref,counter_ref+b.shape[0],dtype='int32')
			counter_ref+=b.shape[0]

			if len(match_index) or len(flip_index):
				print('matched {}'.format(match_index.shape[0]))
				print('flipped {}'.format(flip_index.shape[0]))
				if del_counter_ref.get(r) is not None:
					with Timer() as t:
						b=b[~b.counter_ref.isin(del_counter_ref[r])]
					print('time {}'.format(t.secs))

			match_df = pd.merge(b,a, left_on=merge['straight'], right_on=merge['straight'])
			flip_df=pd.merge(b[~b.counter_ref.isin(match_df.counter_ref)],a, left_on=merge['reverse'], right_on=merge['straight'])

			if len(match_df):
				match_key=np.append(match_key,match_df.counter_ref)
				match_index=np.append(match_index,match_df.counter_prob)
				if del_counter_ref.get(r) is None:
					del_counter_ref[r]=match_key
				else:
					del_counter_ref[r]=np.append(del_counter_ref[r], match_key)
			if len(flip_df):
				flip_key=np.append(flip_key,flip_df.counter_ref)
				flip_index=np.append(flip_index,flip_df.counter_prob)
				if del_counter_ref.get(r) is None:
					del_counter_ref[r]=flip_key
				else:
					del_counter_ref[r]=np.append(del_counter_ref[r], flip_key)
			gc.collect()

	index=np.ones(ID.shape[0],dtype='int')*-1
	flip=np.ones(probes_n_rows,dtype='int')
	index[match_key]=match_index
	index[flip_key]=flip_index
	flip[flip_index]=-1
	print(('Saving results for {} to {} ...'.format(args.study_name,args.out)))
	np.save(os.path.join(args.out,'values_'+reference.name+'_'+args.study_name+'.npy'),index)
	np.save(os.path.join(args.out,'flip_'+reference.name+'_'+args.study_name+'.npy'),flip)
	np.save(os.path.join(args.out,'keys_'+reference.name+'.npy'),ID)
	print ('Data successfully saved')

	mismatch_index=np.setdiff1d(np.arange(probes_n_rows),np.append(match_index,flip_index) )

	if os.path.isfile(os.path.join(args.g,'probes', args.study_name+'_hash_table.csv.gz')):
		try:
			df_hash=pd.read_csv(os.path.join(args.g,'probes', args.study_name+'_hash_table.csv.gz'),sep='\t', compression='gzip', index_col=False)
		except:
			df_hash=pd.read_csv(os.path.join(args.g,'probes', args.study_name+'_hash_table.csv.gz'),sep='\t', index_col=False)


	else:
		df_hash=None
		print(('You do not have hash_table for alleles in your probes folder! '
			   'You used old version of HASE to convert your genotype data.'
			   'To see original codes for allele you can make hash_table using script'
			   '{}/tools/tools.py -hash -g "original genotype folder" '.format(os.environ['HASEDIR'])))

	print('There are {} common variances with reference panel, which will be included in study'.format(np.where(index!=-1)[0].shape[0] ))
	print('There are {} variances from reference panel, which were not found in probes'.format(np.where(index==-1)[0].shape[0] ))
	print('There are {} variances excluded from study (not found in reference panel)'.format( probes_n_rows-np.where(index!=-1)[0].shape[0]  ))
	if args.mismatch_table and mismatch_index.shape[0]!=0:
		df_mismatch=probes.select('probes',where=mismatch_index)
		if df_hash is not None and not hashing:
			df_mismatch=pd.merge(df_hash,df_mismatch,left_on='keys', right_on='allele1')
			df_mismatch['str_allele1']=df_mismatch['allele']
			del df_mismatch['allele']
			df_mismatch=pd.merge(df_hash,df_mismatch,left_on='keys', right_on='allele2')
			df_mismatch['str_allele2']=df_mismatch['allele']
			del df_mismatch['allele']
			del df_mismatch['keys_x']
			del df_mismatch['keys_y']
		df_mismatch.to_csv(os.path.join(args.out,'mismatch_ID_info.csv'))
		print('Mismatch ID info saved to {}'.format(os.path.join(args.out,args.study_name+'_mismatch_ID_info.csv')))
	elif mismatch_index.shape[0]!=0:
		print ('Mismatch examples:')
		print(probes.select('probes',where=mismatch_index[:10]))

	print('There are {} flipped variances'.format(len(flip_index)))
	if args.flipped_table and flip_index.shape[0]!=0:
		df_flipped=probes.select('probes',where=flip_index)
		if df_hash is not None and not hashing:
			df_flipped=pd.merge(df_hash,df_flipped,left_on='keys', right_on='allele1')
			df_flipped['str_allele1']=df_flipped['allele']
			del df_flipped['allele']
			df_flipped=pd.merge(df_hash,df_flipped,left_on='keys', right_on='allele2')
			df_flipped['str_allele2']=df_flipped['allele']
			del df_flipped['allele']
			del df_flipped['keys_x']
			del df_flipped['keys_y']
		df_flipped.to_csv(os.path.join(args.out,'flipped_ID_info.csv'))
		print('Flipped ID info saved to {}'.format(os.path.join(args.out,args.study_name + '_flipped_ID_info.csv')))
	elif flip_index.shape[0]!=0:
		print ('Flipped examples:')
		print(probes.select('probes',where=flip_index[:10]))






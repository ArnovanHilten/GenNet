import sys
import h5py
from  hdgwas.tools import Timer,Mapper, merge_genotype
import os
from hdgwas.data import Reader
import argparse
import gc
import tables
import numpy as np




parser = argparse.ArgumentParser(description='Script to merge genotype')
parser.add_argument("-g", "--genotype",nargs='+', type=str, help="path/paths to genotype data folder")
parser.add_argument('-mapper', type=str, help='Mapper data folder')
parser.add_argument('-mapper_name', type=str, help='Mapper name')
parser.add_argument("-o", "--out", type=str, required=True, help="path to save result folder")
parser.add_argument("-save_name", type=str, required=True, help="merge study name")
parser.add_argument('-study_name', type=str, required=True,nargs='+', help=' Name for saved genotype data, without ext')

parser.add_argument('-cluster', type=str, default='n', choices=['y','n'], help=' Is it parallel cluster job, default no')
parser.add_argument('-node', nargs='+',help='number of nodes / this node number, example: 10 2 ')
parser.add_argument('-split',type=int,help='Split size for merge genotypes')


args = parser.parse_args()
print(args)

if __name__ == '__main__':

	print ('Not implemented!')

	# mapper=Mapper(args.mapper_name)
	# mapper.load(args.mapper)
	# mapper.chunk_size=args.split
    #
    #
	# hdf5_iter=0
	# h5_name=args.save_name
	# pytable_filter=tables.Filters(complevel=9, complib='zlib')
	# gen=[]
	# for i,j in enumerate(args.genotype):
	# 	gen.append(Reader('genotype'))
	# 	gen[i].start(j,hdf5=True, study_name=args.study_name[i], ID=False)
    #
	# RSID=[]
	# SUB_ID=[]
	# for i in gen:
	# 	SUB_ID.append(i.folder._data.get_id())
	# mapper.cluster=args.cluster
	# mapper.node=args.node
    #
	# while True:
	# 	if args.cluster=='n':
	# 		SNPs_index, keys=mapper.get_next()
	# 	else:
	# 		chunk=mapper.chunk_pop()
	# 		if chunk is None:
	# 			SNPs_index=None
	# 			break
	# 		print chunk
	# 		SNPs_index, keys=mapper.get_chunk(chunk)
    #
	# 	if SNPs_index is None:
	# 		break
	# 	RSID.append(keys)
    #
	# 	data=merge_genotype(gen, SNPs_index) #TODO (high) add mapper
	# 	print data.shape
	# 	if args.cluster=='n':
	# 		h5_gen_file = tables.open_file(
	# 			os.path.join(args.out,str(hdf5_iter)+'_'+h5_name+'.h5'), 'w', title=args.save_name)
	# 	else:#TODO (high) check!
	# 		h5_gen_file = tables.open_file(
	# 			os.path.join(args.out,str(chunk[0])+'_' +str(chunk[1])+'_'+h5_name+'.h5'), 'w', title=args.save_name)
	# 	hdf5_iter+=1
    #
	# 	atom = tables.Int8Atom()  # TODO (low) check data format
	# 	genotype = h5_gen_file.create_carray(h5_gen_file.root, 'genotype', atom,
	# 										(data.shape),
	# 										title='Genotype',
	# 										filters=pytable_filter)
	# 	genotype[:] = data
	# 	h5_gen_file.close()
	# 	genotype=None
	# 	data=None
	# 	gc.collect()
	# 	print hdf5_iter
    #
	# RSID=np.array(RSID)
	# SUB_ID=np.array(SUB_ID)
	# if args.cluster=='n':
	# 	np.save(os.path.join(args.out,'RSID.npy'),RSID)
	# 	np.save(os.path.join(args.out,'SUB_ID.npy'),SUB_ID)
    #
	# else:
	# 	np.save(os.path.join(args.out,str(args.node[1])+'_RSID.npy'),RSID)
	# 	np.save(os.path.join(args.out,str(args.node[1])+'_SUB_ID.npy'),SUB_ID)

import numpy as np
import pandas as pd
import os
import sys
import gc
import h5py
import tables
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument("-i", type=str, required=True, help="path to results file")
parser.add_argument("-p", type=str, required=True, help="path to probes file")
parser.add_argument("-o", type=str, required=True, help="path to output folder")
parser.add_argument("-split_ph", action='store_true', default=False, help="flag to split result in separate files per phenotype")
parser.add_argument("-chunk", type=int, default=1000000, help="chunk size to read results file")
args = parser.parse_args()


store=pd.HDFStore(args.p,'r')
df_probes_tmp=store.select('probes',start=0, stop=1)
probes_columns=df_probes_tmp.columns
if 'Rsq' in probes_columns:
    probes_columns=['ID','Rsq','allele1', 'allele2']
else:
    probes_columns = ['ID', 'allele1', 'allele2']


df_probes=store.select('probes',columns=probes_columns)
print('Probes shape {}'.format(df_probes.shape))

if os.path.isfile( args.p.split('.h5')[0] + '_hash_table.csv.gz'):
    try:
        df_hash = pd.read_csv( args.p.split('.h5')[0] + '_hash_table.csv.gz', sep='\t',
                              compression='gzip', index_col=False)
    except:
        df_hash = pd.read_csv( args.p.split('.h5')[0] + '_hash_table.csv.gz', sep='\t',
                              index_col=False)

    print('Hash table found!')
    df_probes = pd.merge(df_probes,df_hash, right_on='keys', left_on='allele1')
    df_probes['allele1'] = df_probes['allele']
    del df_probes['allele']
    df_probes = pd.merge(df_probes,df_hash, right_on='keys', left_on='allele2')
    df_probes['allele2'] = df_probes['allele']
    df_probes=df_probes[probes_columns]
else:
    print('Hash table is not found!!!')


df = pd.read_csv(args.i, compression='infer', sep=" ",chunksize=args.chunk )

phenotypes=[]
for i,df_chunk in enumerate(df):
    print('Chunk number {}'.format(i))
    df_chunk = pd.merge(df_chunk, df_probes, left_on='RSID', right_on='ID')
    df_chunk= df_chunk[probes_columns + [ 'MAF', 'BETA', 'SE', 'p_value', 'phenotype', 't-stat'  ]]
    df_chunk.rename(columns={'MAF': 'AF_coded'}, inplace=True)
    if not args.split_ph:
        hdr= False if os.path.isfile(os.path.join(args.o, os.path.basename(args.i))) else True
        df_chunk.to_csv(os.path.join(args.o, os.path.basename(args.i)), sep=' ', index=None, mode='a', header=hdr)
    else:
        phenotypes = np.unique( np.append( phenotypes, df_chunk.phenotype ) )
        for ph in phenotypes:
            df_tmp = df_chunk.query('phenotype=={}'.format(ph))
            if df_tmp.shape[0]!=0:
                hdr = False if os.path.isfile(os.path.join(args.o, str(ph) + "_" + os.path.basename(args.i))) else True
                df_tmp[probes_columns + ['AF_coded', 'BETA', 'SE', 'p_value', 't-stat']].to_csv(
                os.path.join(args.o, str(ph) + "_" + os.path.basename(args.i)), sep=' ', index=None,mode='a', header=hdr)


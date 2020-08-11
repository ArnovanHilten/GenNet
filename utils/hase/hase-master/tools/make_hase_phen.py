import pandas as pd
import numpy as np
import argparse
import os
import gc

parser = argparse.ArgumentParser(description='Make phenotype folder ready for HASE analysis')
parser.add_argument("-i",required=True, type=str, help="path to nparrays")
parser.add_argument("-id",type=str, help="csv data frame with two columns:"
                                         "1)id: ids identical to ids in genotype data"
                                         "2) exclude: 1 if exclude, 0 if include to analysis")

args = parser.parse_args()
print(args)


df=pd.read_csv(args.id, index_col=0)

print(df.head())
subject_id=[]

for i in df.iterrows():
    if i[1].exclude==1:
        subject_id.append('remove_'+str(i[1].id))
    else:
        subject_id.append(str(i[1].id))

dic={}
dic['id']=subject_id

files=os.listdir(args.i)

for f in files:
    print(f)
    if f.split('.')[-1]!='npy':
        raise ValueError('In {} should be only nparrays, not {}'.format(args.i,f))

    d=np.load(os.path.join(args.i,f))
    ch=f.split('_')[0].split('reg')[1]
    index=f.split('_')[1].split('.npy')[0]
    n,m=d.shape
    if n!=len(subject_id):
        raise ValueError('Number of ids {} from data frame not equal to number of ids {} from file {} '.format(len(subject_id),n,f))
    dic[f]=[ch+"_"+index+"_"+str(i) for i in range(m)]
    d=None
    gc.collect()

np.save(os.path.join(args.i,'info_dic.npy'),dic)
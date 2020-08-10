import numpy as np
import pandas as pd
import os
import sys
import argparse


parser = argparse.ArgumentParser(description='Script to edit phenotypes ids for HASE')
parser.add_argument("-i",required=True, type=str, help="path to nparrays of phenotypes")
parser.add_argument("-ids",type=str, help="path to one column data frame without headers"
                    "with a list of ids to exclude from analysis")
parser.add_argument("-out",required=True, type=str, help="path to save new phenotypes")

args = parser.parse_args()
print(args)

df_path=args.ids
pheno_path=args.i
out_path=args.out

if not os.path.isdir(out_path):
    print("Creating directory {}".format(out_path))
    os.mkdir(out_path)
else:
    if len(os.listdir(out_path))!=0:
        raise ValueError('Output folder is not empty!')

print("Crating soft link for original pheno files in {}".format(out_path))


for i in os.listdir(pheno_path):
    if 'info_dic' not in i:
        os.symlink(os.path.join(pheno_path,i) , os.path.join(out_path,i)   )
        

info_dic=np.load(os.path.join(pheno_path, 'info_dic.npy' )  ).item()

df=pd.read_csv(df_path,header=None, sep=',')
df.columns=['ID']
df['ID']=df.ID.astype('str')

index=pd.Series(info_dic['id']).isin(df.ID)
index=np.where(index==True)[0]

if len(index)!=0:
    print('There are {} ids to remove!'.format(len(index)))
    for i in index:   
        info_dic['id'][i]= "remove_"+ info_dic['id'][i]
else:
    print('No ids to remove!')
    
np.save(os.path.join(out_path,'info_dic.npy'),info_dic)



STUDYNAME=$1
ROOTPATH=$2
HASEROOT=$3

OUT=${ROOTPATH}/OUTPUT/${STUDYNAME}/

rm -rf ${OUT}
mkdir -p ${OUT}
mkdir -p ${OUT}/regression/

mkdir -p ${OUT}/PD_with_b4/
mkdir -p ${OUT}/PD_without_b4/

mkdir -p ${OUT}/PD_without_b4_cluster/
mkdir -p ${OUT}/PD_with_b4_cluster/

mkdir -p ${OUT}/encode/
mkdir -p ${OUT}/encode/encode_genotype/
mkdir -p ${OUT}/encode/encode_phenotype/
mkdir -p ${OUT}/encode/encode_individuals/
mkdir -p ${OUT}/MA_without_b4/
mkdir -p ${OUT}/MA_with_b4/
mkdir -p ${OUT}/MA_encode/
mkdir -p ${OUT}/MA_without_b4_cluster/
mkdir -p ${OUT}/MA_with_b4_cluster/


mkdir -p ${OUT}/summary_regression/
mkdir -p ${OUT}/summary_MA_without_b4/
mkdir -p ${OUT}/summary_MA_with_b4/
mkdir -p ${OUT}/summary_MA_encode/
mkdir -p ${OUT}/summary_MA_without_b4_cluster/
mkdir -p ${OUT}/summary_MA_with_b4_cluster/


python ${HASEROOT}/hase.py -mode encoding \
-g ${ROOTPATH}${STUDYNAME}/${STUDYNAME} \
-study_name ${STUDYNAME} \
-o ${OUT}/encode/ \
-ref_name ref_ES \
-mapper ${ROOTPATH}${STUDYNAME}/mapper/ \
-ph ${ROOTPATH}${STUDYNAME}/phenotype/ \




python ${HASEROOT}/hase.py -mode regression \
-th 0 \
-o ${OUT}/regression/ \
-g ${ROOTPATH}${STUDYNAME}/${STUDYNAME} \
-ph ${ROOTPATH}${STUDYNAME}/phenotype/ \
-cov ${ROOTPATH}${STUDYNAME}/covariates/ \
-ref_name ref_ES \
-mapper ${ROOTPATH}${STUDYNAME}/mapper/ \
-study_name ${STUDYNAME} \
-maf 0




python ${HASEROOT}/hase.py -mode single-meta \
-th 0 \
-o ${OUT}/PD_with_b4/ \
-g ${ROOTPATH}${STUDYNAME}/${STUDYNAME} \
-ph ${ROOTPATH}${STUDYNAME}/phenotype/ \
-cov ${ROOTPATH}${STUDYNAME}/covariates/ \
-study_name ${STUDYNAME} \
-maf 0 \
-mapper ${ROOTPATH}${STUDYNAME}/mapper/ \
-ref_name ref_ES \
-pd_full


python ${HASEROOT}/hase.py -mode single-meta \
-th 0 \
-o ${OUT}/PD_with_b4_cluster/ \
-g ${ROOTPATH}${STUDYNAME}/${STUDYNAME} \
-ph ${ROOTPATH}${STUDYNAME}/phenotype/ \
-cov ${ROOTPATH}${STUDYNAME}/covariates/ \
-study_name ${STUDYNAME} \
-maf 0 \
-mapper ${ROOTPATH}${STUDYNAME}/mapper/ \
-ref_name ref_ES \
-pd_full \
-cluster y \
-node 2 1

python ${HASEROOT}/hase.py -mode single-meta \
-th 0 \
-o ${OUT}/PD_with_b4_cluster/ \
-g ${ROOTPATH}${STUDYNAME}/${STUDYNAME} \
-ph ${ROOTPATH}${STUDYNAME}/phenotype/ \
-cov ${ROOTPATH}${STUDYNAME}/covariates/ \
-study_name ${STUDYNAME} \
-maf 0 \
-mapper ${ROOTPATH}${STUDYNAME}/mapper/ \
-ref_name ref_ES \
-pd_full \
-cluster y \
-node 2 2



python ${HASEROOT}/hase.py -mode single-meta \
-th 0 \
-o ${OUT}/PD_without_b4_cluster/ \
-g ${ROOTPATH}${STUDYNAME}/${STUDYNAME} \
-ph ${ROOTPATH}${STUDYNAME}/phenotype/ \
-cov ${ROOTPATH}${STUDYNAME}/covariates/ \
-study_name ${STUDYNAME} \
-maf 0 \
-mapper ${ROOTPATH}${STUDYNAME}/mapper/ \
-ref_name ref_ES \
-cluster y \
-node 3 1

python ${HASEROOT}/hase.py -mode single-meta \
-th 0 \
-o ${OUT}/PD_without_b4_cluster/ \
-g ${ROOTPATH}${STUDYNAME}/${STUDYNAME} \
-ph ${ROOTPATH}${STUDYNAME}/phenotype/ \
-cov ${ROOTPATH}${STUDYNAME}/covariates/ \
-study_name ${STUDYNAME} \
-maf 0 \
-mapper ${ROOTPATH}${STUDYNAME}/mapper/ \
-ref_name ref_ES \
-cluster y \
-node 3 2

python ${HASEROOT}/hase.py -mode single-meta \
-th 0 \
-o ${OUT}/PD_without_b4_cluster/ \
-g ${ROOTPATH}${STUDYNAME}/${STUDYNAME} \
-ph ${ROOTPATH}${STUDYNAME}/phenotype/ \
-cov ${ROOTPATH}${STUDYNAME}/covariates/ \
-study_name ${STUDYNAME} \
-maf 0 \
-mapper ${ROOTPATH}${STUDYNAME}/mapper/ \
-ref_name ref_ES \
-cluster y \
-node 3 3


python ${HASEROOT}/hase.py -mode single-meta \
-th 0 \
-o ${OUT}/PD_without_b4/ \
-g ${ROOTPATH}${STUDYNAME}/${STUDYNAME} \
-ph ${ROOTPATH}${STUDYNAME}/phenotype/ \
-cov ${ROOTPATH}${STUDYNAME}/covariates/ \
-study_name ${STUDYNAME} \
-maf 0 \
-ref_name ref_ES \
-mapper ${ROOTPATH}${STUDYNAME}/mapper/


rm ${OUT}/PD_without_b4/study_common_id.txt
rm ${OUT}/PD_without_b4/phen_id.txt
rm ${OUT}/PD_without_b4/gen_id.txt
rm ${OUT}/PD_without_b4/cov_id.txt
rm ${OUT}/PD_with_b4/study_common_id.txt
rm ${OUT}/PD_with_b4/phen_id.txt
rm ${OUT}/PD_with_b4/gen_id.txt
rm ${OUT}/PD_with_b4/cov_id.txt

python ${HASEROOT}/hase.py -mode meta-stage \
-th 0 \
-o ${OUT}/MA_with_b4/ \
-g ${ROOTPATH}${STUDYNAME}/${STUDYNAME} \
-ph ${ROOTPATH}${STUDYNAME}/phenotype/ \
-derivatives ${OUT}/PD_with_b4/  \
-study_name ${STUDYNAME} \
-maf 0 \
-ref_name ref_ES \
-mapper ${ROOTPATH}${STUDYNAME}/mapper/

python ${HASEROOT}/hase.py -mode meta-stage \
-th 0 \
-o ${OUT}/MA_without_b4/ \
-g ${ROOTPATH}${STUDYNAME}/${STUDYNAME} \
-ph ${ROOTPATH}${STUDYNAME}/phenotype/ \
-derivatives ${OUT}/PD_without_b4/  \
-study_name ${STUDYNAME} \
-maf 0 \
-ref_name ref_ES \
-mapper ${ROOTPATH}${STUDYNAME}/mapper/


rm ${OUT}/PD_without_b4_cluster/study_common_id.txt
rm ${OUT}/PD_without_b4_cluster/phen_id.txt
rm ${OUT}/PD_without_b4_cluster/gen_id.txt
rm ${OUT}/PD_without_b4_cluster/cov_id.txt
rm ${OUT}/PD_with_b4_cluster/study_common_id.txt
rm ${OUT}/PD_with_b4_cluster/phen_id.txt
rm ${OUT}/PD_with_b4_cluster/gen_id.txt
rm ${OUT}/PD_with_b4_cluster/cov_id.txt

python ${HASEROOT}/hase.py -mode meta-stage \
-th 0 \
-o ${OUT}/MA_with_b4_cluster/ \
-g ${ROOTPATH}${STUDYNAME}/${STUDYNAME} \
-ph ${ROOTPATH}${STUDYNAME}/phenotype/ \
-derivatives ${OUT}/PD_with_b4_cluster/  \
-study_name ${STUDYNAME} \
-maf 0 \
-ref_name ref_ES \
-mapper ${ROOTPATH}${STUDYNAME}/mapper/

python ${HASEROOT}/hase.py -mode meta-stage \
-th 0 \
-o ${OUT}/MA_without_b4_cluster/ \
-g ${ROOTPATH}${STUDYNAME}/${STUDYNAME} \
-ph ${ROOTPATH}${STUDYNAME}/phenotype/ \
-derivatives ${OUT}/PD_without_b4_cluster/  \
-study_name ${STUDYNAME} \
-maf 0 \
-ref_name ref_ES \
-mapper ${ROOTPATH}${STUDYNAME}/mapper/



mkdir ${OUT}/encode/study/
mkdir ${OUT}/encode/study/genotype/
mkdir ${OUT}/encode/study/probes/
mkdir ${OUT}/encode/study/individuals/

mv ${OUT}/encode/encode_genotype/* ${OUT}/encode/study/genotype/
mv ${OUT}/encode/encode_individuals/* ${OUT}/encode/study/individuals/
cp ${ROOTPATH}${STUDYNAME}/${STUDYNAME}/probes/* ${OUT}/encode/study/probes/


python ${HASEROOT}/hase.py -mode meta-stage \
-th 0 \
-o ${OUT}/MA_encode/ \
-g ${OUT}/encode/study/ \
-ph ${OUT}/encode/encode_phenotype/ \
-derivatives ${OUT}/PD_without_b4/  \
-study_name ${STUDYNAME} \
-maf 0 \
-ref_name ref_ES \
-mapper ${ROOTPATH}${STUDYNAME}/mapper/ \
-encoded 1

rm ${OUT}/summary_regression/*
rm ${OUT}/summary_MA_with_b4/*
rm ${OUT}/summary_MA_without_b4/*
rm ${OUT}/summary_MA_without_b4_cluster/*
rm ${OUT}/summary_MA_with_b4_cluster/*
rm ${OUT}/summary_MA_encode/*



python ${HASEROOT}/tools/analyzer.py -r ${OUT}/regression/ -o ${OUT}/summary_regression/
python ${HASEROOT}/tools/analyzer.py -r ${OUT}/MA_with_b4/ -o ${OUT}/summary_MA_with_b4/
python ${HASEROOT}/tools/analyzer.py -r ${OUT}/MA_without_b4/ -o ${OUT}/summary_MA_without_b4/
python ${HASEROOT}/tools/analyzer.py -r ${OUT}/MA_without_b4_cluster/ -o ${OUT}/summary_MA_without_b4_cluster/
python ${HASEROOT}/tools/analyzer.py -r ${OUT}/MA_with_b4_cluster/ -o ${OUT}/summary_MA_with_b4_cluster/
python ${HASEROOT}/tools/analyzer.py -r ${OUT}/MA_encode/ -o ${OUT}/summary_MA_encode/


python -c "
import numpy as np
import pandas as pd
import os
import sys

test=[]
threshold=10**-6

df_regression=pd.read_csv('{}/results.csv'.format('${OUT}/summary_regression/'),sep=' ' )


df_MA_test={}

df_MA_test['MA_with_b4']=pd.read_csv('{}/results.csv'.format('${OUT}/summary_MA_with_b4/'),sep=' ')
df_MA_test['MA_without_b4']=pd.read_csv('{}/results.csv'.format('${OUT}/summary_MA_without_b4/'),sep=' ')
df_MA_test['MA_with_b4']=pd.read_csv( '{}/results.csv'.format('${OUT}/summary_MA_with_b4/'),sep=' ')
df_MA_test['MA_encode']=pd.read_csv( '{}/results.csv'.format('${OUT}/summary_MA_encode/'),sep=' ')
df_MA_test['MA_without_b4_cluster']=pd.read_csv( '{}/results.csv'.format('${OUT}/summary_MA_without_b4_cluster/'),sep=' ' )
df_MA_test['MA_with_b4_cluster']=pd.read_csv( '{}/results.csv'.format('${OUT}/summary_MA_with_b4_cluster/'),sep=' ')


for k in df_MA_test:
    df_check=pd.merge(df_MA_test[k][['RSID','phenotype','BETA','SE','p_value']] ,df_regression[['RSID','phenotype','BETA','SE','p_value']], on=['RSID','phenotype']  )
    df_check['BETA_diff']=np.abs(df_check.BETA_x - df_check.BETA_y)

    if np.max(np.abs(df_check.BETA_x - df_check.BETA_y))<threshold and  np.max(np.abs(df_check.SE_x - df_check.SE_y))<threshold:
        test.append(0)
    else:
        print ('FAILED {}'.format(k))
        print np.max(np.abs(df_check.BETA_x - df_check.BETA_y))
        test.append(1)

if np.sum(test)==0:
    print ('!!! ALL TESTS PASSED !!!')

else:
    print ('{} TESTS FAILED'.format(np.sum(test)))

"


























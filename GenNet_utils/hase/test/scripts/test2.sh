
STUDYNAME1=$1
STUDYNAME2=$2
ROOTPATH=$3
HASEROOT=$4


OUT=${ROOTPATH}/OUTPUT/${STUDYNAME1}_${STUDYNAME2}/

rm -rf ${OUT}
mkdir -p ${OUT}/mapper/
mkdir -p ${OUT}/MA_without_b4/
mkdir -p ${OUT}/MA_with_b4/
mkdir -p ${OUT}/MA_encode/
mkdir -p ${OUT}/MA_encode_10/
mkdir -p ${OUT}/MA_encode_01/


mkdir -p ${OUT}/summary_MA_without_b4/
mkdir -p ${OUT}/summary_MA_with_b4/
mkdir -p ${OUT}/summary_MA_encode/
mkdir -p ${OUT}/summary_MA_encode_10/
mkdir -p ${OUT}/summary_MA_without_b4_reverse/
mkdir -p ${OUT}/summary_MA_encode_reverse/
mkdir -p ${OUT}/summary_MA_encode_01/
mkdir -p ${OUT}/summary_MA_without_b4_effect/

ln -s ${ROOTPATH}${STUDYNAME1}/mapper/* ${OUT}/mapper/
ln -s ${ROOTPATH}${STUDYNAME2}/mapper/values_ref_ES_${STUDYNAME2}.npy ${OUT}/mapper/
ln -s ${ROOTPATH}${STUDYNAME2}/mapper/flip_ref_ES_${STUDYNAME2}.npy ${OUT}/mapper/




if [ -e "${ROOTPATH}/OUTPUT/${STUDYNAME1}/PD_with_b4/${STUDYNAME1}_b4.npy" ] && [ -e "${ROOTPATH}/OUTPUT/${STUDYNAME2}/PD_with_b4/${STUDYNAME2}_b4.npy" ]
then

python  ${HASEROOT}/hase.py -mode meta-stage \
-th 0 \
-o ${OUT}/MA_with_b4/ \
-g ${ROOTPATH}${STUDYNAME1}/${STUDYNAME1} ${ROOTPATH}${STUDYNAME2}/${STUDYNAME2} \
-ph ${ROOTPATH}${STUDYNAME1}/phenotype/ ${ROOTPATH}${STUDYNAME1}/phenotype/ \
-derivatives ${ROOTPATH}/OUTPUT/${STUDYNAME1}/PD_with_b4/ ${ROOTPATH}/OUTPUT/${STUDYNAME2}/PD_with_b4/  \
-study_name ${STUDYNAME1} ${STUDYNAME2} \
-maf 0 \
-ref_name ref_ES \
-mapper ${OUT}/mapper/ \
-encoded 0 0

fi

python  ${HASEROOT}/hase.py -mode meta-stage \
-th 0 \
-o ${OUT}/MA_without_b4/ \
-g ${ROOTPATH}${STUDYNAME1}/${STUDYNAME1} ${ROOTPATH}${STUDYNAME2}/${STUDYNAME2} \
-ph ${ROOTPATH}${STUDYNAME1}/phenotype/ ${ROOTPATH}${STUDYNAME2}/phenotype/ \
-derivatives ${ROOTPATH}/OUTPUT/${STUDYNAME1}/PD_without_b4/ ${ROOTPATH}/OUTPUT/${STUDYNAME2}/PD_without_b4/  \
-study_name ${STUDYNAME1} ${STUDYNAME2} \
-maf 0 \
-ref_name ref_ES \
-mapper ${OUT}/mapper/ \
-encoded 0 0

python  ${HASEROOT}/hase.py -mode meta-stage \
-th 0 \
-o ${OUT}/MA_without_b4_reverse/ \
-g ${ROOTPATH}${STUDYNAME2}/${STUDYNAME2} ${ROOTPATH}${STUDYNAME1}/${STUDYNAME1} \
-ph ${ROOTPATH}${STUDYNAME2}/phenotype/ ${ROOTPATH}${STUDYNAME1}/phenotype/ \
-derivatives ${ROOTPATH}/OUTPUT/${STUDYNAME2}/PD_without_b4/ ${ROOTPATH}/OUTPUT/${STUDYNAME1}/PD_without_b4/  \
-study_name ${STUDYNAME2} ${STUDYNAME1} \
-maf 0 \
-ref_name ref_ES \
-mapper ${OUT}/mapper/ \
-encoded 0 0



python  ${HASEROOT}/hase.py -mode meta-stage \
-th 0 \
-o ${OUT}/MA_encode/ \
-g ${ROOTPATH}/OUTPUT/${STUDYNAME1}/encode/study/ ${ROOTPATH}/OUTPUT/${STUDYNAME2}/encode/study/ \
-ph ${ROOTPATH}/OUTPUT/${STUDYNAME1}/encode/encode_phenotype/ ${ROOTPATH}/OUTPUT/${STUDYNAME2}/encode/encode_phenotype/ \
-derivatives ${ROOTPATH}/OUTPUT/${STUDYNAME1}/PD_without_b4/ ${ROOTPATH}/OUTPUT/${STUDYNAME2}/PD_without_b4/  \
-study_name ${STUDYNAME1} ${STUDYNAME2} \
-maf 0 \
-ref_name ref_ES \
-mapper ${OUT}/mapper/ \
-encoded 1 1



python  ${HASEROOT}/hase.py -mode meta-stage \
-th 0 \
-o ${OUT}/MA_encode_reverse/ \
-g ${ROOTPATH}/OUTPUT/${STUDYNAME2}/encode/study/ ${ROOTPATH}/OUTPUT/${STUDYNAME1}/encode/study/ \
-ph ${ROOTPATH}/OUTPUT/${STUDYNAME2}/encode/encode_phenotype/ ${ROOTPATH}/OUTPUT/${STUDYNAME1}/encode/encode_phenotype/ \
-derivatives ${ROOTPATH}/OUTPUT/${STUDYNAME2}/PD_without_b4/ ${ROOTPATH}/OUTPUT/${STUDYNAME1}/PD_without_b4/  \
-study_name ${STUDYNAME2} ${STUDYNAME1} \
-maf 0 \
-ref_name ref_ES \
-mapper ${OUT}/mapper/ \
-encoded 1 1



python  ${HASEROOT}/hase.py -mode meta-stage \
-th 0 \
-o ${OUT}/MA_encode_10/ \
-g ${ROOTPATH}/OUTPUT/${STUDYNAME1}/encode/study/ ${ROOTPATH}${STUDYNAME2}/${STUDYNAME2} \
-ph ${ROOTPATH}/OUTPUT/${STUDYNAME1}/encode/encode_phenotype/ ${ROOTPATH}${STUDYNAME2}/phenotype/ \
-derivatives ${ROOTPATH}/OUTPUT/${STUDYNAME1}/PD_without_b4/ ${ROOTPATH}/OUTPUT/${STUDYNAME2}/PD_without_b4/  \
-study_name ${STUDYNAME1} ${STUDYNAME2} \
-maf 0 \
-ref_name ref_ES \
-mapper ${OUT}/mapper/ \
-encoded 1 0

python  ${HASEROOT}/hase.py -mode meta-stage \
-th 0 \
-o ${OUT}/MA_encode_01/ \
-g ${ROOTPATH}${STUDYNAME1}/${STUDYNAME1} ${ROOTPATH}/OUTPUT/${STUDYNAME2}/encode/study/ \
-ph ${ROOTPATH}${STUDYNAME1}/phenotype/  ${ROOTPATH}/OUTPUT/${STUDYNAME2}/encode/encode_phenotype/ \
-derivatives ${ROOTPATH}/OUTPUT/${STUDYNAME1}/PD_without_b4/ ${ROOTPATH}/OUTPUT/${STUDYNAME2}/PD_without_b4/  \
-study_name ${STUDYNAME1} ${STUDYNAME2} \
-maf 0 \
-ref_name ref_ES \
-mapper ${OUT}/mapper/ \
-encoded 0 1


python  ${HASEROOT}/hase.py -mode meta-stage \
-th 0 \
-o ${OUT}/MA_without_b4_effect/ \
-g ${ROOTPATH}${STUDYNAME1}/${STUDYNAME1} ${ROOTPATH}${STUDYNAME2}/${STUDYNAME2} \
-ph ${ROOTPATH}${STUDYNAME1}/phenotype/ ${ROOTPATH}${STUDYNAME2}/phenotype/ \
-derivatives ${ROOTPATH}/OUTPUT/${STUDYNAME1}/PD_without_b4/ ${ROOTPATH}/OUTPUT/${STUDYNAME2}/PD_without_b4/  \
-study_name ${STUDYNAME1} ${STUDYNAME2} \
-maf 0 \
-ref_name ref_ES \
-mapper ${OUT}/mapper/ \
-effect_intercept \
-encoded 0 0



if [ -e "${ROOTPATH}/OUTPUT/${STUDYNAME1}/PD_with_b4/${STUDYNAME1}_b4.npy" ] && [ -e "${ROOTPATH}/OUTPUT/${STUDYNAME2}/PD_with_b4/${STUDYNAME2}_b4.npy" ]
then
python  ${HASEROOT}/tools/analyzer.py -r ${OUT}/MA_with_b4/ -o ${OUT}/summary_MA_with_b4/
fi
python  ${HASEROOT}/tools/analyzer.py -r ${OUT}/MA_without_b4/ -o ${OUT}/summary_MA_without_b4/
python  ${HASEROOT}/tools/analyzer.py -r ${OUT}/MA_encode/ -o ${OUT}/summary_MA_encode/
python  ${HASEROOT}/tools/analyzer.py -r ${OUT}/MA_encode_10/ -o ${OUT}/summary_MA_encode_10/
python  ${HASEROOT}/tools/analyzer.py -r ${OUT}/MA_encode_reverse/ -o ${OUT}/summary_MA_encode_reverse/
python  ${HASEROOT}/tools/analyzer.py -r ${OUT}/MA_without_b4_reverse/ -o ${OUT}/summary_MA_without_b4_reverse/
python  ${HASEROOT}/tools/analyzer.py -r ${OUT}MA_encode_01/ -o ${OUT}/summary_MA_encode_01/
python  ${HASEROOT}/tools/analyzer.py -r ${OUT}MA_without_b4_effect/ -o ${OUT}/summary_MA_without_b4_effect/


python -c "
import numpy as np
import pandas as pd
import os
import sys

test=[]
threshold=10**-6

df_MA=pd.read_csv('{}/results.csv'.format('${OUT}/summary_MA_without_b4/'),sep=' ')


df_MA_test={}

df_MA_test['MA_encode']=pd.read_csv('{}/results.csv'.format('${OUT}/summary_MA_encode/'),sep=' ')
df_MA_test['MA_encode_10']=pd.read_csv('{}/results.csv'.format('${OUT}/summary_MA_encode_10/'),sep=' ')
df_MA_test['MA_encode_reverse']=pd.read_csv('{}/results.csv'.format('${OUT}/summary_MA_encode_reverse/'),sep=' ')
df_MA_test['MA_without_b4_reverse']=pd.read_csv('{}/results.csv'.format('${OUT}/summary_MA_without_b4_reverse/'),sep=' ')
df_MA_test['MA_encode_01']=pd.read_csv('{}/results.csv'.format('${OUT}/summary_MA_encode_01/'),sep=' ')
df_MA_test['MA_effect']=pd.read_csv('{}/results.csv'.format('${OUT}/summary_MA_without_b4_effect/'),sep=' ')
if os.path.isfile(  '{}/results.csv'.format('${OUT}/summary_MA_with_b4/')  ):
    df_MA_test['MA_with_b4']=pd.read_csv( '{}/results.csv'.format('${OUT}/summary_MA_with_b4/'),sep=' ')

for k in df_MA_test:
    df_check=pd.merge(df_MA_test[k][['RSID','phenotype','BETA','SE','p_value']] ,df_MA[['RSID','phenotype','BETA','SE','p_value']], on=['RSID','phenotype']  )
    df_check['BETA_diff']=np.abs(df_check.BETA_x - df_check.BETA_y)

    if np.max(np.abs(df_check.BETA_x - df_check.BETA_y))<threshold and  np.max(np.abs(df_check.SE_x - df_check.SE_y))<threshold:
        test.append(0)
    else:
        print ('FAILED {}'.format(k))
        test.append(1)

if np.sum(test)==0:
    print ('!!! ALL TESTS PASSED !!!')

else:
    print ('{} TESTS FAILED'.format(np.sum(test)))

"











#!/usr/bin/env bash
### in this directory should be only VCF files


GENOTYPE_DIR=$1
SAVE_DIR=$2
HASEDIR=$3
STUDYNAME=$4

SUBJECT_ID_FILE='SUB_ID.txt'

SNPs_INFO='SNPs_info.txt'


CWD=`pwd`

cd $GENOTYPE_DIR


rm -f ${SAVE_DIR}/${SUBJECT_ID_FILE}
rm -f ${SAVE_DIR}/${SNPs_INFO}

count=1
for file in *; do
files_order[${count}]=${file}
((count++))
done

echo ${files_order[*]}

zcat -f ${files_order[1]} | awk 'BEGIN{FS="\t"}/^#CHROM/{for (i=10;i<=NF;i++){print $i }; exit}'  >> ${SAVE_DIR}/${SUBJECT_ID_FILE}

for i in ${files_order[*]};
do
echo ${i}
zcat -f ${i} | cut -f1-8 | awk -v name=${i} -v file="${SAVE_DIR}/snps_count.txt" 'BEGIN{FS="\t"}/^[^#]/{i++;print }END{ print name,i >>file }'  >> ${SAVE_DIR}/${SNPs_INFO}
echo $i >> ${SAVE_DIR}/files_order.txt
done


N_SUB=` cat ${SAVE_DIR}/${SUBJECT_ID_FILE} | wc -l `
N_SNPs=`cat ${SAVE_DIR}/${SNPs_INFO} | wc -l `


echo "There are ${N_SUB} subjects in this genotype data"
echo "There are ${N_SNPs} variants in this genotype data"
echo ${N_SUB} >> ${SAVE_DIR}/info.txt
echo ${N_SNPs} >> ${SAVE_DIR}/info.txt

python ${HASEDIR}/tools/VCF2hdf5.py -flag probes  -data ${SAVE_DIR}/${SNPs_INFO} -out ${SAVE_DIR} -study_name ${STUDYNAME}
python ${HASEDIR}/tools/VCF2hdf5.py -flag individuals  -data ${SAVE_DIR}/${SUBJECT_ID_FILE} -out ${SAVE_DIR} -study_name ${STUDYNAME}


rm ${SAVE_DIR}/SNPs_info.txt

cd ${CWD}
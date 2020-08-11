#!/usr/bin/env bash
### in this directory should be only dose and info files


GENOTYPE_DIR=$1
SAVE_DIR=$2
HASEDIR=$3
STUDYNAME=$4

SUBJECT_ID_FILE='SUB_ID.txt'
SUBJECT_FAM_FILE='SUB_FAM.txt'

SNPs_INFO='SNPs_info.txt'


CWD=`pwd`

cd $GENOTYPE_DIR

id_file=`ls  | grep dose | head -n1`


rm -f ${SAVE_DIR}/${SUBJECT_ID_FILE}
rm -f ${SAVE_DIR}/${SNPs_INFO}
rm -f ${SAVE_DIR}/id_convert.txt


zcat -f $id_file | gawk '{ print $1}' | gawk 'BEGIN{FS="->"}{if(NF==2){print $2}else{print $1}}' >> ${SAVE_DIR}/${SUBJECT_ID_FILE}
zcat -f $id_file | gawk '{ print $1}' | gawk 'BEGIN{FS="->"}{if(NF==2){print $1}else{print "NO FAM DATA";exit}}' >> ${SAVE_DIR}/${SUBJECT_FAM_FILE}

files_order=`awk '{if(NR==FNR){split($1,a,".dose");f[a[1]]++}else{split($1,b,".info");f[b[1]]++}}END{for(i in f){if(f[i]==2){print i}}}' <(ls | grep dose) <(ls | grep info)`

for i in $files_order;
do
zcat -f $i.info.gz | awk '{if(NR!=1){print $1,$2,$3,$5,$7}}' >> ${SAVE_DIR}/${SNPs_INFO}
echo $i >> ${SAVE_DIR}/files_order.txt
done


N_SUB=` cat ${SAVE_DIR}/${SUBJECT_ID_FILE} | wc -l `
N_SNPs=`cat ${SAVE_DIR}/${SNPs_INFO} | wc -l `


echo "There are ${N_SUB} subjects in this genotype data"
echo "There are ${N_SNPs} variants in this genotype data"
echo ${N_SUB} >> ${SAVE_DIR}/info.txt
echo ${N_SNPs} >> ${SAVE_DIR}/info.txt

for id in $( cat ${SAVE_DIR}/${SUBJECT_ID_FILE} );
do
echo "sh ${HASEDIR}/tools/id2hdf5.sh ${id} ${GENOTYPE_DIR} ${SAVE_DIR} ${HASEDIR} ${STUDYNAME}" >> ${SAVE_DIR}/id_convert.sh
done

chmod +x ${SAVE_DIR}/id_convert.sh

python ${HASEDIR}/tools/minimac2hdf5.py -flag probes  -data ${SAVE_DIR}/${SNPs_INFO} -out ${SAVE_DIR} -study_name ${STUDYNAME}
python ${HASEDIR}/tools/minimac2hdf5.py -flag individuals  -data ${SAVE_DIR}/${SUBJECT_ID_FILE} -out ${SAVE_DIR} -study_name ${STUDYNAME}


rm ${SAVE_DIR}/SNPs_info.txt

cd ${CWD}




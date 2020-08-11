#!/usr/bin/env bash


CWD=`pwd`

id=$1
GENOTYPE_DIR=$2
SAVE_DIR=$3
HASEDIR=$4
STUDYNAME=$5

cd $GENOTYPE_DIR

for i in $(cat ${SAVE_DIR}/files_order.txt ); do
zcat $i.dose.gz | gawk '{split($1,a,"->");if(a[2]=="'$id'"){$1="";$2="";gsub("  ","",$0);gsub(" ","\n",$0);print $0; exit}}' >>${SAVE_DIR}/${id}.txt
done

python ${HASEDIR}/tools/minimac2hdf5.py -flag genotype -id ${id} -data ${SAVE_DIR}/${id}.txt -out ${SAVE_DIR} -study_name ${STUDYNAME}

rm ${SAVE_DIR}/${id}.txt

cd $CWD


GENOTYPE_DIR=$1
SAVE_DIR=$2
HASEDIR=$3
STUDYNAME=$4
START=$5
FINISH=$6
CHUNK_NUMBER=$7

get_minimac_chunk(){
    echo "***"
    echo $@
    echo "***"
    iter=0
    file_ind=0
    for i in $( cat ${SAVE_DIR}/files_order.txt | awk '{print $1}' ); do

        file=`ls ${GENOTYPE_DIR} | grep ${i}.dose`
        file_ind=$(($file_ind + 1))
        SNPs=`awk "{print NF; exit}" <( zcat -f ${GENOTYPE_DIR}/${file} )`
        if [ ! $SNPs -eq 0 ]; then
            SNPs=$(($SNPs - 2 ))
            echo ${SNPs}, ${file}
            iter=$(($iter + $SNPs))
        fi
        if [ ${iter} -ge ${1} ]; then

            if [ ${iter} -ge ${2} ]; then

                BEGIN=$(($1 - ${iter} + ${SNPs} + 2 ))
                END=$(( $2 - ${iter} + ${SNPs} + 2  ))
                echo ${iter},${SNPs},${BEGIN},${END}
                if [ -f ${SAVE_DIR}/chunk_${START}_${FINISH}.txt ]; then
                paste ${SAVE_DIR}/chunk_${START}_${FINISH}.txt <( zcat -f ${GENOTYPE_DIR}/${file} |  cut -f${BEGIN}-${END}  ) > ${SAVE_DIR}/tmp_chunk_${START}_${FINISH}.txt
                rm ${SAVE_DIR}/chunk_${START}_${FINISH}.txt
                mv ${SAVE_DIR}/tmp_chunk_${START}_${FINISH}.txt ${SAVE_DIR}/chunk_${START}_${FINISH}.txt
                rm ${SAVE_DIR}/tmp_chunk_${START}_${FINISH}.txt
                else
                    cut -f${BEGIN}-${END}  <( zcat -f ${GENOTYPE_DIR}/${file} ) >> ${SAVE_DIR}/chunk_${START}_${FINISH}.txt
                fi
            return 0
            else
                BEGIN=$(( $1 - ${iter} + ${SNPs} + 2 ))
                END=$(( 2 + $SNPs  ))
                echo ${iter},${SNPs},${BEGIN},${END}
                if [ -f ${SAVE_DIR}/chunk_${START}_${FINISH}.txt ]; then
                     paste ${SAVE_DIR}/chunk_${START}_${FINISH}.txt <( zcat -f ${GENOTYPE_DIR}/${file} |  cut -f${BEGIN}-${END}  ) > ${SAVE_DIR}/tmp_chunk_${START}_${FINISH}.txt
                     rm ${SAVE_DIR}/chunk_${START}_${FINISH}.txt
                     mv ${SAVE_DIR}/tmp_chunk_${START}_${FINISH}.txt ${SAVE_DIR}/chunk_${START}_${FINISH}.txt
                     rm ${SAVE_DIR}/tmp_chunk_${START}_${FINISH}.txt
                else
                    cut -f${BEGIN}-${END}  <( zcat -f ${GENOTYPE_DIR}/${file} ) >> ${SAVE_DIR}/chunk_${START}_${FINISH}.txt
                fi
                get_minimac_chunk $(($iter + 1)) ${2}
                return 0
            fi
        fi
    echo ${iter}
    done
}


get_minimac_chunk ${START} ${FINISH}
python ${HASEDIR}/tools/minimac2hdf5.py -flag chunk -id ${CHUNK_NUMBER} -data ${SAVE_DIR}/chunk_${START}_${FINISH}.txt -out ${SAVE_DIR} -study_name ${STUDYNAME}





GENOTYPE_DIR=$1
SAVE_DIR=$2
HASEDIR=$3
STUDYNAME=$4
START=$5
FINISH=$6
CHUNK_NUMBER=$7

get_vcf_chunk(){
    echo "***"
    echo $@
    echo "***"
    iter=0
    file_ind=0
    for i in $( cat ${SAVE_DIR}/files_order.txt | awk '{print }' ); do
        echo $i
        file=${i}
        file_ind=$(($file_ind + 1))
        SNPs="$(awk -v N=${file_ind} '{if(NR==N){print $2}}' ${SAVE_DIR}/snps_count.txt)"
        if [ ! $SNPs -eq 0 ]; then
            echo ${SNPs}, ${file}
            iter=$(($iter + $SNPs))
        fi
        if [ ${iter} -ge ${1} ]; then

            if [ ${iter} -ge ${2} ]; then

                BEGIN=$(($1 - ${iter} + ${SNPs}))
                END=$(( $2 - ${iter} + ${SNPs} ))
                echo ${iter},${SNPs},${BEGIN},${END}
                if [ "$3" = "GT" ]; then
                echo 'use GT'
                    zcat -f ${GENOTYPE_DIR}/${file} | awk 'BEGIN{FS="\t"}/^[^#]/{print }' | cut -f10-  | sed -n "${BEGIN},${END}p;${END}q" |  tr / " "| tr \| " "  | awk -v ind=$4 'BEGIN{FS="\t"}{R="";for (i=1;i<=NF;i++){split($i,a,":");split(a[ind],b," ");if(b[1]=="." || b[2]=="."){g=-7}else{ if(b[1]>1){b[1]=1};if(b[2]>1){b[2]=1}; g=b[1]+b[2]  };if(i==1){R=2-g}else{R=R"\t"2-g} };print R} '  >> ${SAVE_DIR}/chunk_${START}_${FINISH}.txt
                else
                echo "use ${3}"
                    zcat -f ${GENOTYPE_DIR}/${file} | awk 'BEGIN{FS="\t"}/^[^#]/{print }' | cut -f10-  | sed -n "${BEGIN},${END}p;${END}q" | awk -v ind=$4 'BEGIN{FS="\t"}{R="";for (i=1;i<=NF;i++){split($i,a,":");if(a[ind]=="."){a[ind]=-7};if(i==1){R=2-a[ind]}else{R=R"\t"2-a[ind]} };print R} '  >> ${SAVE_DIR}/chunk_${START}_${FINISH}.txt
                fi
            return 0
            else
                BEGIN=$(( $1 - ${iter} + ${SNPs}))
                END=$(( $SNPs  ))
                echo ${iter},${SNPs},${BEGIN},${END}
                if [ "$3" = "GT" ]; then
                echo 'use GT'
                    zcat -f ${GENOTYPE_DIR}/${file} | awk 'BEGIN{FS="\t"}/^[^#]/{print }' | cut -f10-  | sed -n "${BEGIN},${END}p;${END}q" |  tr / " "| tr \| " "  | awk -v ind=$4 'BEGIN{FS="\t"}{R="";for (i=1;i<=NF;i++){split($i,a,":");split(a[ind],b," ");if(b[1]=="." || b[2]=="."){g=-7}else{ if(b[1]>1){b[1]=1};if(b[2]>1){b[2]=1}; g=b[1]+b[2]  };if(i==1){R=2-g}else{R=R"\t"2-g} };print R} '  >> ${SAVE_DIR}/chunk_${START}_${FINISH}.txt
                else
                echo "use ${3}"
                    zcat -f ${GENOTYPE_DIR}/${file} | awk 'BEGIN{FS="\t"}/^[^#]/{print }' | cut -f10-  | sed -n "${BEGIN},${END}p;${END}q" | awk -v ind=$4 'BEGIN{FS="\t"}{R="";for (i=1;i<=NF;i++){split($i,a,":");if(a[ind]=="."){a[ind]=-7};if(i==1){R=2-a[ind]}else{R=R"\t"2-a[ind]} };print R} '  >> ${SAVE_DIR}/chunk_${START}_${FINISH}.txt
                fi
                get_vcf_chunk $(($iter + 1)) ${2} $3 $4

                return 0
            fi
        fi
    echo ${iter}
    done
}

for i in GT DS EC; do

    zcat -f ${GENOTYPE_DIR}/$( head -n1 ${SAVE_DIR}/files_order.txt ) | awk '{if($1!="#CHROM"){print}else{exit} }' | grep -q $i
    if [ $? -eq 0 ]; then
        code=$i
    fi
done

ind=` zcat -f ${GENOTYPE_DIR}/$( head -n1 ${SAVE_DIR}/files_order.txt ) | awk -v code=$code 'BEGIN{FS="\t"}/^[^#]/{ split($9,a,":");for(i=1;i<=length(a);i++){if(a[i]==code){print i}} ; exit }' `

get_vcf_chunk ${START} ${FINISH} ${code} ${ind}
python ${HASEDIR}/tools/VCF2hdf5.py -flag chunk -id ${CHUNK_NUMBER} -data ${SAVE_DIR}/chunk_${START}_${FINISH}.txt -out ${SAVE_DIR} -study_name ${STUDYNAME}




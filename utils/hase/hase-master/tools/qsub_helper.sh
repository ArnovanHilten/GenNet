#!/usr/bin/env bash

ROW=$SGE_TASK_ID
TASK_FILE=$1
`awk -v r=$ROW '{if(NR==r){print }}' ${TASK_FILE}`
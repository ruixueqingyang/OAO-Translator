#!/bin/bash

times=$1 

# cd /home/anjushi/work/coding-6/1207

# make particlefilter

cd /home/anjushi/work/par/Rodinia_3.1/openmp/particlefilter

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

FILE1="/home/anjushi/work/par/Rodinia_3.1/openmp/particlefilter/runtime_1.txt"
FILE5="/home/anjushi/work/par/Rodinia_3.1/openmp/particlefilter/profile_1_oao.txt"
FILE6="/home/anjushi/work/par/Rodinia_3.1/openmp/particlefilter/profile_1_manual.txt"

rm -rf $FILE1  $FILE5  $FILE6

make clean

#make particlefilter  #OMP
make EXPENSES=-DEXPENSES OUT   #OAO
make OAO_manual   #manual

for ((i=1; i<=$times; i ++))
do  
   `nvprof --csv --print-gpu-trace ./OUT.bin -x 128 -y 128 -z 10 -np 10000 1>> runtime_1.txt 2>> profile_1_oao.txt`  #OAO

   `nvprof --csv --print-gpu-trace ./OAO_manual.bin -x 128 -y 128 -z 10 -np 10000 1>> runtime_1.txt 2>> profile_1_manual.txt`  #manual

done

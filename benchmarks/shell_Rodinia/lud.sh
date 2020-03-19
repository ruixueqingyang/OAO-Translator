#!/bin/bash

times=$1 

# cd /home/anjushi/work/coding-6/1207

# make lud/omp

cd /home/wfr/work/Coding/Rodinia_3.1/openmp/lud/omp

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

FILE1="/home/wfr/work/Coding/Rodinia_3.1/openmp/lud/omp/runtime_1.txt"
FILE5="/home/wfr/work/Coding/Rodinia_3.1/openmp/lud/omp/profile_1_oao.txt"
FILE6="/home/wfr/work/Coding/Rodinia_3.1/openmp/lud/omp/profile_1_manual.txt"

rm -rf $FILE1  $FILE5  $FILE6

make clean

make one  #OMP
make EXPENSES=-DEXPENSES out   #OAO
make OAO_manual   #manual

for ((i=1; i<=$times; i ++))
do  
   `./one.bin -s 8000 1>> runtime_1.txt`
   `nvprof --csv --print-gpu-trace ./out.bin -s 8000 1>> /dev/null 2>> profile_1_oao.txt`  #OAO
   `./out.bin -s 8000  1>> runtime_1.txt`
   `nvprof --csv --print-gpu-trace ./OAO_manual.bin -s 8000 1>> /dev/null 2>> profile_1_manual.txt`  #manual
   `./OAO_manual.bin -s 8000 1>> runtime_1.txt`

done

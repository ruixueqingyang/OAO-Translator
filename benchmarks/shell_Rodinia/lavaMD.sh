#!/bin/bash

times=$1 

# cd /home/anjushi/work/coding-6/1207

# make lavaMD

cd /home/wfr/work/Coding/Rodinia_3.1/openmp/lavaMD

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

FILE1="/home/wfr/work/Coding/Rodinia_3.1/openmp/lavaMD/runtime_1.txt"
FILE5="/home/wfr/work/Coding/Rodinia_3.1/openmp/lavaMD/profile_1_oao.txt"
FILE6="/home/wfr/work/Coding/Rodinia_3.1/openmp/lavaMD/profile_1_manual.txt"

rm -rf $FILE1  $FILE5  $FILE6

make clean

make a.out  #OMP
make EXPENSES=-DEXPENSES main_out   #OAO
make main_out_manual   #manual

for ((i=1; i<=$times; i ++))
do  

   `./lavaMD -cores 24 -boxes1d 15 1>> runtime_1.txt`
   `nvprof --csv --print-gpu-trace ./main_out.bin -cores 24 -boxes1d 15 1>> /dev/null 2>> profile_1_oao.txt`  #OAO
   `./main_out.bin -cores 24 -boxes1d 15 1>> runtime_1.txt`
   `nvprof --csv --print-gpu-trace ./main_out_manual.bin -cores 24 -boxes1d 15 1>> /dev/null 2>> profile_1_manual.txt`  #manual
   `./main_out_manual.bin -cores 24 -boxes1d 15 1>> runtime_1.txt`

done

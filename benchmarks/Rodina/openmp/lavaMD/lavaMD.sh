#!/bin/bash

times=$1 

# cd /home/anjushi/work/coding-6/1207

# make lavaMD

cd /home/anjushi/work/par/Rodinia_3.1/openmp/lavaMD

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

FILE1="/home/anjushi/work/par/Rodinia_3.1/openmp/lavaMD/runtime_1.txt"
FILE5="/home/anjushi/work/par/Rodinia_3.1/openmp/lavaMD/profile_1_oao.txt"
FILE6="/home/anjushi/work/par/Rodinia_3.1/openmp/lavaMD/profile_1_manual.txt"

rm -rf $FILE1  $FILE5  $FILE6

make clean

#make lavaMD  #OMP
make EXPENSES=-DEXPENSES main_out   #OAO
make main_out_manual   #manual

for ((i=1; i<=$times; i ++))
do  

   `nvprof --csv --print-gpu-trace ./main_out.bin -cores 24 -boxes1d 15 1>> runtime_1.txt 2>> profile_1_oao.txt`  #OAO
   
   `nvprof --csv --print-gpu-trace ./main_out_manual.bin -cores 24 -boxes1d 15 1>> runtime_1.txt 2>> profile_1_manual.txt`  #manual

done

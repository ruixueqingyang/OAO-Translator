#!/bin/bash

times=$1 

# cd /home/anjushi/work/coding-6/1207

# make srad/srad_v2

cd /home/wfr/work/Coding/Rodinia_3.1/openmp/srad/srad_v2

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

FILE1="/home/wfr/work/Coding/Rodinia_3.1/openmp/srad/srad_v2/runtime_1.txt"
FILE5="/home/wfr/work/Coding/Rodinia_3.1/openmp/srad/srad_v2/profile_1_oao.txt"
FILE6="/home/wfr/work/Coding/Rodinia_3.1/openmp/srad/srad_v2/profile_1_manual.txt"

rm -rf $FILE1  $FILE5  $FILE6

make clean

make srad  #OMP
make EXPENSES=-DEXPENSES srad_out   #OAO
make OAO_manual   #manual

iter=16

for ((i=1; i<=$times; i ++))
do  
   `./srad 1024 1024 0 127 0 127 24 0.5 iter 1>> runtime_1.txt`
   `nvprof --csv --print-gpu-trace ./srad_out.bin 1024 1024 0 127 0 127 24 0.5 iter 1>> /dev/null 2>> profile_1_oao.txt`  #OAO
   `./srad_out.bin 1024 1024 0 127 0 127 24 0.5 iter  1>> runtime_1.txt`
   `nvprof --csv --print-gpu-trace ./OAO_manual.bin 1024 1024 0 127 0 127 24 0.5 iter  1>> /dev/null 2>> profile_1_manual.txt`  #manual
   `./OAO_manual.bin 1024 1024 0 127 0 127 24 0.5 iter  1>> runtime_1.txt`

done

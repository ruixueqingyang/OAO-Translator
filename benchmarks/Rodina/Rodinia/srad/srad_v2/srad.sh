#!/bin/bash

times=$1 

# cd /home/anjushi/work/coding-6/1207

# make srad/srad_v2

cd /home/anjushi/work/par/Rodinia_3.1/openmp/srad/srad_v2

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

FILE1="/home/anjushi/work/par/Rodinia_3.1/openmp/srad/srad_v2/runtime_1.txt"
FILE5="/home/anjushi/work/par/Rodinia_3.1/openmp/srad/srad_v2/profile_1_oao.txt"
FILE6="/home/anjushi/work/par/Rodinia_3.1/openmp/srad/srad_v2/profile_1_manual.txt"

rm -rf $FILE1  $FILE5  $FILE6

make clean

#make srad/srad_v2  #OMP
make EXPENSES=-DEXPENSES srad_out   #OAO
make OAO_manual   #manual

for ((i=1; i<=$times; i ++))
do  

   `nvprof --csv --print-gpu-trace ./srad_out.bin 1024 1024 0 127 0 127 24 0.5 8 1>> runtime_1.txt 2>> profile_1_oao.txt`  #OAO
   #时间变长？
   
   `nvprof --csv --print-gpu-trace ./OAO_manual.bin 1024 1024 0 127 0 127 24 0.5 8 1>> runtime_1.txt 2>> profile_1_manual.txt`  #manual

done

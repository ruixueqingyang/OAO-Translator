#!/bin/bash

cd /home/anjushi/work/polybench/plain/BICG

make clean

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

FILE="/home/anjushi/work/polybench/plain/BICG/result_1.txt"


make DEBUG=-D_DEBUG_1 OMP0  #OMP
make DAWN_1  #DawnCC
make DEBUG=-D_DEBUG_1 OAO   #OAO

cmdcu1=`./OMP`      #OMP
cmdcu2=`./DAWN_1.bin` #DawnCC_1
cmdcu3=`nvprof --csv --print-gpu-trace ./OAO.bin`  #OAO

aaa1=${cmdcu1}
aaa2=${cmdcu2}
aaa3=${cmdcu3}

echo -e "OMP\n""$aaa1\n""DAWN\n""$aaa2\n""OAO\n""$aaa3" > $FILE

#

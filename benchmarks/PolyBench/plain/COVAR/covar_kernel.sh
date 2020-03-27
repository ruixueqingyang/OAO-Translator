#!/bin/bash

cd /home/anjushi/work/polybench/plain/COVAR

make clean

make DEBUG=-D_DEBUG_1 OAOKERNEL  #OMP
make DAWNKERNEL  #DawnCC
make DEBUG=-D_DEBUG_1 OAO   #OAO
FILE="/home/anjushi/work/polybench/plain/COVAR/result_1.txt"


cmdcu1=`./OMP`      #OMP
cmdcu2=`./DAWN_1.bin` #DawnCC
cmdcu3=`./OAO.bin`  #OAO

aaa1=${cmdcu1}
aaa2=${cmdcu2}
aaa3=${cmdcu3}

echo -e "OMP\n""$aaa1\n""DAWN\n""$aaa2\n""OAO\n""$aaa3" > $FILE


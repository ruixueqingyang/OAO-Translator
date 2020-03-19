#!/bin/bash

cd /home/ghn/work/polybench/plain/SYRK

make clean

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

option=$1  
num1=1
num2=2

echo "option is ${option} !"

if [ $option == $num1 ]
then
   make DEBUG=-D_DEBUG_1 CPU0  #OMP
   make DEBUG=-D_DEBUG_1 OMP0  #OMP
   make DAWN_1  #DawnCC
   make DEBUG=-D_DEBUG_1 OAO   #OAO
   FILE="/home/ghn/work/polybench/plain/SYRK/result_1.txt"
elif [ $option == $num2 ]
then
   make DEBUG=-D_DEBUG_2 CPU0  #OMP
   make DEBUG=-D_DEBUG_2 OMP0  #OMP
   make DAWN_2  #DawnCC
   make DEBUG=-D_DEBUG_2 OAO   #OAO
   FILE="/home/ghn/work/polybench/plain/SYRK/result_2.txt"
else
   make DEBUG=-D_DEBUG_ CPU0  #OMP
   make DEBUG=-D_DEBUG_ OMP0  #OMP
   make DAWN  #DawnCC
   make DEBUG=-D_DEBUG_ OAO   #OAO
   FILE="/home/ghn/work/polybench/plain/SYRK/result_3.txt"
fi

cmdcu0=`./CPU`      #CPU
cmdcu1=`./OMP`      #OMP

if [ $option == $num1 ]
then
   cmdcu2=`./DAWN_1.bin` #DawnCC_1
elif [ $option == $num2 ]
then
   cmdcu2=`./DAWN_2.bin` #DawnCC_2
else
   cmdcu2=`./DAWN.bin` #DawnCC
fi

cmdcu3=`./OAO.bin`  #OAO

aaa0=${cmdcu0}
aaa1=${cmdcu1}
aaa2=${cmdcu2}
aaa3=${cmdcu3}

echo -e "SingleCPU\n""$aaa0\n""OMP\n""$aaa1\n""DAWN\n""$aaa2\n""OAO\n""$aaa3" > $FILE

#

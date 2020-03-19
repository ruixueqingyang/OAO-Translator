#!/bin/bash

cd /home/anjushi/work/polybench/plain/2MM

make clean

option=$1  
num1=1
num2=2

echo "option is ${option} !"

if [ $option == $num1 ]
then
   make DEBUG=-D_DEBUG_1 OMP0  #OMP
   make DAWN_1  #DawnCC
   make DEBUG=-D_DEBUG_1 OAO   #OAO
   FILE="/home/anjushi/work/polybench/plain/2MM/result_1.txt"
elif [ $option == $num2 ]
then
   make DEBUG=-D_DEBUG_2 OMP0  #OMP
   make DAWN_2  #DawnCC
   make DEBUG=-D_DEBUG_2 OAO   #OAO
   FILE="/home/anjushi/work/polybench/plain/2MM/result_2.txt"
else
   make DEBUG=-D_DEBUG_ OMP0  #OMP
   make DAWN  #DawnCC
   make DEBUG=-D_DEBUG_ OAO   #OAO
   FILE="/home/anjushi/work/polybench/plain/2MM/result_3.txt"
fi

cmdcu1=`./OMP`      #OMP

if [ $option == $num1 ]
then
   cmdcu2=`./DAWN_1.bin` #DawnCC
elif [ $option == $num2 ]
then
   cmdcu2=`./DAWN_2.bin` #DawnCC
else
   cmdcu2=`./DAWN.bin` #DawnCC
fi

cmdcu3=`./OAO.bin`  #OAO

aaa1=${cmdcu1}
aaa2=${cmdcu2}
aaa3=${cmdcu3}

echo -e "OMP\n""$aaa1\n""DAWN\n""$aaa2\n""OAO\n""$aaa3" > $FILE

#

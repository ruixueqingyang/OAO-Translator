#!/bin/bash

times=$1 

cd /home/anjushi/work/polybench/plain/BICG

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

FILE="/home/anjushi/work/polybench/plain/BICG/result_D3.txt"
FILE2="/home/anjushi/work/polybench/plain/BICG/result_D2.txt"
FILE1="/home/anjushi/work/polybench/plain/BICG/result_D1.txt"
rm -rf $FILE $FILE2 $FILE1

#最小数据量
make clean

make DAWNModify  #DawnC

for ((i=1; i<=$times; i ++))
do

  cmdcu21=`./DAWNModify.bin` #DawnCC

  aaa21=${cmdcu21}

  echo -e "DAWN\n""$aaa21" >> $FILE
done

#中等数据量
make clean

make DAWNModify2  #DawnCC

for ((i=1; i<=$times; i ++))
do
  cmdcu22=`./DAWNModify2.bin` #DawnCC_2

  aaa22=${cmdcu22}

  echo -e "DAWN\n""$aaa22" >> $FILE2
done

#最大数据量
make clean

make DAWNModify1  #DawnCC

for ((i=1; i<=$times; i ++))
do
  cmdcu23=`./DAWNModify1.bin` #DawnCC_1

  aaa23=${cmdcu23}

  echo -e "DAWN\n""$aaa23" >> $FILE1
done

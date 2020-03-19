#!/bin/bash

times=$1 

# cd /home/anjushi/work/coding/1119

# make atax

cd /home/anjushi/work/polybench/plain/ATAX

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

FILE="/home/anjushi/work/polybench/plain/ATAX/result_manual_3.txt"
FILE2="/home/anjushi/work/polybench/plain/ATAX/result_manual_2.txt"
FILE1="/home/anjushi/work/polybench/plain/ATAX/result_manual_1.txt"
rm -rf $FILE $FILE2 $FILE1

#最小数据量
make clean

make DEBUG=-D_DEBUG_ OAO_manual   #OAO

for ((i=1; i<=$times; i ++))
do

  cmdcu31=`./OAO_manual.bin`  #OAO

  aaa31=${cmdcu31}

  echo -e "OAO\n""$aaa31" >> $FILE

done

#中等数据量
make clean

make DEBUG=-D_DEBUG_2 OAO_manual   #OAO

for ((i=1; i<=$times; i ++))
do

  cmdcu32=`./OAO_manual.bin`  #OAO

  aaa32=${cmdcu32}

  echo -e "OAO\n""$aaa32" >> $FILE2

done

#最大数据量
make clean

make DEBUG=-D_DEBUG_1 OAO_manual   #OAO

for ((i=1; i<=$times; i ++))
do

  cmdcu33=`./OAO_manual.bin`  #OAO

  aaa33=${cmdcu33}

  echo -e "OAO\n""$aaa33" >> $FILE1
done

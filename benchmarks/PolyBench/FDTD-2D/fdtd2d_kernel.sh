#!/bin/bash

times=$1 

# cd /home/anjushi/work/coding/1119

# make fdtd2d

cd /home/anjushi/work/polybench/plain/FDTD-2D

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

FILE="/home/anjushi/work/polybench/plain/FDTD-2D/result_kernel_3.txt"
FILE2="/home/anjushi/work/polybench/plain/FDTD-2D/result_kernel_2.txt"
FILE1="/home/anjushi/work/polybench/plain/FDTD-2D/result_kernel_1.txt"
rm -rf $FILE $FILE2 $FILE1

#最小数据量
make clean

#make DEBUG=-D_DEBUG_ CPU0  #OMP
make DEBUG=-D_DEBUG_ OMP0  #OMP
make DAWNKERNEL  #DawnCC
make DEBUG=-D_DEBUG_ EXPENSES=-DEXPENSES OAOKERNEL   #OAO

for ((i=1; i<=$times; i ++))
do
#  cmdcu01=`./CPU`      #CPU
  cmdcu11=`./OMP`      #OMP
  cmdcu21=`./DAWNKERNEL.bin` #DawnCC
  cmdcu31=`./OAOKERNEL.bin`  #OAO

#  aaa01=${cmdcu01}
  aaa11=${cmdcu11}
  aaa21=${cmdcu21}
  aaa31=${cmdcu31}

  echo -e "OMP\n""$aaa11\n""DAWN\n""$aaa21\n""OAO\n""$aaa31" >> $FILE
#  "SingleCPU\n""$aaa01\n"
done

#中等数据量
make clean

#make DEBUG=-D_DEBUG_2 CPU0  #OMP
make DEBUG=-D_DEBUG_2 OMP0  #OMP
make DAWNKERNEL2  #DawnCC
make DEBUG=-D_DEBUG_2 EXPENSES=-DEXPENSES OAOKERNEL   #OAO

for ((i=1; i<=$times; i ++))
do
  #cmdcu02=`./CPU`      #CPU
  cmdcu12=`./OMP`      #OMP
  cmdcu22=`./DAWNKERNEL2.bin` #DawnCC_2
  cmdcu32=`./OAOKERNEL.bin`  #OAO

#  aaa02=${cmdcu02}
  aaa12=${cmdcu12}
  aaa22=${cmdcu22}
  aaa32=${cmdcu32}

  echo -e "OMP\n""$aaa12\n""DAWN\n""$aaa22\n""OAO\n""$aaa32" >> $FILE2
#  "SingleCPU\n""$aaa02\n"
done

#最大数据量
make clean

#make DEBUG=-D_DEBUG_1 CPU0  #OMP
make DEBUG=-D_DEBUG_1 OMP0  #OMP
make DAWNKERNEL1  #DawnCC
make DEBUG=-D_DEBUG_1 EXPENSES=-DEXPENSES OAOKERNEL   #OAO

for ((i=1; i<=$times; i ++))
do
#  cmdcu03=`./CPU`      #CPU
  cmdcu13=`./OMP`      #OMP
  cmdcu23=`./DAWNKERNEL1.bin` #DawnCC_1
  cmdcu33=`./OAOKERNEL.bin`  #OAO

#  aaa03=${cmdcu03}
  aaa13=${cmdcu13}
  aaa23=${cmdcu23}
  aaa33=${cmdcu33}

  echo -e "OMP\n""$aaa13\n""DAWN\n""$aaa23\n""OAO\n""$aaa33" >> $FILE1
#  "SingleCPU\n""$aaa03\n"
done

cd /home/anjushi/work/shell
./covar.sh $times
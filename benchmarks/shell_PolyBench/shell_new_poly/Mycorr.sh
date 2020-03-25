#!/bin/bash

times=$1 

# cd /home/anjushi/work/coding/1119

# make CORR

cd /home/wfr/work/Coding/OAO_2080Ti/plain/CORR

export LD_LIBRARY_PATH=/home/wfr/install/LLVM-9/install-9/lib:$LD_LIBRARY_PATH

RUNTIME_OMP_3="runtime_3_omp.txt"
RUNTIME_OAO_3="runtime_3_oao.txt"
RUNTIME_DAWNCC_3="runtime_3_dawncc.txt"
RUNTIME_MANUAL_3="runtime_3_manual.txt"
PROFILE_OAO_3="profile_3_oao.txt"
PROFILE_DAWNCC_3="profile_3_dawncc.txt"
PROFILE_MANUAL_3="profile_3_manual.txt"

RUNTIME_OMP_2="runtime_2_omp.txt"
RUNTIME_OAO_2="runtime_2_oao.txt"
RUNTIME_DAWNCC_2="runtime_2_dawncc.txt"
RUNTIME_MANUAL_2="runtime_2_manual.txt"
PROFILE_OAO_2="profile_2_oao.txt"
PROFILE_DAWNCC_2="profile_2_dawncc.txt"
PROFILE_MANUAL_2="profile_2_manual.txt"

RUNTIME_OMP_1="runtime_1_omp.txt"
RUNTIME_OAO_1="runtime_1_oao.txt"
RUNTIME_DAWNCC_1="runtime_1_dawncc.txt"
RUNTIME_MANUAL_1="runtime_1_manual.txt"
PROFILE_OAO_1="profile_1_oao.txt"
PROFILE_DAWNCC_1="profile_1_dawncc.txt"
PROFILE_MANUAL_1="profile_1_manual.txt"

#三种数据量
FLAG_3=1
FLAG_2=1
FLAG_1=1

#运行时间 / 传输分析
FLAG_RUNTIME=1
FLAG_PROFILE=1

#不同程序版本
FLAG_OMP=0
FLAG_OAO=1
FLAG_DAWNCC=1
FLAG_MANUAL=1

make clean

#最小数据量
if (($FLAG_3 != 0)); then
  echo "Date Size 3"
  #OAO
  if (($FLAG_OAO != 0)); then
    echo "OAO"
    make DEBUG=-D_DEBUG_ EXPENSES=-DEXPENSES OAO

    if (($FLAG_RUNTIME != 0)); then
      echo "$RUNTIME_OAO_3"
      rm $RUNTIME_OAO_3
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        ./OAO.bin 1>>$RUNTIME_OAO_3
      done
    fi
    if (($FLAG_PROFILE != 0)); then
      echo "$PROFILE_OAO_3"
      rm $PROFILE_OAO_3
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAO.bin 1>>/dev/null 2>>$PROFILE_OAO_3
      done
    fi
  fi

  #DawnCC
  if (($FLAG_DAWNCC != 0)); then
    echo "DawnCC"
    make DAWN

    if (($FLAG_RUNTIME != 0)); then
      echo "$RUNTIME_DAWNCC_3"
      rm $RUNTIME_DAWNCC_3
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        ./DAWN.bin 1>>$RUNTIME_DAWNCC_3
      done
    fi
    if (($FLAG_PROFILE != 0)); then
      echo "$PROFILE_DAWNCC_3"
      rm $PROFILE_DAWNCC_3
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        nvprof --unified-memory-profiling off --csv --print-gpu-trace ./DAWN.bin 1>>/dev/null 2>>$PROFILE_DAWNCC_3
      done
    fi
  fi

  #MANUAL
  if (($FLAG_MANUAL != 0)); then
    echo "MANUAL"
    make DEBUG=-D_DEBUG_ OAO_manual

    if (($FLAG_RUNTIME != 0)); then
      echo "$RUNTIME_MANUAL_3"
      rm $RUNTIME_MANUAL_3
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        ./OAO_manual.bin 1>>$RUNTIME_MANUAL_3
      done
    fi
    if (($FLAG_PROFILE != 0)); then
      echo "$PROFILE_MANUAL_3"
      rm $PROFILE_MANUAL_3
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAO_manual.bin 1>>/dev/null 2>>$PROFILE_MANUAL_3
      done
    fi
  fi

  #OMP
  if (($FLAG_OMP != 0)) && (($FLAG_RUNTIME != 0)); then
    echo "OMP"
    make DEBUG=-D_DEBUG_ OMP
    echo "$RUNTIME_OMP_3"
    rm $RUNTIME_OMP_3
    for ((i = 1; i <= $times; i++)); do
      echo "$i"
      ./OMP.bin 1>>$RUNTIME_OMP_3
    done
  fi

fi

#中等数据量
if (($FLAG_2 != 0)); then
  echo "Data Size 2"

  #OAO
  if (($FLAG_OAO != 0)); then
    echo "OAO"
    make DEBUG=-D_DEBUG_2 EXPENSES=-DEXPENSES OAO

    if (($FLAG_RUNTIME != 0)); then
      echo "$RUNTIME_OAO_2"
      rm $RUNTIME_OAO_2
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        ./OAO.bin 1>>$RUNTIME_OAO_2
      done
    fi
    if (($FLAG_PROFILE != 0)); then
      echo "$PROFILE_OAO_2"
      rm $PROFILE_OAO_2
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAO.bin 1>>/dev/null 2>>$PROFILE_OAO_2
      done
    fi
  fi

  #DawnCC
  if (($FLAG_DAWNCC != 0)); then
    echo "DawnCC"
    make DAWN_2

    if (($FLAG_RUNTIME != 0)); then
      echo "$RUNTIME_DAWNCC_2"
      rm $RUNTIME_DAWNCC_2
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        ./DAWN_2.bin 1>>$RUNTIME_DAWNCC_2
      done
    fi
    if (($FLAG_PROFILE != 0)); then
      echo "$PROFILE_DAWNCC_2"
      rm $PROFILE_DAWNCC_2
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        nvprof --unified-memory-profiling off --csv --print-gpu-trace ./DAWN_2.bin 1>>/dev/null 2>>$PROFILE_DAWNCC_2
      done
    fi
  fi

  #MANUAL
  if (($FLAG_MANUAL != 0)); then
    echo "MANUAL"
    make DEBUG=-D_DEBUG_2 OAO_manual

    if (($FLAG_RUNTIME != 0)); then
      echo "$RUNTIME_MANUAL_2"
      rm $RUNTIME_MANUAL_2
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        ./OAO_manual.bin 1>>$RUNTIME_MANUAL_2
      done
    fi
    if (($FLAG_PROFILE != 0)); then
      echo "$PROFILE_MANUAL_2"
      rm $PROFILE_MANUAL_2
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAO_manual.bin 1>>/dev/null 2>>$PROFILE_MANUAL_2
      done
    fi
  fi

  #OMP
  if (($FLAG_OMP != 0)) && (($FLAG_RUNTIME != 0)); then
    echo "OMP"
    make DEBUG=-D_DEBUG_2 OMP
    echo "$RUNTIME_OMP_2"
    rm $RUNTIME_OMP_2
    for ((i = 1; i <= $times; i++)); do
      echo "$i"
      ./OMP.bin 1>>$RUNTIME_OMP_2
    done
  fi

fi

#最大数据量
if (($FLAG_1 != 0)); then
  echo "Data Size 1"
  #OAO
  if (($FLAG_OAO != 0)); then
    echo "OAO"
    make DEBUG=-D_DEBUG_1 EXPENSES=-DEXPENSES OAO

    if (($FLAG_RUNTIME != 0)); then
      echo "$RUNTIME_OAO_1"
      rm $RUNTIME_OAO_1
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        ./OAO.bin 1>>$RUNTIME_OAO_1
      done
    fi
    if (($FLAG_PROFILE != 0)); then
      echo "$PROFILE_OAO_1"
      rm $PROFILE_OAO_1
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAO.bin 1>>/dev/null 2>>$PROFILE_OAO_1
      done
    fi
  fi

  #DawnCC
  if (($FLAG_DAWNCC != 0)); then
    echo "DawnCC"
    make DAWN_1

    if (($FLAG_RUNTIME != 0)); then
      echo "$RUNTIME_DAWNCC_1"
      rm $RUNTIME_DAWNCC_1
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        ./DAWN_1.bin 1>>$RUNTIME_DAWNCC_1
      done
    fi
    if (($FLAG_PROFILE != 0)); then
      echo "$PROFILE_DAWNCC_1"
      rm $PROFILE_DAWNCC_1
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        nvprof --unified-memory-profiling off --csv --print-gpu-trace ./DAWN_1.bin 1>>/dev/null 2>>$PROFILE_DAWNCC_1
      done
    fi
  fi

  #MANUAL
  if (($FLAG_MANUAL != 0)); then
    echo "MANUAL"
    make DEBUG=-D_DEBUG_1 OAO_manual

    if (($FLAG_RUNTIME != 0)); then
      echo "$RUNTIME_MANUAL_1"
      rm $RUNTIME_MANUAL_1
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        ./OAO_manual.bin 1>>$RUNTIME_MANUAL_1
      done
    fi
    if (($FLAG_PROFILE != 0)); then
      echo "$PROFILE_MANUAL_1"
      rm $PROFILE_MANUAL_1
      for ((i = 1; i <= $times; i++)); do
        echo "$i"
        nvprof --unified-memory-profiling off --csv --print-gpu-trace ./OAO_manual.bin 1>>/dev/null 2>>$PROFILE_MANUAL_1
      done
    fi
  fi

  #OMP
  if (($FLAG_OMP != 0)) && (($FLAG_RUNTIME != 0)); then
    echo "OMP"
    make DEBUG=-D_DEBUG_1 OMP
    echo "$RUNTIME_OMP_1"
    rm $RUNTIME_OMP_1
    for ((i = 1; i <= $times; i++)); do
      echo "$i"
      ./OMP.bin 1>>$RUNTIME_OMP_1
    done
  fi

fi

make clean

#!/bin/bash

if [ ! -d $CLANG_INSTALL ] || [ -z $CLANG_INSTALL ]; then

    CLANG_INSTALL=/LLVM_INSTALLATION_PATH

    USER=$(whoami)
    
    if [ $USER == "wfr" ]; then
        CLANG_INSTALL=/home/wfr/install/LLVM-9/install-9
    elif [ $USER == "ghn" ]; then
        CLANG_INSTALL=/home/ghn/work/llvm-project/install
    fi

    echo $CLANG_INSTALL
fi

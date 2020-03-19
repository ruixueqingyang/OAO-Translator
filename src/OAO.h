/*******************************************************************************
Copyright(C), 2010-2019, 瑞雪轻飏
     FileName: OAO.h
       Author: 瑞雪轻飏
      Version: 0.02
Creation Date: 20181124
  Description: 用来包含 OAO 所需的所有 .h 文件
       Others: //其他内容说明
Function List: //主要函数列表, 每条记录应包含函数名及功能简要说明
    1.…………
    2.…………
History:  //修改历史记录列表, 每条修改记录应包含修改日期、修改者及修改内容简介
    1.Date:
    Author:
    Modification:
    2.…………
*******************************************************************************/

#ifndef OAO_H
#define OAO_H

// #define MY_LLVM_INSTALL_PATH    /home/wfr/MyLLVM/install

#include "OAORewriter.h"
#include <iostream>
#include <fstream>

int FileCopy(std::string dest, std::string src);

/*void DebugPrint(){
    std::cout << "\nDebugPrint: " << std::endl;
    for(int i=0; i<(int)vectorVar.size(); i++){
        if(vectorVar[i].Name == "v9"){
            std::cout << "VARIABLE_INFO: " << std::endl;
            std::cout << "Name: " << vectorVar[i].Name << std::endl;
            std::cout << "ptrDecl: 0x" << std::hex << (long long)vectorVar[i].ptrDecl << std::dec << std::endl;
            std::cout << "TypeName: " << vectorVar[i].TypeName << std::endl;
            std::cout << "isMember: " << vectorVar[i].isMember << std::endl;
            std::cout << "isArrow: " << vectorVar[i].isArrow << std::endl;
            std::cout << "ptrClass: 0x" << std::hex << (long long)vectorVar[i].ptrClass << std::dec << std::endl;
            std::cout << "indexClass: " << vectorVar[i].indexClass << std::endl;
            std::cout << "BeginLine: " << vectorVar[i].BeginLine << std::endl;
            std::cout << "BeginCol: " << vectorVar[i].BeginCol << std::endl;
        }
    }
}*/

#endif // OAO_H

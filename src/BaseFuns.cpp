/*******************************************************************************
Copyright(C), 1992-2019, 瑞雪轻飏
     FileName: BaseFuns.cpp
       Author: 瑞雪轻飏
      Version: 0.01
Creation Date: 20190815
  Description: 实现 一些通用的函数, 只能用 Clang/LLVM 提供的数据类型, 不能用我定义的数据类型
       Others: 是一些底层通用函数, 在 Clang/LLVM 之上, 在我定义的类之下
Function List: //主要函数列表, 每条记录应包含函数名及功能简要说明
    1. main: 主函数
    2.…………
History:  //修改历史记录列表, 每条修改记录应包含修改日期、修改者及修改内容简介
    1.Date:
    Author:
    Modification:
    2.…………
*******************************************************************************/

#include "Clang_LLVM_Inc.h"

using namespace clang;

// wfr 20190718 修补 Stmt等 的结尾位置偏移, 因为 clang AST 中的原始位置可能不准, 例如不包括 ";"
// wfr 20190815 重新定位函数调用等的结尾偏移, 定位到 ";"
SourceLocation fixStmtLoc(SourceLocation OriginEnd, Rewriter& Rewrite){
    SourceLocation TmpLoc;
    if(OriginEnd==TmpLoc){ // 如果位置为空就返回
        return OriginEnd;
    }
    // wfr 20190718 重新定位函数调用等的结尾偏移, 定位到 ";"
    // ghn 20190715 检索源文件
    TmpLoc = OriginEnd.getLocWithOffset(1); // 偏移到 之后的第一个字符
    FileID Fileid = Rewrite.getSourceMgr().getFileID(TmpLoc);
    StringRef filename = Rewrite.getSourceMgr().getFilename(TmpLoc);
    const FileEntry *File = Rewrite.getSourceMgr().getFileEntryForID(Fileid);
    const llvm::MemoryBuffer *buffer = Rewrite.getSourceMgr().getMemoryBufferForFile(File);
    // const char* start = buffer->getBufferStart();
    // const char* end = buffer->getBufferEnd();
    size_t size = buffer->getBufferSize();
    // StringRef ref = buffer->getBuffer();
    int off = Rewrite.getSourceMgr().getFileOffset(TmpLoc); // 偏移到 之后的第一个字符
    int Mapsize = size-off;//检索区域大小(size-off)
    // ghn 20190715 指定位置开始检索
    llvm::ErrorOr <std::unique_ptr<llvm::MemoryBuffer>> range = buffer->getFileSlice(filename, Mapsize, off, false);
    const char* start = range.get()->getBufferStart(); // 检索开始位置
    
    // wfr 20190718 匹配分号
    bool signal = false;
    int i=0;
    for(; i<Mapsize; ++i){
        if(start[i] == ';'){
            signal = true;
            break;
        }else if(start[i] == '\n'){
            break; // 只在当前行判断
        }else{}
    }

    if(signal==true){
        SourceLocation FixedLoc = OriginEnd.getLocWithOffset(1+i);
        return FixedLoc;
    }else{
        // std::cout << "fixStmtLoc 警告: 重新定位 Stmt 结尾偏移失败" << std::endl;
        return OriginEnd;
    }
}
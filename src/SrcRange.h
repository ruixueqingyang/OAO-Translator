/***************************************************************************
Copyright(C), 2010-2019, 瑞雪轻飏
     FileName: SrcRange.h
       Author: 瑞雪轻飏
      Version: 0.1
Creation Date: 20181216
  Description: 用来描述源代码在源文件中的位置
       Others: //其他内容说明
Function List: //主要函数列表, 每条记录应包含函数名及功能简要说明
    1.…………
    2.…………
History:  //修改历史记录列表, 每条修改记录应包含修改日期、修改者及修改内容简介
    1.Date:
    Author:
    Modification:
    2.…………
***************************************************************************/

#ifndef MY_SOURCE_RANGE_H
#define MY_SOURCE_RANGE_H

#include "Clang_LLVM_Inc.h"

using namespace clang;

inline bool operator<=(const SourceLocation &LHS, const SourceLocation &RHS) {
    return LHS.getRawEncoding() <= RHS.getRawEncoding();
}

inline bool operator>(const SourceLocation &LHS, const SourceLocation &RHS) {
    return LHS.getRawEncoding() > RHS.getRawEncoding();
}

inline bool operator>=(const SourceLocation &LHS, const SourceLocation &RHS) {
    return LHS.getRawEncoding() >= RHS.getRawEncoding();
}

class   MY_SOURCE_RANGE{ // 该类用来保存源代码的位置信息
public:
    //const Stmt*   ptrBegin; // 源代码对应的 AST 节点的指针
    //const Stmt*   ptrEnd; // 源代码对应的 AST 节点的指针
    SourceLocation BeginLoc; // SourceLocation 类型表示的源代码中字符的位置
    SourceLocation EndLoc; // SourceLocation 类型表示的源代码中字符的位置
    unsigned int   BeginLine; // 在源码中的开始行号, 这四个是行号/列号
    unsigned int   BeginCol; // 包含该列的字符
    unsigned int   EndLine;
    unsigned int   EndCol; // 包含该列的字符, 即代码范围是一个闭集合, 包含两端的字符
    unsigned int   BeginOffset; // 描述代码起始位置在文件中的偏移
    unsigned int   EndOffset; // 描述代码结束位置在文件中的偏移
    
    //int FileID;

    // wfr 20191119 修补 节点范围, 排除 空格/回车/换行/tab 等空白字符
    // wfr 20191119 MaxOffset 表示允许修改的最大偏移量
    int FixRange(unsigned int MaxOffset, Rewriter& Rewrite){
        SourceLocation EmptyLoc;
        
        // 修补 BeginLoc
        if(BeginLoc!=EmptyLoc){
            int Offset = 0;
            FileID MyFileID = Rewrite.getSourceMgr().getFileID(BeginLoc);
            // StringRef filename = Rewrite.getSourceMgr().getFilename(BeginLoc);
            const FileEntry *File = Rewrite.getSourceMgr().getFileEntryForID(MyFileID);
            // 获得源代码的字符串 buffer
            const llvm::MemoryBuffer *buffer = Rewrite.getSourceMgr().getMemoryBufferForFile(File);
            const char* start = buffer->getBufferStart();
            const char* cursor = start + BeginOffset; // 游标设置到节点开始位置
            for(int i=0; i<MaxOffset; i++){ // 最多只从开始位置向后检查 MaxOffset 个字符
                unsigned int character = (unsigned int)(*cursor);
                // 是 空格/分号/回车/换行/制表符
                if(*cursor==' ' || *cursor==';' || character==0x0D || character==0x0A || character==0x09 ){
                    ++Offset;
                }else{
                    break;
                }
                ++cursor;
            }
            if(Offset>0){
                SourceLocation TmpLoc = BeginLoc.getLocWithOffset(Offset);
                SetBeginLoc(TmpLoc, Rewrite);
            }
        }

        // 修补 EndLoc
        if(EndLoc!=EmptyLoc){
            int Offset = 0;
            FileID MyFileID = Rewrite.getSourceMgr().getFileID(EndLoc);
            // StringRef filename = Rewrite.getSourceMgr().getFilename(EndLoc);
            const FileEntry *File = Rewrite.getSourceMgr().getFileEntryForID(MyFileID);
            // 获得源代码的字符串 buffer
            const llvm::MemoryBuffer *buffer = Rewrite.getSourceMgr().getMemoryBufferForFile(File);
            const char* start = buffer->getBufferStart();
            const char* cursor = start + EndOffset; // 游标设置到节点结束位置
            for(int i=0; i<MaxOffset; i++){ // 最多只从结束位置向前检查 MaxOffset 个字符
                unsigned int character = (unsigned int)(*cursor);
                // 是 空格/回车/换行/制表符
                if(*cursor==' ' || character==0x0D || character==0x0A || character==0x09 ){
                    --Offset;
                }else{
                    break;
                }
                --cursor;
            }
            if(Offset<0){
                SourceLocation TmpLoc = EndLoc.getLocWithOffset(Offset);
                SetEndLoc(TmpLoc, Rewrite);
            }
        }

        if(BeginOffset>EndOffset){ // wfr 20191119 如果起始偏移比结束偏移大了, 就交换起始和结束
            SourceLocation TmpBeginLoc = EndLoc;
            SourceLocation TmpEndLoc = BeginLoc;

            SetBeginLoc(TmpBeginLoc, Rewrite);
            SetEndLoc(TmpEndLoc, Rewrite);
        }

        return 0;
    }

    int SetBeginLoc(const SourceLocation in_BeginLoc, Rewriter Rewrite){
        BeginLoc = Rewrite.getSourceMgr().getExpansionLoc(in_BeginLoc);
        BeginLine = Rewrite.getSourceMgr().getSpellingLineNumber(BeginLoc);
        BeginCol = Rewrite.getSourceMgr().getSpellingColumnNumber(BeginLoc);
        BeginOffset = Rewrite.getSourceMgr().getFileOffset(BeginLoc);

        return 0;
    }

    int SetEndLoc(const SourceLocation in_EndLoc, Rewriter Rewrite){
        EndLoc = Rewrite.getSourceMgr().getExpansionLoc(in_EndLoc);
        EndLine = Rewrite.getSourceMgr().getSpellingLineNumber(EndLoc);
        EndCol = Rewrite.getSourceMgr().getSpellingColumnNumber(EndLoc);
        EndOffset = Rewrite.getSourceMgr().getFileOffset(EndLoc);

        return 0;
    }

    void init(const SourceRange in_SourceRange, Rewriter& Rewrite){
        init(in_SourceRange.getBegin(), in_SourceRange.getEnd(), Rewrite);
    }
    void init(const SourceLocation in_BeginLoc, const SourceLocation in_EndLoc, Rewriter& Rewrite){
        BeginLoc = Rewrite.getSourceMgr().getExpansionLoc(in_BeginLoc);
        EndLoc = Rewrite.getSourceMgr().getExpansionLoc(in_EndLoc);
        BeginLine = Rewrite.getSourceMgr().getSpellingLineNumber(BeginLoc);
        BeginCol = Rewrite.getSourceMgr().getSpellingColumnNumber(BeginLoc);
        EndLine = Rewrite.getSourceMgr().getSpellingLineNumber(EndLoc);
        EndCol = Rewrite.getSourceMgr().getSpellingColumnNumber(EndLoc);
        BeginOffset = Rewrite.getSourceMgr().getFileOffset(BeginLoc);
        EndOffset = Rewrite.getSourceMgr().getFileOffset(EndLoc);
        //std::cout << "离开 MY_SOURCE_RANGE::init" << std::endl;
    }
    void init(const SourceLocation in_BeginLoc, const SourceLocation in_EndLoc, size_t offset, Rewriter& Rewrite){
        //ptrBegin = ptr;
        //ptrEnd = ptr;
        //BeginLoc = ptrBegin->getBeginLoc();
        //EndLoc = ptrEnd->getEndLoc().getLocWithOffset(offset);
        BeginLoc = Rewrite.getSourceMgr().getExpansionLoc(in_BeginLoc);
        EndLoc = Rewrite.getSourceMgr().getExpansionLoc(in_EndLoc);
        EndLoc = EndLoc.getLocWithOffset(offset);
        BeginLine = Rewrite.getSourceMgr().getSpellingLineNumber(BeginLoc);
        BeginCol = Rewrite.getSourceMgr().getSpellingColumnNumber(BeginLoc);
        EndLine = Rewrite.getSourceMgr().getSpellingLineNumber(EndLoc);
        EndCol = Rewrite.getSourceMgr().getSpellingColumnNumber(EndLoc);
        BeginOffset = Rewrite.getSourceMgr().getFileOffset(BeginLoc);
        EndOffset = Rewrite.getSourceMgr().getFileOffset(EndLoc);
    }

    void reset(){
        SourceLocation TmpLoc;
        BeginLoc = TmpLoc;
        EndLoc = TmpLoc;

        BeginLine = 0;
        BeginCol = 0;
        EndLine = 0;
        EndCol = 0;
        BeginOffset = 0;
        EndOffset = 0;
    }

    void init(const MY_SOURCE_RANGE& in_SrcRange){
        BeginLoc = in_SrcRange.BeginLoc;
        EndLoc = in_SrcRange.EndLoc;

        BeginLine = in_SrcRange.BeginLine;
        BeginCol = in_SrcRange.BeginCol;
        EndLine = in_SrcRange.EndLine;
        EndCol = in_SrcRange.EndCol;
        BeginOffset = in_SrcRange.BeginOffset;
        EndOffset = in_SrcRange.EndOffset;
        //FileID = in_SrcRange.FileID;
    }
    
    const MY_SOURCE_RANGE& operator=(const MY_SOURCE_RANGE& in_SrcRange){

        init(in_SrcRange);

        return in_SrcRange;
    }
    MY_SOURCE_RANGE(const MY_SOURCE_RANGE& in_SrcRange){
        init(in_SrcRange);
    }
    MY_SOURCE_RANGE(const SourceLocation in_BeginLoc, const SourceLocation in_EndLoc, Rewriter& Rewrite){
        init(in_BeginLoc, in_EndLoc, Rewrite);
    }
    MY_SOURCE_RANGE(const SourceLocation in_BeginLoc, const SourceLocation in_EndLoc, size_t offset, Rewriter& Rewrite){
        init(in_BeginLoc, in_EndLoc, offset, Rewrite);
    }
    MY_SOURCE_RANGE(const SourceRange in_SourceRange, Rewriter& Rewrite){
        init(in_SourceRange.getBegin(), in_SourceRange.getEnd(), Rewrite);
    }
    MY_SOURCE_RANGE(){
        reset();
    }
};

#endif // MY_SOURCE_RANGE_H

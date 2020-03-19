/*******************************************************************************
Copyright(C), 1992-2019, 瑞雪轻飏
     FileName: Clang_LLVM_Inc.h
       Author: 瑞雪轻飏
      Version: 0.01
Creation Date: 20190131
  Description: 用于包含 Clang/LLVM 的各种头文件, 以及标准 C++ 的各种头文件
Function List: //主要函数列表, 每条记录应包含函数名及功能简要说明
    1.…………
    2.…………
History:  //修改历史记录列表, 每条修改记录应包含修改日期、修改者及修改内容简介
    1.Date:
    Author:
    Modification:
    2.…………
*******************************************************************************/

#ifndef CLANG_LLVM_INC_H
#define CLANG_LLVM_INC_H

#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <system_error>
#include <string.h>
#include <vector>
#include <assert.h>

#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"

#include "clang/Analysis/CFG.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/LangStandard.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Lex/Lexer.h"
#include "clang/Basic/Diagnostic.h"
// wfr 20190328 替换成下边修改过的头文件 #include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/RecursiveASTVisitorProtected.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Type.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CommentVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclLookups.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/LocInfoType.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Module.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Rewrite/Frontend/Rewriters.h"
#include "clang/Rewrite/Core/Rewriter.h"

using namespace clang;

// wfr 20190815 重新定位函数调用等的结尾偏移, 定位到 ";"
SourceLocation fixStmtLoc(SourceLocation OriginEnd, Rewriter& Rewrite);

#endif // CLANG_LLVM_INC_H